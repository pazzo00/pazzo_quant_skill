"""Refresh BAB/QMJ stock signals and then update the six factor premia."""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import factor_pipeline  # noqa: E402
from runtime_paths import project_root  # noqa: E402
from vendor.quarter_data import cal_quarter_data  # noqa: E402
from vendor.report_data_loader import ReportDataLoader  # noqa: E402


PROJECT_ROOT = project_root()


DEFAULTS = {
    "stock_bar": PROJECT_ROOT / "data" / "stock" / "stock_bar_1day.parquet",
    "capital": PROJECT_ROOT / "data" / "stock" / "capital.parquet",
    "market_bar": PROJECT_ROOT / "data" / "index" / "index_bar_1day.parquet",
    "fund_bar": PROJECT_ROOT / "data" / "index" / "index_bar_1day.parquet",
    "report_balance": PROJECT_ROOT / "data" / "stock" / "report_balance.parquet",
    "report_income": PROJECT_ROOT / "data" / "stock" / "report_income.parquet",
    "report_cashflow": PROJECT_ROOT / "data" / "stock" / "report_cashflow.parquet",
    "bab_factor": PROJECT_ROOT / "daily_factors" / "BAB" / "bab.parquet",
    "qmj_factor": PROJECT_ROOT / "daily_factors" / "QMJ" / "qmj.parquet",
    "premium": PROJECT_ROOT / "fund_return_analysis" / "premium.parquet",
}
SOURCE_COLUMNS = {
    "stock_bar": {"datetime", "code", "close", "factor"},
    "capital": {"datetime", "code", "market_cap", "pb_ratio"},
    "market_bar": {"datetime", "code", "close", "factor"},
    "fund_bar": {"datetime", "code", "close", "factor"},
    "report_balance": {"datetime", "code", "report_period"},
    "report_income": {"datetime", "code", "report_period"},
    "report_cashflow": {"datetime", "code", "report_period"},
}


def _inclusive_end(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1) if timestamp == timestamp.normalize() else timestamp


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _date_bounds(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    parquet = pq.ParquetFile(path)
    column_index = parquet.schema_arrow.names.index("datetime")
    minima, maxima = [], []
    for row_group in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row_group).column(column_index).statistics
        if stats is not None and stats.has_min_max:
            minima.append(pd.Timestamp(stats.min))
            maxima.append(pd.Timestamp(stats.max))
    if minima:
        return min(minima), max(maxima)
    dates = pd.read_parquet(path, columns=["datetime"])["datetime"]
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def _load_config(config_path: str | None) -> dict[str, Path]:
    values = {key: Path(value) for key, value in DEFAULTS.items()}
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"数据路径配置不存在: {path}")
        configured = json.loads(path.read_text(encoding="utf-8"))
        for key, raw in configured.items():
            if key in values:
                candidate = Path(raw)
                values[key] = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return values


def _validate_sources(paths: dict[str, Path]) -> list[dict]:
    problems = []
    for key, required in SOURCE_COLUMNS.items():
        path = paths[key]
        if not path.exists():
            problems.append({"key": key, "path": str(path), "problem": "missing"})
            continue
        if path.suffix.lower() != ".parquet":
            problems.append({"key": key, "path": str(path), "problem": "prepare 当前要求 Parquet"})
            continue
        columns = set(pq.read_schema(path).names)
        missing = sorted(required - columns)
        if missing:
            problems.append({"key": key, "path": str(path), "problem": f"missing_columns:{missing}"})
    return problems


def _market_end(path: Path, market_code: str) -> pd.Timestamp:
    dates = pd.read_parquet(path, columns=["datetime", "code"], filters=[("code", "==", market_code)])
    if dates.empty:
        raise ValueError(f"市场行情中找不到 {market_code}")
    return pd.Timestamp(dates["datetime"].max())


def _trade_days(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    dates = pd.read_parquet(path, columns=["datetime"], filters=[("datetime", ">=", start), ("datetime", "<=", end)])
    return pd.DatetimeIndex(pd.to_datetime(dates["datetime"].unique())).sort_values()


def _missing_factor_dates(path: Path, expected: pd.DatetimeIndex, value_column: str) -> pd.DatetimeIndex:
    if not path.exists():
        return expected
    schema = set(pq.read_schema(path).names)
    required = {"datetime", "code", value_column}
    if not required.issubset(schema):
        raise ValueError(f"{path} 缺少列 {sorted(required - schema)}")
    if expected.empty:
        return expected
    frame = pd.read_parquet(
        path,
        columns=["datetime", value_column],
        filters=[("datetime", ">=", expected.min()), ("datetime", "<=", expected.max())],
    )
    available = pd.DatetimeIndex(frame.loc[frame[value_column].notna(), "datetime"].unique())
    return expected.difference(available)


def _adjusted_returns(path: Path, start: pd.Timestamp, end: pd.Timestamp, codes: list[str] | None = None) -> pd.DataFrame:
    filters = [("datetime", ">=", start), ("datetime", "<=", end)]
    if codes:
        filters.append(("code", "in", codes))
    frame = pd.read_parquet(path, columns=["datetime", "code", "close", "factor"], filters=filters)
    frame["adjusted_price"] = frame["close"] * frame["factor"]
    prices = frame.drop_duplicates(["datetime", "code"], keep="last").pivot(index="datetime", columns="code", values="adjusted_price").sort_index()
    return prices.pct_change(fill_method=None)


def refresh_bab(paths: dict[str, Path], missing_dates: pd.DatetimeIndex, market_code: str) -> dict:
    if missing_dates.empty:
        return {"status": "up_to_date", "rows_added": 0}
    load_start = missing_dates.min() - pd.Timedelta(days=600)
    stock_returns = _adjusted_returns(paths["stock_bar"], load_start, missing_dates.max())
    market_returns = _adjusted_returns(paths["market_bar"], load_start, missing_dates.max(), [market_code])[market_code]
    market = pd.concat({"mkt": market_returns, **{f"lag{lag}": market_returns.shift(lag) for lag in range(1, 6)}}, axis=1).dropna()
    stock_returns = stock_returns.reindex(market.index)
    rows = []
    for day in missing_dates:
        eligible = market.index[market.index <= day]
        if len(eligible) < 252:
            continue
        window_days = eligible[-252:]
        x = market.loc[window_days].to_numpy(dtype=float)
        y = stock_returns.loc[window_days].to_numpy(dtype=float)
        values = np.full(y.shape[1], np.nan)
        complete_columns = np.isfinite(y).all(axis=0)
        x_centered = x - x.mean(axis=0)
        if complete_columns.any():
            y_complete = y[:, complete_columns]
            coefficients = np.linalg.lstsq(x_centered, y_complete - y_complete.mean(axis=0), rcond=None)[0]
            values[complete_columns] = coefficients.mean(axis=0)
        for column in np.flatnonzero(~complete_columns):
            valid = np.isfinite(y[:, column])
            if valid.sum() < 200:
                continue
            local_x = x[valid]
            local_y = y[valid, column]
            coefficients = np.linalg.lstsq(local_x - local_x.mean(axis=0), local_y - local_y.mean(), rcond=None)[0]
            values[column] = coefficients.mean()
        row = pd.DataFrame({"datetime": day, "code": stock_returns.columns, "bab": 0.5 * values + 0.5})
        rows.append(row)
    if not rows:
        raise ValueError("BAB 缺失日期无法得到至少 252 个市场收益观测")
    added = pd.concat(rows, ignore_index=True)
    output = paths["bab_factor"]
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame(columns=added.columns)
    combined = pd.concat([existing, added], ignore_index=True).sort_values(["datetime", "code"]).drop_duplicates(["datetime", "code"], keep="last")
    _atomic_write(combined, output)
    return {"status": "updated", "dates_added": int(added["datetime"].nunique()), "rows_added": len(added), "output": str(output)}


def _cross_section_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1).replace(0, np.nan), axis=0)


def _load_report(
    path: Path,
    fields: list[str],
    lag: str,
    trade_days: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    codes=None,
):
    return ReportDataLoader.load_data(
        str(path), start=start, end=end, codes=codes, fields=fields, lag=lag, trade_days=trade_days.tolist()
    )


def _calculate_zscore(paths: dict[str, Path], trade_days: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    balance = _load_report(paths["report_balance"], [
        "total_assets", "total_current_assets", "total_current_liability", "total_liability",
        "total_owner_equities", "surplus_reserve_fund", "retained_profit",
    ], "0Q", trade_days, start, end)
    income = _load_report(paths["report_income"], [
        "operating_revenue", "financial_expense", "net_profit", "income_tax",
    ], "1Q", trade_days, start, end, balance.codes)
    season = income.season_data.values == 1
    revenue = cal_quarter_data(income.data, season, 0, 4, income.season_data.values)
    financial_expense = cal_quarter_data(income.data, season, 1, 4, income.season_data.values)
    net_profit = cal_quarter_data(income.data, season, 2, 4, income.season_data.values)
    income_tax = cal_quarter_data(income.data, season, 3, 4, income.season_data.values)
    assets = balance.data[:, 0, :]
    working_capital = balance.data[:, 1, :] - balance.data[:, 2, :]
    liabilities = balance.data[:, 3, :]
    equity = balance.data[:, 4, :]
    retained = balance.data[:, 5, :] + balance.data[:, 6, :]
    ebit = financial_expense + net_profit + income_tax
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 1.2 * working_capital / assets + 1.4 * retained / assets + 3.3 * ebit / assets + 0.6 * equity / liabilities + 0.999 * revenue / assets
    return pd.DataFrame(values, index=trade_days, columns=balance.codes).replace([np.inf, -np.inf], np.nan)


def _calculate_oscore(paths: dict[str, Path], trade_days: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    balance = _load_report(paths["report_balance"], [
        "total_assets", "total_current_assets", "total_current_liability", "total_liability",
    ], "0Q", trade_days, start, end)
    income = _load_report(paths["report_income"], ["operating_revenue", "net_profit"], "2Q", trade_days, start, end, balance.codes)
    season = income.season_data.values
    first_quarter = season == 1
    revenue = cal_quarter_data(income.data, first_quarter, 0, 2, season)
    net_profit = cal_quarter_data(income.data, first_quarter, 1, 2, season)
    prior_season = income.season_data.copy() - 1
    prior_season = prior_season.replace(0, 4)
    prior_profit = cal_quarter_data(income.data, prior_season.values == 1, 3, 2, prior_season.values)
    yearly = _load_report(paths["report_income"], ["net_profit"], "1Y", trade_days, start, end, balance.codes)
    assets, current_assets, current_liabilities, liabilities = (balance.data[:, index, :] for index in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        change_income = (net_profit - prior_profit) / (np.abs(net_profit) + np.abs(prior_profit))
        two_losses = (yearly.data[:, 0, :] < 0) & (yearly.data[:, 1, :] < 0)
        values = (-1.32 - 0.407 * np.log(assets) + 6.03 * liabilities / assets
                  - 1.43 * (current_assets - current_liabilities) / assets
                  + 0.0757 * current_liabilities / current_assets - 1.72 * (liabilities > assets)
                  - 2.37 * net_profit / assets + 0.285 * two_losses - 0.521 * change_income
                  - 1.83 * revenue / liabilities)
    return pd.DataFrame(values, index=trade_days, columns=balance.codes).replace([np.inf, -np.inf], np.nan)


def refresh_qmj(paths: dict[str, Path], missing_dates: pd.DatetimeIndex) -> dict:
    if missing_dates.empty:
        return {"status": "up_to_date", "rows_added": 0}
    history_start = missing_dates.min().normalize() - pd.DateOffset(years=7)
    trade_days = _trade_days(paths["stock_bar"], history_start, missing_dates.max())
    if len(trade_days) < 1251:
        raise ValueError("QMJ 至少需要 1251 个历史交易日")
    start, end = trade_days.min(), trade_days.max()
    balance = _load_report(paths["report_balance"], [
        "total_assets", "total_liability", "total_owner_equities",
    ], "0Q", trade_days, start, end)
    codes = list(balance.codes)
    income = _load_report(paths["report_income"], [
        "operating_revenue", "operating_cost", "net_profit",
    ], "1Q", trade_days, start, end, codes)
    season = income.season_data.values == 1
    revenue = cal_quarter_data(income.data, season, 0, 3, income.season_data.values)
    cost = cal_quarter_data(income.data, season, 1, 3, income.season_data.values)
    net_profit = cal_quarter_data(income.data, season, 2, 3, income.season_data.values)
    cash = _load_report(paths["report_cashflow"], ["net_operate_cash_flow"], "1Q", trade_days, start, end, codes)
    cash_flow = cal_quarter_data(cash.data, cash.season_data.values == 1, 0, 1, cash.season_data.values)
    assets, liabilities, equity = (balance.data[:, index, :] for index in range(3))
    gross = revenue - cost
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics = {
            "gpoa": gross / assets,
            "roe": net_profit / equity,
            "roa": net_profit / assets,
            "cfoa": cash_flow / assets,
            "acc": (net_profit - cash_flow) / assets,
        }
    profitability, growth = [], []
    roe_vol = None
    for name, values in metrics.items():
        frame = pd.DataFrame(values, index=trade_days, columns=codes).replace([np.inf, -np.inf], np.nan)
        if name == "roe":
            roe_vol = frame.mask(frame == frame.shift()).rolling(1250, min_periods=4).std().ffill()
        profitability.append(_cross_section_zscore(frame))
        growth.append(_cross_section_zscore(frame.diff(1250).div(frame).replace([np.inf, -np.inf], np.nan)))
    profitability_df = pd.DataFrame(np.nanmean(np.stack(profitability), axis=0), index=trade_days, columns=codes)
    growth_df = pd.DataFrame(np.nanmean(np.stack(growth), axis=0), index=trade_days, columns=codes)
    bab = pd.read_parquet(paths["bab_factor"], columns=["datetime", "code", "bab"], filters=[("datetime", ">=", start), ("datetime", "<=", end)])
    bab = bab.drop_duplicates(["datetime", "code"], keep="last").pivot(index="datetime", columns="code", values="bab").reindex(index=trade_days, columns=codes)
    oscore = _calculate_oscore(paths, trade_days, start, end).reindex(columns=codes)
    zscore = _calculate_zscore(paths, trade_days, start, end).reindex(columns=codes)
    leverage = pd.DataFrame(liabilities / assets, index=trade_days, columns=codes).replace([np.inf, -np.inf], np.nan)
    safety_parts = []
    for frame in (roe_vol, bab, oscore, zscore, leverage):
        safety_parts.append(_cross_section_zscore(frame.replace([np.inf, -np.inf], np.nan)))
    safety_df = pd.DataFrame(np.nanmean(np.stack(safety_parts), axis=0), index=trade_days, columns=codes)
    qmj_parts = [_cross_section_zscore(frame) for frame in (growth_df, profitability_df, safety_df)]
    qmj_df = pd.DataFrame(np.nanmean(np.stack(qmj_parts), axis=0), index=trade_days, columns=codes)
    frames = {"growth": growth_df, "profitability": profitability_df, "safety": safety_df, "qmj": qmj_df}
    missing = pd.DatetimeIndex(missing_dates).intersection(trade_days)
    long_frames = []
    for name, frame in frames.items():
        item = frame.loc[missing].rename_axis("datetime").reset_index().melt(id_vars="datetime", var_name="code", value_name=name)
        long_frames.append(item.set_index(["datetime", "code"]))
    added = pd.concat(long_frames, axis=1).reset_index()
    output = paths["qmj_factor"]
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame(columns=added.columns)
    combined = pd.concat([existing, added], ignore_index=True).sort_values(["datetime", "code"]).drop_duplicates(["datetime", "code"], keep="last")
    _atomic_write(combined, output)
    return {"status": "updated", "dates_added": int(added["datetime"].nunique()), "rows_added": len(added), "output": str(output)}


def prepare(args: argparse.Namespace) -> dict:
    paths = _load_config(args.config)
    problems = _validate_sources(paths)
    if problems:
        result = {"status": "needs_paths", "problems": problems, "required_keys": sorted(DEFAULTS)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    source_end = {
        "stock_bar": _date_bounds(paths["stock_bar"])[1],
        "capital": _date_bounds(paths["capital"])[1],
        "market_code": _market_end(paths["market_bar"], args.market_code),
    }
    target = min(source_end.values())
    requested_target = _inclusive_end(args.target_end) if args.target_end else None
    if args.target_end:
        target = min(target, requested_target)
    limiting_sources = [key for key, value in source_end.items() if value == min(source_end.values())]
    expected = _trade_days(paths["stock_bar"], pd.Timestamp(args.factor_start), target)
    bab_missing = _missing_factor_dates(paths["bab_factor"], expected, "bab")
    qmj_missing = _missing_factor_dates(paths["qmj_factor"], expected, "qmj")
    plan = {
        "status": "source_limited" if requested_target is not None and requested_target > target else ("dry_run" if args.dry_run else "ok"),
        "target_end": str(target),
        "requested_target_end": args.target_end,
        "source_end": {key: str(value) for key, value in source_end.items()},
        "target_limited_by": limiting_sources if requested_target is not None and requested_target > target else [],
        "bab_missing_dates": len(bab_missing),
        "qmj_missing_dates": len(qmj_missing),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    plan["bab"] = refresh_bab(paths, bab_missing, args.market_code)
    qmj_missing = _missing_factor_dates(paths["qmj_factor"], expected, "qmj")
    plan["qmj"] = refresh_qmj(paths, qmj_missing)
    premium_args = Namespace(
        stock_bar=str(paths["stock_bar"]), capital=str(paths["capital"]), bab=str(paths["bab_factor"]),
        qmj=str(paths["qmj_factor"]), output=str(paths["premium"]), start=None, end=str(target),
        signal_lag=args.signal_lag, rebuild_range=False, dry_run=False,
    )
    plan["premium"] = factor_pipeline.update_premiums(premium_args)
    plan["status"] = "source_limited" if requested_target is not None and requested_target > target else "ok"
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON 数据路径映射；未提供时尝试项目默认路径")
    parser.add_argument("--target-end", help="通常传本次分析/回测截止日")
    parser.add_argument("--factor-start", default="2013-01-04")
    parser.add_argument("--market-code", default="000985.SH")
    parser.add_argument("--signal-lag", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
