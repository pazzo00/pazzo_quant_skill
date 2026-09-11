"""Update local Fama-French-style, UMD, BAB, and QMJ premia; estimate exposures; select quantiles."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from runtime_paths import project_root


PROJECT_ROOT = project_root()
DEFAULT_PREMIUM = PROJECT_ROOT / "fund_return_analysis" / "premium.parquet"
DEFAULT_STOCK_BAR = PROJECT_ROOT / "data" / "stock" / "stock_bar_1day.parquet"
DEFAULT_CAPITAL = PROJECT_ROOT / "data" / "stock" / "capital.parquet"
DEFAULT_FUND_BAR = PROJECT_ROOT / "data" / "index" / "index_bar_1day.parquet"
DEFAULT_BAB = PROJECT_ROOT / "daily_factors" / "BAB" / "bab.parquet"
DEFAULT_QMJ = PROJECT_ROOT / "daily_factors" / "QMJ" / "qmj.parquet"
STANDARD_FACTORS = ["mkt_rtn", "smb", "hml", "umd", "bab", "qmj"]
MODEL_FACTORS = {
    "capm": ["mkt_rtn"],
    "four_factor": ["mkt_rtn", "smb", "hml", "umd"],
    "six_factor": STANDARD_FACTORS,
}


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _inclusive_end(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp == timestamp.normalize():
        return timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def _read_table(path: Path, columns: list[str] | None = None, filters=None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, usecols=columns)
        if "datetime" in frame:
            frame["datetime"] = pd.to_datetime(frame["datetime"])
        if filters:
            for col, op, value in filters:
                if op == ">=":
                    frame = frame[frame[col] >= value]
                elif op == "<=":
                    frame = frame[frame[col] <= value]
                elif op == "in":
                    frame = frame[frame[col].isin(value)]
                else:
                    raise ValueError(f"CSV 暂不支持过滤操作: {op}")
        return frame
    return pd.read_parquet(path, columns=columns, filters=filters)


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if path.suffix.lower() == ".csv":
        frame.to_csv(tmp, index=False)
    else:
        frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _date_bounds(path: Path, column: str = "datetime") -> tuple[pd.Timestamp, pd.Timestamp]:
    parquet = pq.ParquetFile(path)
    index = parquet.schema_arrow.names.index(column)
    minima: list[pd.Timestamp] = []
    maxima: list[pd.Timestamp] = []
    for row_group in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row_group).column(index).statistics
        if stats is not None and stats.has_min_max:
            minima.append(pd.Timestamp(stats.min))
            maxima.append(pd.Timestamp(stats.max))
    if minima:
        return min(minima), max(maxima)
    dates = pd.read_parquet(path, columns=[column])[column]
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def _vw_return(frame: pd.DataFrame) -> float:
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["rtn", "market_cap"])
    frame = frame[frame["market_cap"] > 0]
    if frame.empty or not np.isfinite(frame["market_cap"].sum()):
        return np.nan
    weights = frame["market_cap"] / frame["market_cap"].sum()
    return float(np.dot(frame["rtn"], weights))


def _top_bottom(signal: pd.Series, returns: pd.Series, cap: pd.Series, pct: float = 0.30) -> float:
    frame = pd.concat([signal.rename("signal"), returns.rename("rtn"), cap.rename("market_cap")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().sort_values(["signal"], kind="mergesort")
    count = int(len(frame) * pct)
    if count < 1:
        return np.nan
    return _vw_return(frame.iloc[-count:]) - _vw_return(frame.iloc[:count])


def _two_by_three(
    signal: pd.Series, returns: pd.Series, cap: pd.Series, *, exact_legacy_split: bool = True
) -> tuple[float, float]:
    frame = pd.concat([signal.rename("signal"), cap.rename("market_cap"), returns.rename("rtn")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame["market_cap"] > 0].sort_values("market_cap", kind="mergesort")
    half = len(frame) // 2
    if half < 2:
        return np.nan, np.nan
    small = frame.iloc[:half].sort_values("signal", kind="mergesort")
    big = frame.iloc[half:].sort_values("signal", kind="mergesort")
    buckets: list[list[float]] = [[], []]
    start = 0
    for fraction in (0.30, 0.70, 1.00):
        end = int(half * fraction)
        buckets[0].append(_vw_return(small.iloc[start:end]))
        big_end = end if exact_legacy_split else int(len(big) * fraction)
        big_start = start if exact_legacy_split else int(len(big) * (0 if fraction == 0.30 else (0.30 if fraction == 0.70 else 0.70)))
        buckets[1].append(_vw_return(big.iloc[big_start:big_end]))
        start = end
    small_returns, big_returns = buckets
    smb = float(np.nanmean(small_returns) - np.nanmean(big_returns))
    high_minus_low = float(0.5 * (small_returns[-1] + big_returns[-1]) - 0.5 * (small_returns[0] + big_returns[0]))
    return smb, high_minus_low


def _bab_return(signal: pd.Series, returns: pd.Series) -> float:
    signal = signal.replace([np.inf, -np.inf], np.nan).dropna()
    if len(signal) < 4:
        return np.nan
    weights = signal.rank(pct=True, ascending=False, method="average")
    weights = weights - weights.mean()
    gross = weights.abs().sum()
    if not np.isfinite(gross) or gross == 0:
        return np.nan
    weights = weights / gross
    positive_beta = float((signal * weights.where(weights > 0)).sum())
    negative_beta = float((signal * weights.where(weights < 0)).abs().sum())
    if positive_beta <= 0 or negative_beta <= 0:
        return np.nan
    weights.loc[weights > 0] /= positive_beta
    weights.loc[weights < 0] /= negative_beta
    weighted_returns = weights * returns.reindex(weights.index)
    return float(weighted_returns.sum(min_count=1)) if weighted_returns.notna().any() else np.nan


def _wide(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    frame = frame.drop_duplicates(["datetime", "code"], keep="last")
    return frame.pivot(index="datetime", columns="code", values=value).sort_index()


def update_premiums(args: argparse.Namespace) -> dict:
    stock_path, capital_path = _path(args.stock_bar), _path(args.capital)
    bab_path, qmj_path = _path(args.bab), _path(args.qmj)
    output = _path(args.output)
    bounds = {name: _date_bounds(path) for name, path in {
        "stock_bar": stock_path, "capital": capital_path, "bab": bab_path, "qmj": qmj_path
    }.items()}
    common_start = max(value[0] for value in bounds.values())
    common_end = min(value[1] for value in bounds.values())
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame(columns=["datetime"] + STANDARD_FACTORS)
    if not existing.empty:
        existing["datetime"] = pd.to_datetime(existing["datetime"])
    if args.start:
        requested_start = pd.Timestamp(args.start)
    elif not existing.empty:
        requested_start = pd.Timestamp(existing["datetime"].max()) + pd.Timedelta(days=1)
    else:
        requested_start = common_start
    start = max(requested_start, common_start)
    end = min(_inclusive_end(args.end) if args.end else common_end, common_end)
    summary = {
        "status": "dry_run" if args.dry_run else "ok",
        "requested_start": str(requested_start),
        "effective_start": str(start),
        "effective_end": str(end),
        "common_available_end": str(common_end),
        "source_bounds": {key: [str(value[0]), str(value[1])] for key, value in bounds.items()},
        "output": str(output),
        "signal_lag": args.signal_lag,
    }
    if start > end:
        summary.update(status="up_to_date", rows_computed=0)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    lookback_start = start - pd.Timedelta(days=500 + args.signal_lag * 3)
    filters = [("datetime", ">=", lookback_start), ("datetime", "<=", end)]
    bars = _read_table(stock_path, ["datetime", "code", "close", "factor"], filters)
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars["adjusted_price"] = bars["close"] * bars["factor"]
    prices = _wide(bars[["datetime", "code", "adjusted_price"]], "adjusted_price")
    returns = prices.pct_change(fill_method=None)
    momentum = prices.shift(252) / prices.shift(21) - 1.0

    range_filters = [("datetime", ">=", start - pd.Timedelta(days=args.signal_lag * 3 + 7)), ("datetime", "<=", end)]
    capital = _read_table(capital_path, ["datetime", "code", "market_cap", "pb_ratio"], range_filters)
    capital["datetime"] = pd.to_datetime(capital["datetime"])
    caps = _wide(capital[["datetime", "code", "market_cap"]], "market_cap")
    bp_data = capital.assign(bp=1.0 / capital["pb_ratio"].replace(0, np.nan))
    bp = _wide(bp_data[["datetime", "code", "bp"]], "bp")
    bab_long = _read_table(bab_path, ["datetime", "code", "bab"], range_filters)
    qmj_long = _read_table(qmj_path, ["datetime", "code", "qmj"], range_filters)
    bab_long["datetime"] = pd.to_datetime(bab_long["datetime"])
    qmj_long["datetime"] = pd.to_datetime(qmj_long["datetime"])
    bab = _wide(bab_long, "bab")
    qmj = _wide(qmj_long, "qmj")
    if args.signal_lag:
        caps, bp, bab, qmj = (item.shift(args.signal_lag) for item in (caps, bp, bab, qmj))
        momentum = momentum.shift(args.signal_lag)

    candidate_days = returns.index[(returns.index >= start) & (returns.index <= end)]
    common_days = candidate_days.intersection(caps.index).intersection(bp.index).intersection(bab.index).intersection(qmj.index)
    rows: list[dict] = []
    for day in common_days:
        day_returns = returns.loc[day]
        day_cap = caps.loc[day]
        market_frame = pd.concat([
            momentum.loc[day].rename("signal"), day_returns.rename("rtn"), day_cap.rename("market_cap")
        ], axis=1).dropna()
        market = _vw_return(market_frame)
        smb, hml = _two_by_three(bp.loc[day], day_returns, day_cap)
        _, qmj_premium = _two_by_three(qmj.loc[day], day_returns, day_cap)
        rows.append({
            "datetime": day,
            "mkt_rtn": market,
            "smb": smb,
            "hml": hml,
            "umd": _top_bottom(momentum.loc[day], day_returns, day_cap),
            "bab": _bab_return(bab.loc[day], day_returns),
            "qmj": qmj_premium,
        })
    computed = pd.DataFrame(rows)
    if computed.empty:
        summary.update(status="no_common_dates", rows_computed=0)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    invalid = computed[STANDARD_FACTORS].isna().any(axis=1)
    if invalid.any():
        bad_dates = computed.loc[invalid, "datetime"].astype(str).tolist()[:10]
        raise ValueError(f"以下日期存在无法计算的因子溢价（最多显示 10 个）: {bad_dates}")

    if existing.empty:
        merged = computed
    else:
        if args.rebuild_range or args.start:
            existing = existing[~existing["datetime"].isin(computed["datetime"])]
        merged = pd.concat([existing, computed], ignore_index=True, sort=False)
        merged = merged.sort_values("datetime").drop_duplicates("datetime", keep="last")
    _atomic_write(merged, output)
    summary.update(rows_computed=len(computed), output_rows=len(merged), first_computed=str(computed["datetime"].min()), last_computed=str(computed["datetime"].max()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _load_returns(path: Path, codes: Iterable[str], start: pd.Timestamp, end: pd.Timestamp, return_column: str | None) -> pd.DataFrame:
    codes = list(dict.fromkeys(str(code) for code in codes))
    columns = ["datetime", "code", return_column] if return_column else ["datetime", "code", "close", "factor"]
    frame = _read_table(path, columns, [("datetime", ">=", start - pd.Timedelta(days=7)), ("datetime", "<=", end), ("code", "in", codes)])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    if return_column:
        result = _wide(frame, return_column).reindex(columns=codes)
        return result[(result.index >= start) & (result.index <= end)]
    frame["adjusted_price"] = frame["close"] * frame["factor"]
    result = _wide(frame[["datetime", "code", "adjusted_price"]], "adjusted_price").pct_change(fill_method=None).reindex(columns=codes)
    return result[(result.index >= start) & (result.index <= end)]


def estimate_exposures(args: argparse.Namespace) -> pd.DataFrame:
    requested_end = pd.Timestamp(args.as_of)
    end = _inclusive_end(requested_end)
    models = {"custom": list(dict.fromkeys(args.factors))} if args.factors else {
        name: MODEL_FACTORS[name] for name in args.models
    }
    all_factors = list(dict.fromkeys(factor for factors in models.values() for factor in factors))
    premium = _read_table(_path(args.premium), ["datetime"] + all_factors)
    premium["datetime"] = pd.to_datetime(premium["datetime"])
    premium = premium[premium["datetime"] <= end].set_index("datetime").sort_index()
    premium = premium.replace([np.inf, -np.inf], np.nan)
    if args.start_as_of:
        requested_start = pd.Timestamp(args.start_as_of)
        eligible_days = premium.index[(premium.index >= requested_start) & (premium.index <= end)].unique().sort_values()
        as_of_dates = _rebalance_dates(eligible_days, args.frequency)
    else:
        available = premium.index[premium.index <= end]
        as_of_dates = pd.DatetimeIndex([available.max()]) if len(available) else pd.DatetimeIndex([])
    if as_of_dates.empty:
        raise ValueError("指定范围内没有可用的因子交易日")
    history_start = as_of_dates.min().normalize() - pd.DateOffset(years=args.years)
    returns = _load_returns(_path(args.returns_path), args.codes, history_start, end, args.return_column)
    results: list[dict] = []
    for as_of in as_of_dates:
        window_start = as_of.normalize() - pd.DateOffset(years=args.years)
        premium_window = premium[(premium.index >= window_start) & (premium.index <= as_of)]
        returns_window = returns[(returns.index >= window_start) & (returns.index <= as_of)]
        for model, factors in models.items():
            for code in args.codes:
                joined = pd.concat([returns_window[code].rename("fund_rtn"), premium_window[factors]], axis=1).dropna()
                row: dict = {
                    "as_of": as_of,
                    "code": code,
                    "model": model,
                    "status": "ok",
                    "n_obs": len(joined),
                    "alpha_daily": np.nan,
                    "r2": np.nan,
                    "window_start": joined.index.min() if not joined.empty else pd.NaT,
                    "window_end": joined.index.max() if not joined.empty else pd.NaT,
                    **{factor: np.nan for factor in STANDARD_FACTORS},
                }
                if len(joined) < args.min_observations:
                    row.update(status=f"insufficient_observations:{len(joined)}<{args.min_observations}")
                    results.append(row)
                    continue
                x = joined[factors].to_numpy(dtype=float)
                y = joined["fund_rtn"].to_numpy(dtype=float)
                design = np.column_stack([np.ones(len(x)), x])
                coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
                if rank < design.shape[1]:
                    row.update(status=f"rank_deficient:{rank}<{design.shape[1]}")
                    results.append(row)
                    continue
                prediction = design @ coefficients
                residual = y - prediction
                total = float(np.dot(y - y.mean(), y - y.mean()))
                row["r2"] = np.nan if total == 0 else 1.0 - float(np.dot(residual, residual)) / total
                row["alpha_daily"] = float(coefficients[0])
                row.update({factor: float(value) for factor, value in zip(factors, coefficients[1:])})
                results.append(row)
    result = pd.DataFrame(results)
    output = _path(args.output) if args.output else PROJECT_ROOT / "fund_return_analysis" / "outputs" / requested_end.strftime("%Y%m%d") / "exposures.parquet"
    _atomic_write(result, output)
    print(json.dumps({
        "status": "ok",
        "output": str(output),
        "rows": len(result),
        "successful": int(result["status"].eq("ok").sum()),
        "models": list(models),
        "as_of_count": len(as_of_dates),
    }, ensure_ascii=False, indent=2))
    return result


def _rebalance_dates(trading_days: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(trading_days).sort_values().unique()
    normalized = frequency.lower().replace("-", "_")
    if normalized in {"d", "daily", "day"}:
        return days
    periods = {
        "w": "W-FRI", "weekly": "W-FRI", "week": "W-FRI",
        "m": "M", "monthly": "M", "month": "M",
        "q": "Q", "quarterly": "Q", "quarter": "Q",
        "y": "Y", "yearly": "Y", "annual": "Y", "year": "Y",
    }
    if normalized not in periods:
        raise ValueError("frequency 仅支持 daily/weekly/monthly/quarterly/yearly")
    grouped = pd.Series(days, index=days).groupby(days.to_period(periods[normalized])).max()
    return pd.DatetimeIndex(grouped.to_numpy()).sort_values()


def select_quantile(args: argparse.Namespace) -> pd.DataFrame:
    frame = _read_table(_path(args.input))
    if args.code_column not in frame or args.factor not in frame:
        raise ValueError(f"输入必须包含 {args.code_column!r} 和 {args.factor!r}")
    frame[args.factor] = pd.to_numeric(frame[args.factor], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[args.code_column, args.factor]).copy()
    frame[args.code_column] = frame[args.code_column].astype(str)
    frame = frame.drop_duplicates(args.code_column, keep="last")
    ascending = args.side == "bottom"
    frame = frame.sort_values([args.factor, args.code_column], ascending=[ascending, True], kind="mergesort")
    count = max(1, math.ceil(len(frame) * args.pct))
    selected = frame.iloc[:count].copy()
    selected.insert(1, "selection_rank", np.arange(1, len(selected) + 1))
    selected.insert(2, "selection_side", args.side)
    selected.insert(3, "selection_pct", args.pct)
    output = _path(args.output)
    _atomic_write(selected, output)
    print(json.dumps({"status": "ok", "input_rows": len(frame), "selected_rows": len(selected), "output": str(output)}, ensure_ascii=False, indent=2))
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    update = sub.add_parser("update-premiums", help="增量更新本地 MKT/SMB/HML+UMD+BAB+QMJ 溢价")
    update.add_argument("--stock-bar", default=str(DEFAULT_STOCK_BAR))
    update.add_argument("--capital", default=str(DEFAULT_CAPITAL))
    update.add_argument("--bab", default=str(DEFAULT_BAB))
    update.add_argument("--qmj", default=str(DEFAULT_QMJ))
    update.add_argument("--output", default=str(DEFAULT_PREMIUM))
    update.add_argument("--start")
    update.add_argument("--end")
    update.add_argument("--signal-lag", type=int, choices=(0, 1), default=0)
    update.add_argument("--rebuild-range", action="store_true")
    update.add_argument("--dry-run", action="store_true")
    update.set_defaults(func=update_premiums)

    exposure = sub.add_parser("exposures", help="维护 CAPM、四因子和六因子滚动暴露")
    exposure.add_argument("--codes", nargs="+", required=True)
    exposure.add_argument("--as-of", required=True)
    exposure.add_argument("--start-as-of", help="提供后按 frequency 输出一系列滚动截面")
    exposure.add_argument("--frequency", default="monthly", choices=("daily", "weekly", "monthly", "quarterly", "yearly"))
    exposure.add_argument("--years", type=int, default=5)
    exposure.add_argument("--models", nargs="+", choices=tuple(MODEL_FACTORS), default=list(MODEL_FACTORS))
    exposure.add_argument("--factors", nargs="+", help="高级用法：覆盖三模型并计算一套 custom 暴露")
    exposure.add_argument("--premium", default=str(DEFAULT_PREMIUM))
    exposure.add_argument("--returns-path", default=str(DEFAULT_FUND_BAR))
    exposure.add_argument("--return-column")
    exposure.add_argument("--min-observations", type=int, default=252)
    exposure.add_argument("--output")
    exposure.set_defaults(func=estimate_exposures)

    select = sub.add_parser("select", help="按给定因子值选前/后分位")
    select.add_argument("--input", required=True)
    select.add_argument("--factor", required=True)
    select.add_argument("--side", choices=("top", "bottom"), required=True)
    select.add_argument("--pct", type=float, default=0.10)
    select.add_argument("--code-column", default="code")
    select.add_argument("--output", required=True)
    select.set_defaults(func=select_quantile)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "pct") and not 0 < args.pct <= 1:
        parser.error("--pct 必须在 (0, 1] 内")
    if hasattr(args, "years") and args.years <= 0:
        parser.error("--years 必须大于 0")
    args.func(args)


if __name__ == "__main__":
    main()
