"""Walk-forward portfolio optimization with user-selected rebalance frequency and curve plotting."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import factor_pipeline  # noqa: E402
import optimize_portfolio  # noqa: E402
import prepare_data  # noqa: E402
from runtime_paths import project_root  # noqa: E402


PROJECT_ROOT = project_root()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def _cross_section(data: pd.DataFrame, day: pd.Timestamp, date_column: str | None, model: str | None) -> pd.DataFrame:
    frame = data
    if model and "model" in frame:
        frame = frame[frame["model"] == model]
    if date_column:
        dates = pd.to_datetime(frame[date_column])
        available = dates[dates <= day]
        if available.empty:
            return frame.iloc[0:0].copy()
        frame = frame[dates == available.max()]
    return frame.copy()


def _select(frame: pd.DataFrame, factor: str | None, side: str, pct: float, code_column: str) -> pd.DataFrame:
    frame = frame.drop_duplicates(code_column, keep="last")
    if not factor:
        return frame
    if factor not in frame:
        raise ValueError(f"候选池缺少筛选因子列 {factor!r}")
    frame[factor] = pd.to_numeric(frame[factor], errors="coerce")
    frame = frame.dropna(subset=[factor])
    count = max(1, math.ceil(len(frame) * pct))
    return frame.sort_values([factor, code_column], ascending=[side == "bottom", True], kind="mergesort").iloc[:count]


def _performance(returns: pd.Series, risk_free_rate: float) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {}
    nav = (1.0 + returns).cumprod()
    years = len(returns) / 252.0
    annual_return = float(nav.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and nav.iloc[-1] > 0 else np.nan
    volatility = float(returns.std(ddof=1) * math.sqrt(252.0))
    drawdown = nav / nav.cummax() - 1.0
    return {
        "total_return": float(nav.iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": (annual_return - risk_free_rate) / volatility if volatility > 0 else None,
        "max_drawdown": float(drawdown.min()),
        "trading_days": len(returns),
    }


def _simulate_weight_schedule(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    code_column: str,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Apply the same return and turnover convention as the main backtest."""
    effective = weights.dropna(subset=["effective_date"]).copy()
    effective["effective_date"] = pd.to_datetime(effective["effective_date"])
    effective_groups = {day: group for day, group in effective.groupby("effective_date")}
    active = pd.Series(dtype=float)
    daily_rows: list[dict] = []
    for day in trading_days:
        turnover = 0.0
        if day in effective_groups:
            new = effective_groups[day].set_index(code_column)["weight"]
            union = active.index.union(new.index)
            old_aligned = active.reindex(union, fill_value=0.0)
            new_aligned = new.reindex(union, fill_value=0.0)
            turnover = (
                float(np.abs(new_aligned - old_aligned).sum())
                if active.empty
                else float(0.5 * np.abs(new_aligned - old_aligned).sum())
            )
            active = new
        if active.empty:
            continue
        day_returns = returns.loc[day].reindex(active.index)
        available = day_returns.notna()
        if not available.any():
            portfolio_return = 0.0
            invested_weight = 0.0
        else:
            usable_weights = active[available]
            invested_weight = float(usable_weights.sum())
            usable_weights = usable_weights / invested_weight
            portfolio_return = float(np.dot(usable_weights, day_returns[available]))
        net_return = portfolio_return - turnover * transaction_cost_bps / 10000.0
        daily_rows.append(
            {
                "datetime": day,
                "gross_return": portfolio_return,
                "net_return": net_return,
                "turnover": turnover,
                "invested_weight_before_rescale": invested_weight,
            }
        )
    if not daily_rows:
        raise ValueError("等权诊断基准没有形成可回测的持仓区间")
    frame = pd.DataFrame(daily_rows).set_index("datetime")
    frame["nav"] = (1.0 + frame["net_return"]).cumprod()
    return frame


def _run_prepare(args: argparse.Namespace) -> None:
    if args.skip_prepare:
        return
    prepare_args = Namespace(
        config=args.data_config,
        target_end=args.end,
        factor_start=args.factor_start,
        market_code=args.market_code,
        signal_lag=args.signal_lag,
        dry_run=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        prepare_data.prepare(prepare_args)


def run_backtest(args: argparse.Namespace) -> dict:
    _run_prepare(args)
    if args.data_config and Path(args.returns_path).resolve() == optimize_portfolio.DEFAULT_RETURNS.resolve():
        args.returns_path = str(prepare_data._load_config(args.data_config)["fund_bar"])
    universe = _read(_path(args.universe))
    if args.code_column not in universe:
        raise ValueError(f"候选池缺少代码列 {args.code_column!r}")
    universe[args.code_column] = universe[args.code_column].astype(str)
    date_column = args.date_column
    if not date_column:
        date_column = "as_of" if "as_of" in universe else ("datetime" if "datetime" in universe else None)
    if date_column:
        universe[date_column] = pd.to_datetime(universe[date_column])
    start = pd.Timestamp(args.start)
    end = factor_pipeline._inclusive_end(args.end)
    all_codes = sorted(universe[args.code_column].dropna().unique().tolist())
    returns = optimize_portfolio._load_returns(_path(args.returns_path), all_codes, start, end, args.return_column)
    trading_days = returns.index[(returns.index >= start) & (returns.index <= end)]
    rebalance_dates = factor_pipeline._rebalance_dates(trading_days, args.rebalance_frequency)
    if rebalance_dates.empty:
        raise ValueError("回测区间内没有调仓日")

    current_weights: dict[str, float] = {}
    weight_records: list[pd.DataFrame] = []
    comparison_weight_records: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="factor-backtest-") as temporary:
        temporary_path = Path(temporary)
        for index, day in enumerate(rebalance_dates):
            cross_section = _cross_section(universe, day, date_column, args.model)
            selected = _select(cross_section, args.selection_factor, args.selection_side, args.selection_pct, args.code_column)
            if selected.empty:
                raise ValueError(f"{day} 没有可优化的候选标的")
            current_column = None
            if args.max_turnover is not None:
                current_column = "__current_weight"
                selected[current_column] = selected[args.code_column].map(current_weights).fillna(0.0)
            input_path = temporary_path / f"universe_{index}.parquet"
            output_path = temporary_path / f"weights_{index}.parquet"
            selected.to_parquet(input_path, index=False)
            optimizer_args = Namespace(
                universe=str(input_path), as_of=str(day), objective=args.objective, output=str(output_path),
                code_column=args.code_column, returns_path=str(_path(args.returns_path)), return_column=args.return_column,
                years=args.lookback_years, min_observations=args.min_observations,
                expected_return_column=args.expected_return_column, score_column=args.score_column,
                score_target=args.score_target, risk_aversion=args.risk_aversion,
                risk_free_rate=args.risk_free_rate, min_weight=args.min_weight, max_weight=args.max_weight,
                factor_eq=args.factor_eq, factor_min=args.factor_min, factor_max=args.factor_max,
                current_weight_column=current_column, max_turnover=args.max_turnover,
                max_iterations=args.max_iterations, tolerance=args.tolerance,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                solved = optimize_portfolio.solve(optimizer_args)
            solved = solved[[args.code_column, "weight"]].copy()
            solved["rebalance_date"] = day
            later_days = trading_days[trading_days > day]
            effective_date = later_days[0] if len(later_days) else pd.NaT
            solved["effective_date"] = effective_date
            weight_records.append(solved)
            current_weights = dict(zip(solved[args.code_column], solved["weight"]))
            if args.selection_weighting_comparison and pd.notna(effective_date):
                positive = solved.loc[solved["weight"] > args.active_weight_tolerance]
                holding_count = len(positive)
                if holding_count == 0:
                    raise ValueError(f"{day} 优化结果没有正权重持仓，无法生成等权诊断")
                same_holdings = positive[[args.code_column]].copy()
                same_holdings["weight"] = 1.0 / holding_count
                same_holdings["strategy"] = "optimized_holdings_equal_weight"
                ranked = selected.sort_values(
                    [args.selection_factor, args.code_column],
                    ascending=[args.selection_side == "bottom", True],
                    kind="mergesort",
                ).iloc[:holding_count]
                if len(ranked) != holding_count:
                    raise ValueError(f"{day} 因子排序标的不足，无法匹配 {holding_count} 个持仓")
                factor_top = ranked[[args.code_column]].copy()
                factor_top["weight"] = 1.0 / holding_count
                factor_top["strategy"] = "factor_top_n_equal_weight"
                for benchmark_weights in (same_holdings, factor_top):
                    benchmark_weights["rebalance_date"] = day
                    benchmark_weights["effective_date"] = effective_date
                    benchmark_weights["holding_count"] = holding_count
                    comparison_weight_records.append(benchmark_weights)

    weights = pd.concat(weight_records, ignore_index=True)
    effective = weights.dropna(subset=["effective_date"])
    effective_groups = {day: group for day, group in effective.groupby("effective_date")}
    active = pd.Series(dtype=float)
    daily_rows = []
    for day in trading_days:
        turnover = 0.0
        if day in effective_groups:
            new = effective_groups[day].set_index(args.code_column)["weight"]
            union = active.index.union(new.index)
            old_aligned = active.reindex(union, fill_value=0.0)
            new_aligned = new.reindex(union, fill_value=0.0)
            turnover = float(np.abs(new_aligned - old_aligned).sum()) if active.empty else float(0.5 * np.abs(new_aligned - old_aligned).sum())
            active = new
        if active.empty:
            continue
        day_returns = returns.loc[day].reindex(active.index)
        available = day_returns.notna()
        if not available.any():
            portfolio_return = 0.0
            invested_weight = 0.0
        else:
            usable_weights = active[available]
            invested_weight = float(usable_weights.sum())
            usable_weights = usable_weights / invested_weight
            portfolio_return = float(np.dot(usable_weights, day_returns[available]))
        net_return = portfolio_return - turnover * args.transaction_cost_bps / 10000.0
        daily_rows.append({
            "datetime": day,
            "gross_return": portfolio_return,
            "net_return": net_return,
            "turnover": turnover,
            "invested_weight_before_rescale": invested_weight,
        })
    curve = pd.DataFrame(daily_rows).set_index("datetime")
    if curve.empty:
        raise ValueError("没有形成可回测的持仓区间")
    curve["portfolio_nav"] = (1.0 + curve["net_return"]).cumprod()

    if args.benchmark_code:
        benchmark = optimize_portfolio._load_returns(
            _path(args.returns_path), [args.benchmark_code], start, end, args.return_column
        )[args.benchmark_code].reindex(curve.index).fillna(0.0)
        benchmark_name = args.benchmark_code
    else:
        benchmark = returns.reindex(curve.index).mean(axis=1, skipna=True).fillna(0.0)
        benchmark_name = "universe_equal_weight"
    curve["benchmark_return"] = benchmark
    curve["benchmark_nav"] = (1.0 + benchmark).cumprod()
    curve = curve.reset_index()

    output_dir = _path(args.output_dir) if args.output_dir else PROJECT_ROOT / "fund_return_analysis" / "outputs" / pd.Timestamp(args.end).strftime("%Y%m%d") / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(output_dir / "backtest_curve.parquet", index=False)
    weights.to_parquet(output_dir / "rebalance_weights.parquet", index=False)
    comparison_summary = None
    comparison_figure_path = None
    if args.selection_weighting_comparison:
        if not comparison_weight_records:
            raise ValueError("未生成选择与权重诊断基准")
        comparison_weights = pd.concat(comparison_weight_records, ignore_index=True)
        comparison_weights.to_parquet(
            output_dir / "selection_weighting_comparison_weights.parquet", index=False
        )
        curve_indexed = curve.set_index("datetime")
        comparison_curve = pd.DataFrame(
            {"optimized_portfolio": curve_indexed["portfolio_nav"]}, index=curve_indexed.index
        )
        comparison_metrics = {
            "optimized_portfolio": _performance(
                curve_indexed["net_return"], args.risk_free_rate
            )
        }
        comparison_turnover = {
            "optimized_portfolio": float(curve_indexed["turnover"].sum())
        }
        for strategy, strategy_weights in comparison_weights.groupby("strategy", sort=False):
            simulated = _simulate_weight_schedule(
                strategy_weights,
                returns,
                trading_days,
                args.code_column,
                args.transaction_cost_bps,
            ).reindex(curve_indexed.index)
            comparison_curve[strategy] = simulated["nav"]
            comparison_metrics[strategy] = _performance(
                simulated["net_return"], args.risk_free_rate
            )
            comparison_turnover[strategy] = float(simulated["turnover"].sum())
        comparison_curve.index.name = "datetime"
        comparison_curve.reset_index().to_parquet(
            output_dir / "selection_weighting_comparison.parquet", index=False
        )
        comparison_figure_path = output_dir / "selection_weighting_comparison.png"
        comparison_fig, comparison_axis = plt.subplots(figsize=(12, 6.75))
        labels = {
            "optimized_portfolio": "Optimized portfolio",
            "optimized_holdings_equal_weight": "Same holdings, equal weight",
            "factor_top_n_equal_weight": "Factor top-N, equal weight",
        }
        styles = {
            "optimized_portfolio": {"color": "#B79250", "linestyle": "-", "linewidth": 2.4},
            "optimized_holdings_equal_weight": {"color": "#4F7C90", "linestyle": "--", "linewidth": 2.0},
            "factor_top_n_equal_weight": {"color": "#C7654C", "linestyle": ":", "linewidth": 2.0},
        }
        for column in comparison_curve:
            comparison_axis.plot(
                comparison_curve.index,
                comparison_curve[column],
                label=labels[column],
                **styles[column],
            )
        comparison_axis.set_title(
            f"Selection and weighting comparison ({args.rebalance_frequency} rebalance)"
        )
        comparison_axis.set_xlabel("Date")
        comparison_axis.set_ylabel("Net asset value")
        comparison_axis.grid(alpha=0.25)
        comparison_axis.legend()
        comparison_fig.tight_layout()
        comparison_fig.savefig(comparison_figure_path, dpi=180)
        plt.close(comparison_fig)
        comparison_summary = {
            "definition": {
                "optimized_portfolio": "original optimized portfolio",
                "optimized_holdings_equal_weight": "same positive-weight holdings as the optimized portfolio, equal target weights",
                "factor_top_n_equal_weight": "highest or lowest selection-factor N instruments according to selection-side, equal target weights; N equals the optimized active holding count",
            },
            "active_weight_tolerance": args.active_weight_tolerance,
            "performance": comparison_metrics,
            "total_one_way_turnover": comparison_turnover,
            "outputs": {
                "curve": str(output_dir / "selection_weighting_comparison.parquet"),
                "weights": str(output_dir / "selection_weighting_comparison_weights.parquet"),
                "figure": str(comparison_figure_path),
            },
        }
    fig, axis = plt.subplots(figsize=(12, 6.75))
    axis.plot(curve["datetime"], curve["portfolio_nav"], label="Optimized portfolio", linewidth=2)
    axis.plot(curve["datetime"], curve["benchmark_nav"], label=benchmark_name, linewidth=1.5, alpha=0.8)
    axis.set_title(f"Walk-forward backtest ({args.rebalance_frequency} rebalance)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Net asset value")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    figure_path = output_dir / "backtest_curve.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": "ok",
        "period": [str(curve["datetime"].min()), str(curve["datetime"].max())],
        "rebalance_frequency": args.rebalance_frequency,
        "rebalance_count": int(weights["rebalance_date"].nunique()),
        "selection": {"factor": args.selection_factor, "side": args.selection_side, "pct": args.selection_pct},
        "objective": args.objective,
        "transaction_cost_bps": args.transaction_cost_bps,
        "total_one_way_turnover": float(curve["turnover"].sum()),
        "portfolio": _performance(curve.set_index("datetime")["net_return"], args.risk_free_rate),
        "benchmark": _performance(curve.set_index("datetime")["benchmark_return"], args.risk_free_rate),
        "outputs": {
            "curve": str(output_dir / "backtest_curve.parquet"),
            "weights": str(output_dir / "rebalance_weights.parquet"),
            "figure": str(figure_path),
        },
    }
    if comparison_summary is not None:
        summary["selection_weighting_comparison"] = comparison_summary
    (output_dir / "backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--rebalance-frequency", default="monthly", choices=("daily", "weekly", "monthly", "quarterly", "yearly"))
    parser.add_argument("--objective", required=True, choices=(
        "min_variance", "max_return", "max_sharpe", "mean_variance", "max_score", "min_score", "target_score"
    ))
    parser.add_argument("--output-dir")
    parser.add_argument("--code-column", default="code")
    parser.add_argument("--date-column")
    parser.add_argument("--model", choices=("capm", "four_factor", "six_factor", "custom"))
    parser.add_argument("--selection-factor")
    parser.add_argument("--selection-side", choices=("top", "bottom"), default="top")
    parser.add_argument("--selection-pct", type=float, default=0.10)
    parser.add_argument("--returns-path", default=str(optimize_portfolio.DEFAULT_RETURNS))
    parser.add_argument("--return-column")
    parser.add_argument("--benchmark-code")
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--min-observations", type=int, default=252)
    parser.add_argument("--expected-return-column")
    parser.add_argument("--score-column")
    parser.add_argument("--score-target", type=float)
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float)
    parser.add_argument("--factor-eq", action="append")
    parser.add_argument("--factor-min", action="append")
    parser.add_argument("--factor-max", action="append")
    parser.add_argument("--max-turnover", type=float)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument(
        "--selection-weighting-comparison",
        action="store_true",
        help="额外输出优化组合、同持仓等权、因子Top-N等权三线诊断图",
    )
    parser.add_argument(
        "--active-weight-tolerance",
        type=float,
        default=1e-10,
        help="判定优化组合实际持仓的正权重阈值",
    )
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--data-config")
    parser.add_argument("--factor-start", default="2013-01-04")
    parser.add_argument("--market-code", default="000985.SH")
    parser.add_argument("--signal-lag", type=int, choices=(0, 1), default=0)
    parser.add_argument("--skip-prepare", action="store_true", help="仅供已在同一任务中成功执行 prepare 时使用")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0 < args.selection_pct <= 1:
        parser.error("--selection-pct 必须在 (0, 1] 内")
    if args.transaction_cost_bps < 0:
        parser.error("--transaction-cost-bps 必须非负")
    if args.selection_weighting_comparison and not args.selection_factor:
        parser.error("--selection-weighting-comparison 需要 --selection-factor")
    if args.active_weight_tolerance < 0:
        parser.error("--active-weight-tolerance 必须非负")
    run_backtest(args)


if __name__ == "__main__":
    main()
