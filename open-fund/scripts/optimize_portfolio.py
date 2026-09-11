"""Solve long-only portfolio weights for a selected universe."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from runtime_paths import project_root


PROJECT_ROOT = project_root()
DEFAULT_RETURNS = PROJECT_ROOT / "data" / "index" / "index_bar_1day.parquet"


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _inclusive_end(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp == timestamp.normalize():
        return timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def _read(path: Path, columns: list[str] | None = None, filters=None) -> pd.DataFrame:
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


def _parse_pairs(values: list[str] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"约束必须写成 factor=value，收到: {item}")
        key, raw = item.split("=", 1)
        result[key.strip()] = float(raw)
    return result


def _load_returns(
    path: Path, codes: list[str], start: pd.Timestamp, end: pd.Timestamp, return_column: str | None
) -> pd.DataFrame:
    columns = ["datetime", "code", return_column] if return_column else ["datetime", "code", "close", "factor"]
    frame = _read(path, columns, [("datetime", ">=", start - pd.Timedelta(days=7)), ("datetime", "<=", end), ("code", "in", codes)])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.drop_duplicates(["datetime", "code"], keep="last")
    if return_column:
        result = frame.pivot(index="datetime", columns="code", values=return_column).reindex(columns=codes)
        return result[(result.index >= start) & (result.index <= end)]
    frame["adjusted_price"] = frame["close"] * frame["factor"]
    prices = frame.pivot(index="datetime", columns="code", values="adjusted_price").sort_index()
    result = prices.pct_change(fill_method=None).reindex(columns=codes)
    return result[(result.index >= start) & (result.index <= end)]


def _nearest_psd(covariance: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    covariance = (covariance + covariance.T) / 2.0
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, floor)
    return (vectors * values) @ vectors.T


def _risk_model(returns: pd.DataFrame, min_observations: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    counts = returns.notna().sum().astype(int).to_dict()
    missing = {code: count for code, count in counts.items() if count < min_observations}
    if missing:
        raise ValueError(f"以下标的有效收益观测不足 {min_observations}: {missing}")
    means = returns.mean(skipna=True).to_numpy(dtype=float) * 252.0
    covariance_df = returns.cov(min_periods=min_observations) * 252.0
    variances = returns.var(skipna=True).to_numpy(dtype=float) * 252.0
    covariance = covariance_df.to_numpy(dtype=float).copy()
    for i in range(len(covariance)):
        covariance[i, i] = variances[i]
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = _nearest_psd(covariance)
    return means, covariance, counts


def _constraint_report(
    weights: np.ndarray,
    factors: pd.DataFrame,
    eq: dict[str, float],
    minimum: dict[str, float],
    maximum: dict[str, float],
    current: np.ndarray | None,
    max_turnover: float | None,
) -> dict:
    exposures = {column: float(np.dot(weights, factors[column])) for column in sorted(set(eq) | set(minimum) | set(maximum))}
    report = {
        "weight_sum": float(weights.sum()),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "factor_exposures": exposures,
        "factor_eq_residuals": {key: exposures[key] - value for key, value in eq.items()},
        "factor_min_slacks": {key: exposures[key] - value for key, value in minimum.items()},
        "factor_max_slacks": {key: value - exposures[key] for key, value in maximum.items()},
    }
    if current is not None:
        report["turnover_one_way"] = float(0.5 * np.abs(weights - current).sum())
        if max_turnover is not None:
            report["turnover_slack"] = max_turnover - report["turnover_one_way"]
    return report


def solve(args: argparse.Namespace) -> pd.DataFrame:
    universe = _read(_path(args.universe))
    if args.code_column not in universe:
        raise ValueError(f"候选池缺少代码列 {args.code_column!r}")
    universe = universe.dropna(subset=[args.code_column]).copy()
    universe[args.code_column] = universe[args.code_column].astype(str)
    universe = universe.drop_duplicates(args.code_column, keep="last").sort_values(args.code_column).reset_index(drop=True)
    codes = universe[args.code_column].tolist()
    count = len(codes)
    if count == 0:
        raise ValueError("候选池为空")

    requested_as_of = pd.Timestamp(args.as_of)
    as_of = _inclusive_end(requested_as_of)
    start = requested_as_of.normalize() - pd.DateOffset(years=args.years)
    returns = _load_returns(_path(args.returns_path), codes, start, as_of, args.return_column)
    historical_mean, covariance, observation_counts = _risk_model(returns, args.min_observations)
    expected = (
        universe[args.expected_return_column].to_numpy(dtype=float)
        if args.expected_return_column
        else historical_mean
    )
    if not np.isfinite(expected).all():
        raise ValueError("预期收益包含空值或无穷值")

    score = None
    if args.objective in {"max_score", "min_score", "target_score"}:
        if not args.score_column or args.score_column not in universe:
            raise ValueError(f"目标 {args.objective} 需要候选池中的 --score-column")
        score = universe[args.score_column].to_numpy(dtype=float)
        if not np.isfinite(score).all():
            raise ValueError("目标分数包含空值或无穷值")
    if args.objective == "target_score" and args.score_target is None:
        raise ValueError("target_score 需要 --score-target")

    factor_eq = _parse_pairs(args.factor_eq)
    factor_min = _parse_pairs(args.factor_min)
    factor_max = _parse_pairs(args.factor_max)
    factor_columns = sorted(set(factor_eq) | set(factor_min) | set(factor_max))
    missing_columns = [column for column in factor_columns if column not in universe]
    if missing_columns:
        raise ValueError(f"候选池缺少约束因子列: {missing_columns}")
    if factor_columns and not np.isfinite(universe[factor_columns].to_numpy(dtype=float)).all():
        raise ValueError("因子约束列包含空值或无穷值")

    default_max = max(1.0 / count, min(0.20, 2.0 / count))
    maximum_weight = args.max_weight if args.max_weight is not None else default_max
    minimum_weight = args.min_weight
    if minimum_weight * count > 1 + 1e-12 or maximum_weight * count < 1 - 1e-12:
        raise ValueError(f"权重上下界不可行: N={count}, min={minimum_weight}, max={maximum_weight}")

    current = None
    if args.current_weight_column:
        if args.current_weight_column not in universe:
            raise ValueError(f"候选池缺少当前权重列 {args.current_weight_column!r}")
        current = universe[args.current_weight_column].fillna(0.0).to_numpy(dtype=float)
        if current.sum() > 0:
            current = current / current.sum()
    constraints: list[dict[str, Callable]] = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    for column, target in factor_eq.items():
        values = universe[column].to_numpy(dtype=float)
        constraints.append({"type": "eq", "fun": lambda w, v=values, t=target: float(np.dot(w, v) - t)})
    for column, target in factor_min.items():
        values = universe[column].to_numpy(dtype=float)
        constraints.append({"type": "ineq", "fun": lambda w, v=values, t=target: float(np.dot(w, v) - t)})
    for column, target in factor_max.items():
        values = universe[column].to_numpy(dtype=float)
        constraints.append({"type": "ineq", "fun": lambda w, v=values, t=target: float(t - np.dot(w, v))})
    if args.max_turnover is not None:
        if current is None:
            raise ValueError("--max-turnover 需要 --current-weight-column")
        constraints.append({"type": "ineq", "fun": lambda w: float(args.max_turnover - 0.5 * np.abs(w - current).sum())})

    def portfolio_variance(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    def objective(weights: np.ndarray) -> float:
        variance = portfolio_variance(weights)
        portfolio_return = float(np.dot(weights, expected))
        if args.objective == "min_variance":
            return variance
        if args.objective == "max_return":
            return -portfolio_return
        if args.objective == "max_sharpe":
            return -(portfolio_return - args.risk_free_rate) / math.sqrt(max(variance, 1e-16))
        if args.objective == "mean_variance":
            return args.risk_aversion * variance - portfolio_return
        if args.objective == "max_score":
            return -float(np.dot(weights, score))
        if args.objective == "min_score":
            return float(np.dot(weights, score))
        if args.objective == "target_score":
            gap = float(np.dot(weights, score) - args.score_target)
            return gap * gap + args.risk_aversion * variance
        raise ValueError(f"未知目标: {args.objective}")

    initial = np.full(count, 1.0 / count)
    if current is not None and np.all((current >= minimum_weight) & (current <= maximum_weight)):
        initial = current.copy()
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(minimum_weight, maximum_weight)] * count,
        constraints=constraints,
        options={"maxiter": args.max_iterations, "ftol": args.tolerance, "disp": False},
    )
    weights = np.asarray(result.x, dtype=float)
    report = _constraint_report(weights, universe, factor_eq, factor_min, factor_max, current, args.max_turnover)
    tolerance = max(1e-6, args.tolerance * 10)
    feasible = (
        abs(report["weight_sum"] - 1.0) <= tolerance
        and report["min_weight"] >= minimum_weight - tolerance
        and report["max_weight"] <= maximum_weight + tolerance
        and all(abs(value) <= tolerance for value in report["factor_eq_residuals"].values())
        and all(value >= -tolerance for value in report["factor_min_slacks"].values())
        and all(value >= -tolerance for value in report["factor_max_slacks"].values())
        and report.get("turnover_slack", 0.0) >= -tolerance
    )
    if not result.success or not feasible or not np.isfinite(weights).all():
        raise RuntimeError(f"优化失败或约束不满足: solver={result.message}; checks={report}")

    output_frame = universe.copy()
    output_frame.insert(1, "weight", weights)
    output = _path(args.output)
    _atomic_write(output_frame, output)
    variance = portfolio_variance(weights)
    portfolio_return = float(np.dot(weights, expected))
    summary = {
        "status": "ok",
        "objective": args.objective,
        "objective_value": float(result.fun),
        "solver": "scipy.optimize.SLSQP",
        "solver_message": str(result.message),
        "iterations": int(result.nit),
        "as_of_requested": str(requested_as_of),
        "return_window_start": str(returns.dropna(how="all").index.min()),
        "return_window_end": str(returns.dropna(how="all").index.max()),
        "observation_counts": observation_counts,
        "expected_return_source": args.expected_return_column or "historical_annualized_mean",
        "portfolio_expected_return_annual": portfolio_return,
        "portfolio_volatility_annual": math.sqrt(max(variance, 0.0)),
        "portfolio_sharpe": (portfolio_return - args.risk_free_rate) / math.sqrt(variance) if variance > 0 else None,
        "weight_bounds": [minimum_weight, maximum_weight],
        "used_default_max_weight": args.max_weight is None,
        "checks": report,
        "output": str(output),
    }
    if score is not None:
        summary["portfolio_score"] = float(np.dot(weights, score))
        summary["score_column"] = args.score_column
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True, help="含 code 和可选因子/预期收益/当前权重列的 CSV 或 Parquet")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--objective", required=True, choices=(
        "min_variance", "max_return", "max_sharpe", "mean_variance", "max_score", "min_score", "target_score"
    ))
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-column", default="code")
    parser.add_argument("--returns-path", default=str(DEFAULT_RETURNS))
    parser.add_argument("--return-column")
    parser.add_argument("--years", type=int, default=5)
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
    parser.add_argument("--current-weight-column")
    parser.add_argument("--max-turnover", type=float, help="单边换手率上限，0.20 表示 20%%")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.years <= 0 or args.min_observations < 2:
        parser.error("--years 必须大于 0，--min-observations 必须至少为 2")
    if args.min_weight < 0 or (args.max_weight is not None and args.max_weight <= 0):
        parser.error("当前版本只支持多头权重；上下界必须非负")
    if args.max_turnover is not None and args.max_turnover < 0:
        parser.error("--max-turnover 必须非负")
    solve(args)


if __name__ == "__main__":
    main()
