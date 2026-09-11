# 工作流参考

## 首次路径确认与每次 prepare

脚本从当前工作目录解析以下默认文件。若 Skill 安装在其他位置，代码仍可运行；数据位置由工作目录、`FACTOR_PREMIUM_PROJECT_ROOT` 或 `--config` 决定。

安装第三方依赖：

```powershell
python -m pip install -r "{SKILL_DIR}/requirements.txt"
```

项目原先使用的 `BaseDataLoader`、`ReportDataLoader` 和季度财务数据转换函数已经复制并整理到 `scripts/vendor/`，运行时不再导入 Skill 文件夹之外的 Python 模块。行情和财报属于用户数据，不随开源 Skill 分发。

以下是默认文件：

| 用途 | 文件 |
|---|---|
| 股票日线 | `data/stock/stock_bar_1day.parquet` |
| 市值与 PB | `data/stock/capital.parquet` |
| 基金/指数日线 | `data/index/index_bar_1day.parquet` |
| BAB 特征 | `daily_factors/BAB/bab.parquet` |
| QMJ 特征 | `daily_factors/QMJ/qmj.parquet` |
| 合并因子溢价 | `fund_return_analysis/premium.parquet` |
| 资产负债表 | `data/stock/report_balance.parquet` |
| 利润表 | `data/stock/report_income.parquet` |
| 现金流量表 | `data/stock/report_cashflow.parquet` |

所有长表至少使用 `datetime`、`code`。行情表使用 `close`、`factor`；市值表使用 `market_cap`、`pb_ratio`；财报表还需 `report_period`。

默认路径无法验证时，请用户提供 JSON，不要自行猜测：

可以先复制 `assets/factor_data_paths.example.json`，再填写实际路径。

```json
{
  "stock_bar": "D:/data/stock_bar_1day.parquet",
  "capital": "D:/data/capital.parquet",
  "market_bar": "D:/data/index_bar_1day.parquet",
  "fund_bar": "D:/data/fund_bar_1day.parquet",
  "report_balance": "D:/data/report_balance.parquet",
  "report_income": "D:/data/report_income.parquet",
  "report_cashflow": "D:/data/report_cashflow.parquet",
  "bab_factor": "D:/outputs/bab.parquet",
  "qmj_factor": "D:/outputs/qmj.parquet",
  "premium": "D:/outputs/premium.parquet"
}
```

每次先预检：

```powershell
python "{SKILL_DIR}/scripts/prepare_data.py" --config factor_data_paths.json --target-end 2025-12-31 --dry-run
```

确认后执行补齐和更新：

```powershell
python "{SKILL_DIR}/scripts/prepare_data.py" --config factor_data_paths.json --target-end 2025-12-31
```

prepare 的顺序固定为 BAB → QMJ → 六因子溢价。BAB 使用 252 日股票收益对市场当期及 5 个滞后收益回归；QMJ 维护 growth、profitability、safety 和综合 qmj，其中 safety 使用刚更新的 BAB。若所需市场指数默认代码不是 `000985.SH`，使用 `--market-code` 指定。

## Fama–French 风格因子在哪里更新

`scripts/prepare_data.py` 在补齐 BAB 和 QMJ 后调用 `scripts/factor_pipeline.py` 的 `update_premiums()`。该函数在每个共同交易日一次性写出：

- `mkt_rtn`：本地股票池按市值加权的市场收益；当前没有减无风险利率。
- `smb`：按市值二分、按账面市值比三分后的 2×3 组合规模差。
- `hml`：同一 2×3 组合中的高账面市值比减低账面市值比。
- `umd`：过去 12 个月剔除最近 1 个月的动量信号，最高 30% 减最低 30%。
- `bab`：使用已补齐的个股 BAB 信号形成 beta 中性化多空收益。
- `qmj`：按市值二分、按 QMJ 三分后的高质量减低质量收益。

因此，Fama–French、UMD、BAB 和 QMJ 的最终溢价都由同一个 `update-premiums` 入口增量维护。这里是本地口径，不应称为 Kenneth French 官方下载数据。

## 单独重算六因子溢价

先查看将要更新的范围：

```powershell
python "{SKILL_DIR}/scripts/factor_pipeline.py" update-premiums --dry-run
```

增量更新：

```powershell
python "{SKILL_DIR}/scripts/factor_pipeline.py" update-premiums
```

指定范围重算并覆盖该范围：

```powershell
python "{SKILL_DIR}/scripts/factor_pipeline.py" update-premiums --start 2025-01-01 --end 2025-12-31 --rebuild-range
```

可交易口径使用上一交易日特征：

```powershell
python "{SKILL_DIR}/scripts/factor_pipeline.py" update-premiums --start 2025-01-01 --signal-lag 1 --rebuild-range
```

日常调用不应直接从这里开始，应先执行 prepare。这里只用于指定区间复核或重算。

## 三模型滚动暴露

```powershell
python "{SKILL_DIR}/scripts/factor_pipeline.py" exposures --codes 000933.SH 000985.SH --as-of 2025-12-31 --years 5 --models capm four_factor six_factor --output fund_return_analysis/outputs/20251231/exposures.parquet
```

按月维护一段时间内的三模型暴露：

```powershell
python "{SKILL_DIR}/scripts/factor_pipeline.py" exposures --codes 000933.SH 000985.SH --start-as-of 2020-01-01 --as-of 2025-12-31 --frequency monthly --years 5 --output rolling_exposures.parquet
```

输出每个“截面日期×标的×模型”一行，列包括 `as_of`、`model`、`alpha_daily`、各因子 beta、`r2`、`n_obs`、`window_start`、`window_end` 和 `status`。自定义收益文件可用 `--returns-path`；若文件已经含日收益，增加 `--return-column rtn`。

## 3. 选择前/后 10%

输入建议为宽表，每行一个标的，例如：

```text
code,qmj,bab,expected_return
FUND_A,0.31,-0.10,0.08
FUND_B,-0.42,0.15,0.05
```

选择 QMJ 前 10%：

```powershell
python "{SKILL_DIR}/scripts/factor_pipeline.py" select --input factor_values.csv --factor qmj --side top --pct 0.10 --output selected.csv
```

选择后 10% 时把 `--side` 改为 `bottom`。若用户直接给候选代码，不运行本步骤，直接准备只有 `code` 列的候选池文件。

## 单期优化持仓

最小方差：

```powershell
python "{SKILL_DIR}/scripts/optimize_portfolio.py" --universe selected.csv --as-of 2025-12-31 --objective min_variance --output weights.csv
```

均值—方差（`expected_return` 按年化小数输入，例如 8% 写作 `0.08`）：

```powershell
python "{SKILL_DIR}/scripts/optimize_portfolio.py" --universe selected.csv --as-of 2025-12-31 --objective mean_variance --expected-return-column expected_return --risk-aversion 5 --max-weight 0.20 --output weights.csv
```

最大化某个用户给定分数：

```powershell
python "{SKILL_DIR}/scripts/optimize_portfolio.py" --universe selected.csv --as-of 2025-12-31 --objective max_score --score-column qmj --max-weight 0.20 --output weights.csv
```

在最小方差下要求组合 QMJ 暴露等于 0.30、BAB 暴露至少 -0.10：

```powershell
python "{SKILL_DIR}/scripts/optimize_portfolio.py" --universe selected.csv --as-of 2025-12-31 --objective min_variance --factor-eq qmj=0.30 --factor-min bab=-0.10 --output weights.csv
```

支持的目标：

- `min_variance`：最小年化方差。
- `max_return`：最大预期年化收益。
- `max_sharpe`：最大预期夏普比率。
- `mean_variance`：最小化 `风险厌恶系数 × 年化方差 - 预期年化收益`。
- `max_score` / `min_score`：最大化/最小化用户给定标的分数的加权平均。
- `target_score`：靠近 `--score-target`，同时用 `--risk-aversion` 惩罚方差。

约束参数：`--min-weight`、`--max-weight`、`--factor-eq`、`--factor-min`、`--factor-max`、`--current-weight-column` 和 `--max-turnover`。因子约束可重复传入。若约束不可行，不要静默放宽；把冲突约束报告给用户。

优化结果旁会生成同名 `.summary.json`，包含组合指标、约束检验和求解器信息。

## 指定频率的滚动优化与回测曲线

以月频调仓、每次选择 QMJ 前 10%、最小方差优化为例：

```powershell
python "{SKILL_DIR}/scripts/backtest_portfolio.py" --universe rolling_exposures.parquet --model six_factor --start 2020-01-01 --end 2025-12-31 --rebalance-frequency monthly --selection-factor qmj --selection-side top --selection-pct 0.10 --objective min_variance --transaction-cost-bps 5 --output-dir fund_return_analysis/outputs/20251231/backtest
```

调仓频率支持 `daily`、`weekly`、`monthly`、`quarterly`、`yearly`。回测程序默认会再次运行 prepare；只有同一任务中已经成功 prepare 时才传 `--skip-prepare`。

动态候选池应包含 `as_of` 或 `datetime`，程序在每个调仓日取不晚于当日的最新截面。静态候选池不需要日期列。调仓日计算的新权重从下一交易日开始承接收益，交易成本按 `transaction_cost_bps × 单边换手率` 扣除。

若用户要求使用基金实际费率但没有提供费率文件，先按候选基金代码抓取公开费率：

```powershell
python "{SKILL_DIR}/scripts/update_fund_fees.py" --input selected_fund_exposures.parquet --output fund_fees.parquet --holding-days 365
```

脚本取最低金额档的当前申购费率（有平台优惠时取页面展示的优惠费率）和指定持有天数对应的赎回费率。查询失败或页面缺项时，申购、赎回各自按已取得样本的均值替代；`*_imputed` 字段记录替代情况。当前公开费率用于历史回测属于静态费率近似，不能表述为历史时点真实费率。净值已经内含的管理费、托管费和销售服务费不重复扣除。

## 选择与权重归因三线图

若要判断表现来自标的选择还是权重优化，在普通滚动回测命令后增加 `--selection-weighting-comparison`。该选项要求同时提供 `--selection-factor`。

完整示例：

```powershell
python "{SKILL_DIR}/scripts/backtest_portfolio.py" --universe rolling_exposures.parquet --model capm --start 2018-01-01 --end 2025-12-31 --rebalance-frequency yearly --selection-factor max_mean_reversion --selection-side top --selection-pct 0.20 --objective max_score --score-column alpha_daily --max-weight 0.10 --transaction-cost-bps 5 --selection-weighting-comparison --output-dir fund_return_analysis/outputs/capm_comparison
```

诊断图包含：

- `optimized_portfolio`：原优化组合。
- `optimized_holdings_equal_weight`：保留优化组合中权重大于 `1e-10` 的同一批标的，改为等权。
- `factor_top_n_equal_weight`：按 `--selection-side` 取筛选因子最优的 N 只标的并等权；N 等于当期优化组合实际持仓数。

三组沿用相同调仓日、下一交易日生效规则、收益文件和 `transaction-cost-bps`。需要调整正权重判定阈值时使用 `--active-weight-tolerance`。

输出目录包含：

- `backtest_curve.png`：优化组合与基准净值曲线。
- `backtest_curve.parquet`：逐日毛收益、净收益、换手和净值。
- `rebalance_weights.parquet`：每个调仓日的选择结果与下一生效日权重。
- `backtest_summary.json`：绩效、频率、成本、目标和文件位置。
- `selection_weighting_comparison.png`：优化组合、同持仓等权和因子最优 N 只等权的三线净值图（启用诊断选项时）。
- `selection_weighting_comparison.parquet`：三组逐日净值。
- `selection_weighting_comparison_weights.parquet`：两组等权诊断基准的历次权重、调仓日、生效日和当期 N。
