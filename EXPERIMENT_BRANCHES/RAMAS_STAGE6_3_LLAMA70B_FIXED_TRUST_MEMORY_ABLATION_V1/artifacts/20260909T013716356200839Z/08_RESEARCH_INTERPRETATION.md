# RAMAS Stage 6.3: matched memory experiment

Evidence flag: **INCONCLUSIVE**.

The trace covers 2021-01-02 through 2025-05-28 (1608 days). 2021 is a warm-up year. The primary comparison uses completed returns after 2021, with each arm's state continuing across years.

All post-2021 results are **reused historical diagnostics**, not a fresh untouched out-of-sample test. Any incomplete calendar year is reported only through the last available date.

Both arms keep agent trust fixed at **beta=0.05** on every day and regime. The same Llama model generates each arm's own fresh decisions. Only the memory arm receives retrieved completed experiences and their summary. Model weights do not change. Fixed trust does not bypass allocation constraints or guarantee different trades.

The shared legacy forecast-target and simulated execution assumptions remain limitations. Correct replay and arithmetic alone do not establish realistic executable performance.

After warm-up, the memory arm's average daily log return minus the no-memory arm's is **+0.0369 basis points/day**. The two-sided 95% paired block interval is **[-0.0340, +0.1221]** basis points/day (1 basis point = 0.01%).

The interval includes zero. This run does not establish a reliable directional benefit from memory under this diagnostic.

The interval resamples the two realized return paths together in calendar blocks. It does not rerun learners on alternative histories and is not a calibrated p-value. It cannot establish that more years of memory cause steadily better decisions.

## Return and risk

| Scope | Strategy | Days | Return | Annualized volatility | Largest drawdown | Sharpe | Worst-5%-day average loss |
|---|---|---:|---:|---:|---:|---:|---:|
| FULL | MEMORY | 1608 | +116.78% | +27.30% | +44.96% | 0.780 | +3.25% |
| FULL | NO_MEMORY | 1608 | +115.65% | +27.35% | +45.18% | 0.775 | +3.26% |
| FULL | CASH | 1608 | +0.00% | +0.00% | -0.00% | undefined | +0.00% |
| FULL | BTC_BUY_AND_HOLD | 1608 | +266.76% | +61.13% | +76.64% | 0.788 | +7.28% |
| FULL | STATIC45_REBALANCED_SECONDARY | 1608 | +117.98% | +27.51% | +45.05% | 0.781 | +3.27% |
| FULL | STATIC50_REBALANCED | 1608 | +133.08% | +30.56% | +48.87% | 0.782 | +3.64% |
| POST2021 | MEMORY | 1244 | +68.62% | +23.93% | +36.25% | 0.761 | +2.86% |
| POST2021 | NO_MEMORY | 1244 | +67.85% | +23.96% | +36.25% | 0.754 | +2.86% |
| POST2021 | CASH | 1244 | +0.00% | +0.00% | -0.00% | undefined | +0.00% |
| POST2021 | BTC_BUY_AND_HOLD | 1244 | +133.21% | +53.92% | +66.95% | 0.731 | +6.46% |
| POST2021 | STATIC45_REBALANCED_SECONDARY | 1244 | +64.48% | +24.26% | +36.85% | 0.723 | +2.91% |
| POST2021 | STATIC50_REBALANCED | 1244 | +71.79% | +26.96% | +40.21% | 0.724 | +3.23% |
| 2021 | MEMORY | 364 | +28.56% | +36.58% | +27.41% | 0.872 | +4.16% |
| 2021 | NO_MEMORY | 364 | +28.48% | +36.69% | +27.72% | 0.868 | +4.17% |
| 2021 | CASH | 364 | +0.00% | +0.00% | -0.00% | undefined | +0.00% |
| 2021 | BTC_BUY_AND_HOLD | 364 | +57.27% | +81.14% | +53.11% | 0.964 | +9.03% |
| 2021 | STATIC45_REBALANCED_SECONDARY | 364 | +32.53% | +36.51% | +26.91% | 0.956 | +4.06% |
| 2021 | STATIC50_REBALANCED | 364 | +35.67% | +40.57% | +29.61% | 0.957 | +4.52% |
| 2022 | MEMORY | 365 | -33.31% | +27.03% | +36.25% | -1.363 | +3.77% |
| 2022 | NO_MEMORY | 365 | -33.31% | +27.03% | +36.25% | -1.363 | +3.77% |
| 2022 | CASH | 365 | +0.00% | +0.00% | -0.00% | undefined | +0.00% |
| 2022 | BTC_BUY_AND_HOLD | 365 | -64.22% | +64.02% | +66.95% | -1.281 | +8.82% |
| 2022 | STATIC45_REBALANCED_SECONDARY | 365 | -33.81% | +28.81% | +36.85% | -1.288 | +3.97% |
| 2022 | STATIC50_REBALANCED | 365 | -37.09% | +32.01% | +40.21% | -1.287 | +4.41% |
| 2023 | MEMORY | 365 | +61.32% | +21.42% | +9.61% | 2.341 | +2.19% |
| 2023 | NO_MEMORY | 365 | +61.06% | +21.49% | +9.64% | 2.327 | +2.20% |
| 2023 | CASH | 365 | +0.00% | +0.00% | -0.00% | undefined | +0.00% |
| 2023 | BTC_BUY_AND_HOLD | 365 | +155.62% | +44.02% | +20.02% | 2.352 | +4.49% |
| 2023 | STATIC45_REBALANCED_SECONDARY | 365 | +55.99% | +19.81% | +9.37% | 2.346 | +2.02% |
| 2023 | STATIC50_REBALANCED | 365 | +63.52% | +22.01% | +10.38% | 2.346 | +2.24% |
| 2024 | MEMORY | 366 | +48.47% | +24.30% | +12.31% | 1.744 | +2.59% |
| 2024 | NO_MEMORY | 366 | +48.14% | +24.35% | +12.54% | 1.732 | +2.61% |
| 2024 | CASH | 366 | +0.00% | +0.00% | -0.00% | undefined | +0.00% |
| 2024 | BTC_BUY_AND_HOLD | 366 | +121.32% | +53.13% | +26.14% | 1.756 | +5.59% |
| 2024 | STATIC45_REBALANCED_SECONDARY | 366 | +47.74% | +23.91% | +11.55% | 1.749 | +2.52% |
| 2024 | STATIC50_REBALANCED | 366 | +53.77% | +26.57% | +12.87% | 1.749 | +2.80% |
| 2025 | MEMORY | 148 | +5.57% | +19.93% | +13.38% | 0.770 | +2.31% |
| 2025 | NO_MEMORY | 148 | +5.48% | +19.94% | +13.38% | 0.760 | +2.31% |
| 2025 | CASH | 148 | +0.00% | +0.00% | -0.00% | undefined | +0.00% |
| 2025 | BTC_BUY_AND_HOLD | 148 | +15.21% | +49.77% | +28.11% | 0.949 | +5.64% |
| 2025 | STATIC45_REBALANCED_SECONDARY | 148 | +7.82% | +22.40% | +13.20% | 0.941 | +2.54% |
| 2025 | STATIC50_REBALANCED | 148 | +8.60% | +24.88% | +14.62% | 0.942 | +2.82% |

Cash has no assumed interest. Buy-and-hold buys once on the first original day. The static50 control rebalances daily to 50% BTC/50% cash. The 45% control remains an exploratory exposure diagnostic chosen from an earlier result. Report boundaries do not restart positions or charge new entry costs. Model-running costs are excluded.

## Forecast market conditions

| Scope | Forecast | Strategy | Days | Average daily growth | Daily volatility | Worst-5%-day average loss |
|---|---|---|---:|---:|---:|---:|
| FULL | bull | MEMORY | 531 | +0.1647% | +1.51% | +3.02% |
| FULL | bear | MEMORY | 503 | +0.0051% | +1.45% | +3.46% |
| FULL | mix | MEMORY | 574 | -0.0219% | +1.32% | +3.27% |
| FULL | bull | NO_MEMORY | 531 | +0.1637% | +1.51% | +3.05% |
| FULL | bear | NO_MEMORY | 503 | +0.0051% | +1.45% | +3.46% |
| FULL | mix | NO_MEMORY | 574 | -0.0219% | +1.32% | +3.27% |
| FULL | bull | CASH | 531 | +0.0000% | +0.00% | +0.00% |
| FULL | bear | CASH | 503 | +0.0000% | +0.00% | +0.00% |
| FULL | mix | CASH | 574 | +0.0000% | +0.00% | +0.00% |
| FULL | bull | BTC_BUY_AND_HOLD | 531 | +0.3252% | +3.15% | +6.22% |
| FULL | bear | BTC_BUY_AND_HOLD | 503 | -0.0097% | +3.63% | +8.47% |
| FULL | mix | BTC_BUY_AND_HOLD | 574 | -0.0655% | +2.82% | +6.90% |
| FULL | bull | STATIC45_REBALANCED_SECONDARY | 531 | +0.1578% | +1.42% | +2.80% |
| FULL | bear | STATIC45_REBALANCED_SECONDARY | 503 | +0.0113% | +1.63% | +3.81% |
| FULL | mix | STATIC45_REBALANCED_SECONDARY | 574 | -0.0200% | +1.27% | +3.11% |
| FULL | bull | STATIC50_REBALANCED | 531 | +0.1742% | +1.57% | +3.11% |
| FULL | bear | STATIC50_REBALANCED | 503 | +0.0110% | +1.81% | +4.24% |
| FULL | mix | STATIC50_REBALANCED | 574 | -0.0232% | +1.41% | +3.45% |
| POST2021 | bull | MEMORY | 388 | +0.1361% | +1.25% | +2.52% |
| POST2021 | bear | MEMORY | 380 | -0.0034% | +1.35% | +3.27% |
| POST2021 | mix | MEMORY | 476 | +0.0017% | +1.17% | +2.75% |
| POST2021 | bull | NO_MEMORY | 388 | +0.1349% | +1.26% | +2.54% |
| POST2021 | bear | NO_MEMORY | 380 | -0.0034% | +1.35% | +3.27% |
| POST2021 | mix | NO_MEMORY | 476 | +0.0017% | +1.17% | +2.75% |
| POST2021 | bull | CASH | 388 | +0.0000% | +0.00% | +0.00% |
| POST2021 | bear | CASH | 380 | +0.0000% | +0.00% | +0.00% |
| POST2021 | mix | CASH | 476 | +0.0000% | +0.00% | +0.00% |
| POST2021 | bull | BTC_BUY_AND_HOLD | 388 | +0.2645% | +2.61% | +5.25% |
| POST2021 | bear | BTC_BUY_AND_HOLD | 380 | -0.0274% | +3.33% | +8.03% |
| POST2021 | mix | BTC_BUY_AND_HOLD | 476 | -0.0155% | +2.52% | +5.99% |
| POST2021 | bull | STATIC45_REBALANCED_SECONDARY | 388 | +0.1269% | +1.17% | +2.36% |
| POST2021 | bear | STATIC45_REBALANCED_SECONDARY | 380 | +0.0008% | +1.50% | +3.61% |
| POST2021 | mix | STATIC45_REBALANCED_SECONDARY | 476 | +0.0005% | +1.14% | +2.70% |
| POST2021 | bull | STATIC50_REBALANCED | 388 | +0.1402% | +1.31% | +2.63% |
| POST2021 | bear | STATIC50_REBALANCED | 380 | -0.0004% | +1.67% | +4.01% |
| POST2021 | mix | STATIC50_REBALANCED | 476 | -0.0002% | +1.26% | +3.00% |

The labels are the archived router forecasts supplied to the agents, not verified realized market regimes. The target-alignment limitation still applies. Average daily growth is geometric. Bull, Bear and Mix days are scattered through time; they are not separate investable strategies. Their daily returns retain actual trading costs. We therefore do not calculate a misleading drawdown after joining disconnected regime days. The CSV also contains year-by-regime breakdowns.

## What this test can establish

The primary contrast tests whether enabling retrieval of completed experience improves this system's realized post-warm-up log growth. Both arms generate their own actions and carry their own portfolios and outcome ledgers forward. Trust is fixed at beta=0.05 in both. The no-memory arm still owns an outcome ledger, but sees no retrieved episodes or summary. Neither arm updates trust or model weights. Consequently, this is a test of episodic retrieval's added value, not a comparison against a system with no learning of any kind. Memory citations, growing storage, and action disagreement show mechanisms or behavioral changes; they are not proof of useful learning. Return, drawdown, tail risk, and regime-specific trade-offs should all be reported. A weak metric does not erase a valid technical finding, and a positive metric does not prove general superiority.

The diagnostics CSV counts invalid/fallback days, action disagreements that produce identical exposure, and each arm's desired-versus-final exposure changes. Identical final exposure can hide action differences because trust blending and the common risk projection limit actual trading changes. Transport interruptions are recorded by the runtime and pause the run for resume; they are not fabricated into trading observations.

This report never stops the experiment for economic underperformance. The complete trace is retained for scientific interpretation. Additional independently controlled repetitions and a protocol fixed before new data would be needed for stronger generalization claims.

## Supplementary memory-by-trust comparison

The optional archived adaptive-trust pair is used only as a secondary four-cell diagnostic. Its responses are never reused as decisions in either new fixed-trust arm.

Adaptive memory gap minus fixed memory gap: **-0.0386 basis points/day**, with a joint paired-block interval **[-0.1846, +0.0897]**. All four cells use the same resampled calendar indices. This interaction does not by itself establish useful continuous learning.

## Output detail

11_MONTHLY_METRICS.csv reports every recorded month; 12_DAILY_METRICS.csv records each strategy's daily return, BTC/cash fractions, turnover, paid cost, original-path wealth and running drawdown. 07_ACTION_AND_STATE_DIAGNOSTICS.csv tracks advice, desired exposure, final exposure, suppressed changes and earlier-year memory. 14_COMPUTATIONAL_DIAGNOSTICS.json records observed input/output tokens and latency from committed trading calls, excluding preflight. Missing usage is marked unavailable, never zero-priced.
