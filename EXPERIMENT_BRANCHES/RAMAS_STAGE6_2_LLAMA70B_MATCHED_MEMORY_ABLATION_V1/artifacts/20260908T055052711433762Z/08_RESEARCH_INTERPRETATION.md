# RAMAS Stage 6.2: matched memory experiment

Evidence flag: **INCONCLUSIVE**.

The trace covers 2021-01-02 through 2025-05-28 (1608 days). 2021 is a warm-up year. The primary comparison uses completed returns after 2021, with each arm's state continuing across years.

All post-2021 results are **reused historical diagnostics**, not a fresh untouched out-of-sample test. Any incomplete calendar year is reported only through the last available date.

After warm-up, the memory arm's average daily log return minus the no-memory arm's is **-0.0017 basis points/day**. The two-sided 95% paired block interval is **[-0.1265, +0.0992]** basis points/day (1 basis point = 0.01%).

The interval includes zero. This run does not establish a reliable directional benefit from memory under this diagnostic.

The interval resamples the two realized return paths together in calendar blocks. It does not rerun learners on alternative histories and is not a calibrated p-value. It cannot establish that more years of memory cause steadily better decisions.

## Return and risk

| Scope | Strategy | Days | Return | Largest drawdown | Sharpe | Worst-5%-day average loss |
|---|---|---:|---:|---:|---:|---:|
| FULL | MEMORY | 1608 | +116.30% | +45.61% | 0.774 | +3.30% |
| FULL | NO_MEMORY | 1608 | +116.34% | +45.61% | 0.774 | +3.30% |
| FULL | CASH | 1608 | +0.00% | -0.00% | undefined | +0.00% |
| FULL | BTC_BUY_AND_HOLD | 1608 | +266.76% | +76.64% | 0.788 | +7.28% |
| FULL | STATIC45_REBALANCED_SECONDARY | 1608 | +117.98% | +45.05% | 0.781 | +3.27% |
| POST2021 | MEMORY | 1244 | +67.73% | +37.09% | 0.745 | +2.92% |
| POST2021 | NO_MEMORY | 1244 | +67.77% | +37.09% | 0.745 | +2.92% |
| POST2021 | CASH | 1244 | +0.00% | -0.00% | undefined | +0.00% |
| POST2021 | BTC_BUY_AND_HOLD | 1244 | +133.21% | +66.95% | 0.731 | +6.46% |
| POST2021 | STATIC45_REBALANCED_SECONDARY | 1244 | +64.48% | +36.85% | 0.723 | +2.91% |
| 2021 | MEMORY | 364 | +28.95% | +27.32% | 0.882 | +4.15% |
| 2021 | NO_MEMORY | 364 | +28.95% | +27.32% | 0.882 | +4.15% |
| 2021 | CASH | 364 | +0.00% | -0.00% | undefined | +0.00% |
| 2021 | BTC_BUY_AND_HOLD | 364 | +57.27% | +53.11% | 0.964 | +9.03% |
| 2021 | STATIC45_REBALANCED_SECONDARY | 364 | +32.53% | +26.91% | 0.956 | +4.06% |
| 2022 | MEMORY | 365 | -34.58% | +37.09% | -1.410 | +3.87% |
| 2022 | NO_MEMORY | 365 | -34.58% | +37.09% | -1.410 | +3.87% |
| 2022 | CASH | 365 | +0.00% | -0.00% | undefined | +0.00% |
| 2022 | BTC_BUY_AND_HOLD | 365 | -64.22% | +66.95% | -1.281 | +8.82% |
| 2022 | STATIC45_REBALANCED_SECONDARY | 365 | -33.81% | +36.85% | -1.288 | +3.97% |
| 2023 | MEMORY | 365 | +59.28% | +9.37% | 2.353 | +2.09% |
| 2023 | NO_MEMORY | 365 | +59.28% | +9.37% | 2.353 | +2.09% |
| 2023 | CASH | 365 | +0.00% | -0.00% | undefined | +0.00% |
| 2023 | BTC_BUY_AND_HOLD | 365 | +155.62% | +20.02% | 2.352 | +4.49% |
| 2023 | STATIC45_REBALANCED_SECONDARY | 365 | +55.99% | +9.37% | 2.346 | +2.02% |
| 2024 | MEMORY | 366 | +50.73% | +12.53% | 1.742 | +2.70% |
| 2024 | NO_MEMORY | 366 | +51.01% | +12.58% | 1.746 | +2.70% |
| 2024 | CASH | 366 | +0.00% | -0.00% | undefined | +0.00% |
| 2024 | BTC_BUY_AND_HOLD | 366 | +121.32% | +26.14% | 1.756 | +5.59% |
| 2024 | STATIC45_REBALANCED_SECONDARY | 366 | +47.74% | +11.55% | 1.749 | +2.52% |
| 2025 | MEMORY | 148 | +6.78% | +13.04% | 0.860 | +2.46% |
| 2025 | NO_MEMORY | 148 | +6.61% | +13.04% | 0.840 | +2.46% |
| 2025 | CASH | 148 | +0.00% | -0.00% | undefined | +0.00% |
| 2025 | BTC_BUY_AND_HOLD | 148 | +15.21% | +28.11% | 0.949 | +5.64% |
| 2025 | STATIC45_REBALANCED_SECONDARY | 148 | +7.82% | +13.20% | 0.941 | +2.54% |

Cash has no assumed interest. Buy-and-hold buys once on the first original day. The 45% BTC control is a secondary exposure diagnostic selected from the previous Stage6.1 result and fixed for this rerun. Report boundaries do not restart positions or charge new entry costs. Model-running costs are excluded.

## Forecast market conditions

| Scope | Forecast | Strategy | Days | Average daily growth | Daily volatility | Worst-5%-day average loss |
|---|---|---|---:|---:|---:|---:|
| FULL | bull | MEMORY | 531 | +0.1642% | +1.52% | +3.06% |
| FULL | bear | MEMORY | 503 | +0.0051% | +1.48% | +3.54% |
| FULL | mix | MEMORY | 574 | -0.0219% | +1.32% | +3.27% |
| FULL | bull | NO_MEMORY | 531 | +0.1643% | +1.52% | +3.06% |
| FULL | bear | NO_MEMORY | 503 | +0.0051% | +1.48% | +3.54% |
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
| POST2021 | bull | MEMORY | 388 | +0.1347% | +1.28% | +2.60% |
| POST2021 | bear | MEMORY | 380 | -0.0034% | +1.38% | +3.40% |
| POST2021 | mix | MEMORY | 476 | +0.0017% | +1.17% | +2.75% |
| POST2021 | bull | NO_MEMORY | 388 | +0.1348% | +1.29% | +2.61% |
| POST2021 | bear | NO_MEMORY | 380 | -0.0034% | +1.38% | +3.40% |
| POST2021 | mix | NO_MEMORY | 476 | +0.0016% | +1.17% | +2.75% |
| POST2021 | bull | CASH | 388 | +0.0000% | +0.00% | +0.00% |
| POST2021 | bear | CASH | 380 | +0.0000% | +0.00% | +0.00% |
| POST2021 | mix | CASH | 476 | +0.0000% | +0.00% | +0.00% |
| POST2021 | bull | BTC_BUY_AND_HOLD | 388 | +0.2645% | +2.61% | +5.25% |
| POST2021 | bear | BTC_BUY_AND_HOLD | 380 | -0.0274% | +3.33% | +8.03% |
| POST2021 | mix | BTC_BUY_AND_HOLD | 476 | -0.0155% | +2.52% | +5.99% |
| POST2021 | bull | STATIC45_REBALANCED_SECONDARY | 388 | +0.1269% | +1.17% | +2.36% |
| POST2021 | bear | STATIC45_REBALANCED_SECONDARY | 380 | +0.0008% | +1.50% | +3.61% |
| POST2021 | mix | STATIC45_REBALANCED_SECONDARY | 476 | +0.0005% | +1.14% | +2.70% |

The labels are forecasts available when decisions were made. Average daily growth is geometric. Bull, Bear and Mix days are scattered through time; they are not separate investable strategies. Their daily returns retain actual trading costs. We therefore do not calculate a misleading drawdown after joining disconnected regime days. The CSV also contains year-by-regime breakdowns.

## What this test can establish

The primary contrast tests whether enabling retrieval of completed experience improves this system's realized post-warm-up log growth. Both arms must generate their own actions, update their own trust, and carry their own portfolios forward. The no-memory arm still has its own updating trust ledger; only episodic retrieval and its supplied summary are disabled. Consequently, this is a test of episodic retrieval's added value, not a comparison against a system with no learning of any kind. Memory citations, growing storage, and action disagreement show mechanisms or behavioral changes; they are not proof of useful learning. Return, drawdown, tail risk, and regime-specific trade-offs should all be reported. A weak metric does not erase a valid technical finding, and a positive metric does not prove general superiority.

The diagnostics CSV counts invalid/fallback days, action disagreements that produce identical exposure, and each arm's desired-versus-final exposure changes. Identical final exposure can hide action differences because trust blending and the common risk projection limit actual trading changes. Transport interruptions are recorded by the runtime and pause the run for resume; they are not fabricated into trading observations.

This report never stops the experiment for economic underperformance. The complete trace is retained for scientific interpretation. Additional independently controlled repetitions and a protocol fixed before new data would be needed for stronger generalization claims.
