# Scientific Method

## Frozen question

Does correcting the invalid conditional-tail calculation and blocking unsupported trust updates make the Cash/Buy-and-Hold Router economically better than constant BTC exposure with the same mean exposure?

## Repair only

No expert, feature, posterior, transaction cost, risk limit, trust learning rate, or tail penalty is changed. DQN and specialist training remain prohibited.

For regime posterior weights `q`, the repair estimates the upper 5% tail directly under the weighted empirical distribution. Fractional boundary mass is included. Therefore the conditional tail estimate cannot exceed the maximum observed loss.

## Evidence support

A regime row updates only when both conditions hold during the completed return month:

- posterior mass at least 20 expected observations;
- Kish effective sample size at least 20.

Twenty is fixed as `ceil(1 / 0.05)`, the minimum effective support required for one 5% tail observation. It was not selected by portfolio-return search. If either condition fails, the trust row remains exactly unchanged.

## Primary comparator and economic gate

The primary comparator holds constant BTC exposure equal to the repaired Router's mean exposure and uses the same 10-basis-point cost accounting.

Every condition must pass:

1. positive full-period terminal-growth advantage;
2. one-sided circular 30-day block-bootstrap `p < 0.05` using 10,000 deterministic resamples;
3. positive growth advantage in at least two of three calendar years;
4. maximum drawdown no worse than matched exposure;
5. daily 95% loss CVaR no worse than matched exposure.

Passing is development evidence for the Router repair only. It cannot override the Stage-0 specialist rejection.
