# One accounting and reporting standard

Version 1.0; applies to all new suite reports. Historical files retain their original names and values.

**The headline measure is net return after trading costs.** Earlier phrases “net return” and “return after trading costs” denoted the same recorded quantity. This revision standardizes names and equations; it does not silently replace the cost model. Net here means after the modeled trading charge, not after taxes, management fees, inference, funding or all possible expenses. This research report is not a GIPS compliance claim.

At each one-day decision/holding interval, let p be the BTC weight just before trading, x the chosen weight, r_BTC the subsequent simple BTC return, and c=0.001 the cost per unit of one-way turnover. Cash earns zero interest; portfolios are long-only, unlevered and self-financing under the existing weight model.

- Turnover: u=abs(x-p).
- Trading-cost fraction of pretrade wealth: k=c*u.
- Daily net return after trading costs: r_net=(1-k)*(1+x*r_BTC)-1.
- Next pretrade BTC weight: p_next=x*(1+r_BTC)/(1+x*r_BTC).
- Wealth: V_next=V*(1+r_net).
- Daily net log return after trading costs: ell=ln(1+r_net).
- Period net return after trading costs: R_net=product(1+r_net)-1=exp(sum(ell))-1.

Cost is charged before the holding-period return. Do not replace it with x*r_BTC-c*u, which drops the cross-term. B&H starts from cash, buys BTC once, and pays its initial cost. Static50 rebalances to 50% BTC every day and pays its own turnover costs. Neither benchmark's holdings are reset at January. Monthly/yearly summaries reset only their reporting origin, never the actual account.

## A single numerical example

Suppose the account contains $10,000, p=0.40, x=0.50 and BTC subsequently gains 2%. Turnover is 0.10 and modeled cost is $10,000*0.001*0.10=$1. Invested wealth after cost is $9,999. The portfolio gains 1%, ending at $10,098.99.

Daily net return after trading costs = 1.009899-1 = 0.9899%.
Daily net log return after trading costs = ln(1.009899), approximately 0.9850% in log units. These are two representations, not two different profits. On the next day, begin from BTC weight .50*1.02/1.01=.504950495, not .50.

## Metric dictionary

| Measure | Definition and reporting convention |
|---|---|
| Net compounded return after trading costs | Product of daily wealth multipliers minus one; headline percentage |
| Cumulative net log return after trading costs | Sum of daily log returns; additive across time, not an additive attribution across interacting components |
| Mean daily net log return | Mean ell; paired differences reported in basis points/day (multiply by 10,000) |
| CAGR | exp(sum(ell)*365.25/calendar_days)-1 for contiguous windows at least365days; blank for shorter windows, including partial2025 |
| Daily volatility | Sample standard deviation of daily net simple returns, ddof=1 |
| Annualized volatility | Daily volatility *sqrt(365.25); conventional crypto-calendar scaling, not a guarantee of independence |
| Sharpe | sqrt(365.25)*mean(daily net simple excess return)/sample std; risk-free proxy0 here; undefined when denominator0 |
| Sortino | sqrt(365.25)*mean(daily net simple return)/sqrt(mean(min(r_net,0)^2)); target0, undefined denominator0 |
| Maximum drawdown | max(1-V/running_peak), including initial wealth in peaks; full-period peaks cross years |
| Calmar-style ratio over the reporting window | CAGR/maximum drawdown when both defined; explicitly not restricted to the conventional trailing36-month window |
| Daily expected shortfall95 | Average worst5% daily losses using fractional tail mass, not arbitrary ceil/floor sample counts |
| Costs and activity | Own turnover, cost fraction and cost per initial wealth unit, exposure, action count, latency and tokens when recorded |

The annualization constant is365.25, preserving Stage6.2/6.3. Do not switch between365,365.25 and252 across reports.

All machine-readable returns, volatility, drawdown and exposure fields use decimal fractions unless a field explicitly says bps or percentage points. CSV blanks/JSON null mean undefined/not applicable, not zero. Cash Sharpe/Sortino/Calmar remain undefined.

Daily, monthly and yearly records are all exported. Year2025 ends May28; the original2021 warm-up has364holding rows. A year-report drawdown can be smaller than full-period drawdown because the latter can begin at a previous year's peak.

Bull/Bear/Mix use the common archived forecast label, not future realized labels. Noncontiguous regime slices report counts, average daily returns, conditional daily volatility/ES, exposure and cumulative log contribution. They do not receive a fabricated tradable equity curve, annualized return or maximum drawdown. Conditional log contributions sum to the corresponding complete period's log growth.

## Statistical comparison

The primary new contrast is fixed-beta Llama with memory minus the frozen deterministic advisor, using post2021 mean paired daily net log return. Unchanged prior memory contrasts are also reconstructed. Thirty-day joint circular blocks, 5,000 draws, seed16062 give pointwise95% percentile intervals. Secondary component comparisons are exploratory and not multiplicity-adjusted; do not select a favorable one as a new primary. Resampling observed paths does not rerun learning, account for model-seed uncertainty, or make reused historical data fresh OOS. An interval containing zero is inconclusive, not proof of equivalence.

## Definition references

Sharpe's original exposition defines the ratio from mean and standard deviation of differential returns and discusses time dependence and annualization: [William F. Sharpe, The Sharpe Ratio (1994)](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm). The cost-before-return equation above is our pinned experiment convention.

Geometric linking follows the principle of linking subperiod wealth multipliers; terminology and disclosures matter because professional gross/net-of-fees classifications are more specific than our after-trading-costs label: [GIPS Standards Handbook for Firms](https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/). We do not claim compliance.
