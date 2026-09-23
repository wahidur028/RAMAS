# Scientific method: RAMAS Stage 5.3

## Research question

Can a single LLM-agent add statistically supported residual allocation value
over a transparent regime controller when its autonomy is limited to causal
tool selection, a three-action residual grid, frozen risk controls, and
evidence-gated memory?

## Why this test is justified

Stage 5.1 established a valid model interface but not incremental value. The
agent copied the distilled static mapping on 93.24% of development days. Its 74
deviations reduced core-policy growth from 58.48% to 49.70%, reduced Sharpe from
0.807 to 0.721, and had one-sided block-bootstrap p-value 0.8201. All five
promoted lessons were later rolled back.

The repair therefore preserves the useful router and removes the LLM from the
default control path. It asks a narrower falsifiable question: can agentic tool
use identify occasional residual corrections with incremental value?

## Predeclared mechanism

The frozen transparent controller maps Bear to 0.25, Bull to 0.50, and Mix to
0.25. The agent is triggered only by `transition_day` or router confidence
strictly below 0.80. It uses a two-stage loop:

1. The planner selects deterministic tools.
2. The runtime executes the selected tools on information available at the
   decision close.
3. The decision component observes tool results and proposes one residual.
4. Code validates citations and quantitative eligibility.
5. The unchanged risk layer projects the final desired exposure.

This is an LLM-agent, not a one-shot LLM forecast: the model chooses tools,
receives environmental observations, acts within a bounded action space, and
receives matured next-day feedback through incident memory.

## Causal controls

- Features use prices no later than the decision date.
- Outcomes are paired with the next calendar day's return.
- Retrieved incidents must precede the current decision and have completed
  outcomes.
- The compare tool never receives the current decision's next-day return.
- Transaction costs use symmetric turnover from drifted pretrade exposure.
- Both the residual agent and transparent comparator use separate path-dependent
  pretrade exposure and the same frozen risk layer.
- No 2024–2025 file is required or evaluated.

## Tool evidence gate

A nonzero residual is accepted only if:

- all four required tools were selected and executed;
- every required tool-result identifier was cited;
- the compare tool evaluated at least 12 same-regime, same-uncertainty-bucket
  matured incidents;
- mean one-step log advantage is positive;
- candidate daily-loss CVaR is no worse than the transparent action;
- advantage is positive in both non-overlapping chronological halves.

This gate supports action selection but is not a profitability claim. Only the
full path-dependent replay and final bootstrap gate can make that claim.

## Memory gate

Lesson text is quarantined. Promotion requires 30 same-context incidents,
positive mean residual advantage, a positive 95% circular-block-bootstrap lower
bound, non-worse CVaR, and two positive chronological blocks. Rollback is
automatic after 20 additional incidents if cumulative advantage is non-positive
or CVaR becomes worse. Monthly trust updates and model fine-tuning are absent.

## Development kill gate

The primary comparator is the transparent controller under identical risk and
accounting. The branch passes only if every condition holds:

- positive total growth advantage;
- Sharpe improvement at least 0.05;
- one-sided 30-day circular-block-bootstrap p-value below 0.05;
- daily-loss CVaR not worse;
- positive growth advantage in at least two calendar years;
- at least 99% valid plans and decisions on triggered days.

If any check fails, the economic LLM-agent branch stops. The reused OOS window
cannot rescue it. A pass allows freezing the single-agent design for a future
prospective confirmation; it still does not establish deployable profitability.

## Evidence labels

- Controlled provider: mechanical validation only.
- 2021–2023: development evidence only.
- 2024–2025: sealed in this package.
- Prospective data: required for confirmation after a development pass.
