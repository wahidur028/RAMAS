# Defect and Repair Record

## Observed defect

The original real-data Router assigned terminal Buy-and-Hold trust of 37.52% in Bear, 7.93% in Bull and 16.00% in Mix, even though the Bull posterior had the expected positive next-return semantics.

The cause was not an inverted label. The trust updater:

1. selected the worst 5% of all market days;
2. rescaled that global tail signal;
3. conditioned it on each regime afterward;
4. updated every regime row regardless of posterior support.

This allowed conditional tail components larger than the maximum observed daily loss and produced large trust changes from negligible regime mass.

## Frozen correction

- exact weighted conditional upper-tail mean;
- minimum posterior mass 20;
- minimum effective sample size 20;
- unsupported rows unchanged;
- complete update audit emitted;
- fixed matched-exposure economic gate.

No return-driven threshold search is permitted.
