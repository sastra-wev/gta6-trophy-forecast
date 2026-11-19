# GTA VI Trophy Rarity Forecast

A forecasting program for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
for each of GTA VI's 55 PS5 trophies, predict the five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95) of its Exophase earn rate at the target
instant, scored by mean pinball loss on the logit scale.

[![Competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sastra-wev/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sastra-wev/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

## How it works

`main.py` (Python 3, standard library only, deterministic, no network)
reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

The page being forecast is a step function: Exophase refreshes its counts
on its own schedule, so consecutive identical reads are a single
observation. All fitting runs on the **distinct page states**, on the
house curve's clock `u = ln(1 + hours since 2026-11-19T00:00Z)`.

Per trophy, the median blends two extrapolations on the logit scale:

- **Earners/pool decomposition** — earners (`rate × players_tracked`) and
  the tracked pool are fit separately (weighted least squares of the log
  against `u`, anchored at the last read) and recombined, so pool growth
  dilutes rates honestly instead of being read as negative progress.
- **Direct trend** — a weighted least-squares line of `logit(rate)`
  against `u`.

Both slopes are **shrunk toward the trophy's tier slope** (hierarchical
shrinkage: thin data borrows the tier's trajectory) and **damped** by
`1.7 / (1.7 + Δu)`, because a slope fitted on the recent window is the
steep part of an S-curve and reading it straight out to the target
overshoots. Implied rates are capped relative to the last observed rate.
Degenerate series fall back to tier priors: never-observed or
zero-earner trophies get a small tier "emergence" median, and a trophy
pinned near 100% stays there.

The spread is a per-trophy logit sigma combining a horizon term, the
fits' residual volatility with an extrapolation-leverage factor, and a
regime term that widens when the trailing page state has been frozen for
hours (a refresh is due, and the target read will not show the stale
page). Secret trophies get a slightly wider band.

## Validation

On the kit's practice cases (flat launch-night histories) the program
produces valid, byte-identical outputs in ~0.03 s per case, with
dilution-tilted medians and bands wide enough to survive the refresh
that zeroed the house baseline's coverage. On synthetic launch-week
worlds with known ground truth (pool growth ×2–5, logistic progression,
refresh shocks, slow-grind), it beats the house last-rate baseline's
pinball loss in every world tested (~35% lower loss overall).
