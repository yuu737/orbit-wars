# Orbit Wars Experiment Log

Last updated: 2026-06-06

## How to use this file

- Record each meaningful bot change as one experiment entry.
- Keep the focus on intent, evidence, and next action.
- Update this file when a change is tested locally or on Kaggle.

## Template

```text
## YYYY-MM-DD - vX.Y short name

Goal:
- What weakness are we trying to fix?

Change:
- What code or behavior changed?

Local evidence:
- `evaluate.py` result summary

Kaggle evidence:
- Submission score, rank movement, or replay observations

Decision:
- Keep / revert / revise

Next:
- What to try next
```

## 2026-06-05 - Starter baseline

Goal:
- Get a working submission and local evaluation baseline.

Change:
- Used the starter nearest-planet bot.
- Added `evaluate.py` for repeatable multi-seed testing.

Local evidence:
- `random` over 20 games
- Win rate: `70%`
- Average score diff: `+1621.75`
- Crash rate: `0%`

Kaggle evidence:
- First submission completed successfully.
- Validation logs showed no runtime error.

Decision:
- Keep as a baseline only.

Next:
- Replace nearest-only expansion with a value-based policy.

## 2026-06-05 - V1 value expansion

Goal:
- Expand more intelligently than nearest-target-only behavior.

Change:
- Added value scoring using `production`, `capture cost`, and `travel time`.
- Added ship reservation to avoid emptying owned planets.
- Avoided double-committing multiple planets to the same target.

Local evidence:
- `random` over 20 games
- Win rate: `95%`
- Average score diff: `+13069.95`
- Crash rate: `0%`

Kaggle evidence:
- Submission 1 started from this family of logic.

Decision:
- Keep.

Next:
- Improve shot quality with rotation prediction and sun avoidance.

## 2026-06-05 - V2 trajectory quality

Goal:
- Reduce losses from bad firing lines and stale target positions.

Change:
- Predicted arrival-time positions for rotating planets.
- Penalized and then rejected shots that cross the sun.

Local evidence:
- `random` over 20 games
- Win rate: `100%`
- Average score diff: `+18869.60`
- Crash rate: `0%`

Kaggle evidence:
- First real losing replay still showed severe early expansion weakness.

Decision:
- Keep.

Next:
- Make the early game more aggressive.

## 2026-06-05 - V2.1 early-expansion tuning

Goal:
- Fix passivity seen in the first real 1v1 loss.

Change:
- Reduced early ship reserves.
- Boosted neutral expansion value before turn `80`.

Local evidence:
- `random` over 20 games
- Win rate: `100%`
- Average score diff: `+18692.35`
- Crash rate: `0%`

Kaggle evidence:
- Another real 1v1 loss showed the opening was still too slow.
- At turn `25`, we still had only `1` planet.

Decision:
- Partial improvement, but not enough.

Next:
- Add an explicit opening mode instead of only tuning weights.

## 2026-06-06 - V2.2 opening mode

Goal:
- Fix repeated 1v1 losses caused by a stalled opening.

Change:
- Added an opening mode for turns `< 90`.
- Prioritized cheap nearby neutral planets before normal value scoring.
- Lowered opening reserves further to force earlier expansion.

Local evidence:
- `random` over 20 games
- Win rate: `100%`
- Average score diff: `+19040.00`
- Crash rate: `0%`

Kaggle evidence:
- Intended to address two replay findings:
- Loss 1: opponent had `7` planets by turn `50` while we had `4`
- Loss 2: opponent had `3` planets by turn `25` while we had `1`

Decision:
- Keep and submit.

Next:
- Evaluate this version on Kaggle and inspect whether early planet count closes the gap.
- If the opening improves, move next to defense and reinforcement logic.

## 2026-06-06 - V2.3 reinforcement pass

Goal:
- Stop newly captured or high-production planets from staying dangerously thin.

Change:
- Added a desired garrison target that scales by planet production and game phase.
- Added a simple reinforcement pass before attack selection.
- Allowed strong planets to send modest support to weak owned planets when the route is safe.

Local evidence:
- `random` over 20 games
- Win rate: `100%`
- Average score diff: `+21767.65`
- Crash rate: `0%`

Kaggle evidence:
- Not submitted.
- Head-to-head against `v2.2` over 20 games and both seats:
- Win rate: `30%`
- Average score diff: `-4432.25`

Decision:
- Rejected as the main line.

Next:
- Keep the self-play comparison harness.
- Use `v2.2` as the active submission baseline.
- Revisit defense later with enemy-aware logic instead of generic reinforcement first.
