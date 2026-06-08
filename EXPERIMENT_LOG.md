# Orbit Wars Experiment Log

Last updated: 2026-06-07

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

## 2026-06-06 - V2.4 threat-aware defense

Goal:
- Stop midgame collapses by defending planets that appear to be directly threatened by enemy fleets.

Change:
- Estimated incoming pressure from nearby fleets that are roughly aligned with owned planets.
- Prioritized reinforcement only for planets with a predicted deficit against that pressure.

Local evidence:
- `random` over 20 games and both seats
- Win rate: `100%`
- Average score diff: `+20796.00`
- Crash rate: `0%`
- Head-to-head against `v2.2` over 20 games and both seats
- Win rate: `20%`
- Average score diff: `-7883.70`

Kaggle evidence:
- Not submitted.

Decision:
- Rejected as the main line.

Next:
- Keep the snapshot for reference.
- Continue from `v2.2` as the active baseline.
- Try enemy-aware defense again only with tighter constraints or better threat targeting.

## 2026-06-06 - V4.0 Hairate baseline adoption

Goal:
- Reach a clearly winning local benchmark against the strongest in-repo opponents.
- Stop spending iterations on weaker heuristic branches once a dominant baseline was available.

Change:
- Replaced `main.py` with the stronger `hairate` agent baseline.
- Kept the improved `evaluate.py` harness for repeatable 2-player and 4-player checks.

Local evidence:
- `main.py` vs `bots/main_v3_2_berserker_rush.py` over 100 games and both seats
- Win rate: `100%`
- Average score diff: `+3078.63`
- Crash rate: `0%`
- `main.py` vs `bots/main_v2_9_highprod_recapture.py` over 100 games and both seats
- Win rate: `100%`
- Average score diff: `+3415.07`
- Crash rate: `0%`

Kaggle evidence:
- Not submitted yet in this log entry.

Decision:
- Keep as the new strongest baseline.

Next:
- Submit this version to Kaggle.
- If needed, tune only from this stronger baseline rather than from earlier V2/V3 heuristics.

## 2026-06-06 - V3.5 split opening lookahead

Goal:
- Stay fully on a self-owned code path while surpassing the earlier V3.2 and V2.9 baselines.
- Add safer 4-player behavior without giving up 2-player pressure.
- Improve opening neutral selection with a shallow future-value bonus.

Change:
- Added a lightweight projected reserve requirement from near-term incoming fleet events.
- Split major thresholds between 2-player and 4-player modes.
- Added a small opening-only future-value bonus for neutral targets.
- Saved this snapshot as [main_v3_5_split_opening_lookahead.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v3_5_split_opening_lookahead.py:1).

Local evidence:
- `main.py` vs `bots/main_v3_2_berserker_rush.py` over 40 games and both seats
- Win rate: `70.0%`
- Average score diff: `+8815.70`
- Crash rate: `0%`
- `main.py` vs `bots/main_v2_9_highprod_recapture.py` over 40 games and both seats
- Win rate: `65.0%`
- Average score diff: `+3112.88`
- Crash rate: `0%`
- `main.py` in 4-player games vs `random` over 40 games and all seats
- Win rate: `92.5%`
- Average placement: `1.15`

Kaggle evidence:
- Not submitted yet in this log entry.

Decision:
- Keep as the current strongest self-owned baseline.

Next:
- Evaluate against `hairate` again and continue only with changes that preserve the gains over `V3.2` and `V2.9`.

## 2026-06-06 - V3.6 4P targeting

Goal:
- Keep the self-owned `V3.5` strengths in 2-player matches.
- Improve mixed 4-player results against stronger non-random opponents.

Change:
- Added lightweight player-power estimation from planets and fleets.
- In 4-player enemy attacks, prefer cheaper hits on weaker opponents.
- Penalize expensive attacks into the current leader unless the target is unusually cheap or valuable.
- Saved this snapshot as [main_v3_6_4p_targeting.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v3_6_4p_targeting.py:1).

Local evidence:
- `main.py` vs `bots/main_v3_2_berserker_rush.py` over 40 games and both seats
- Win rate: `70.0%`
- Average score diff: `+8815.70`
- Crash rate: `0%`
- `main.py` vs `bots/main_v2_9_highprod_recapture.py` over 40 games and both seats
- Win rate: `65.0%`
- Average score diff: `+3112.88`
- Crash rate: `0%`
- `main.py` in 4-player games vs `bots/main_v2_9_highprod_recapture.py`, `bots/v2_7_highprod_pressure.py`, and `random` over 40 games and all seats
- Win rate: `35.0%`
- Average placement: `2.05`
- Average score diff: `-5499.68`
- Previous self-owned baseline on the same mixed 4-player benchmark: `32.5%` win rate, `2.15` average placement, `-6344.25` average score diff

Kaggle evidence:
- Not submitted yet in this log entry.

Decision:
- Keep as the new strongest self-owned baseline.

Next:
- Continue improving 4-player mixed-field stability without giving back the 2-player edge over `V3.2` and `V2.9`.

## 2026-06-06 - V3.6a adaptive 4P targeting

Goal:
- Beat or at least not fall behind `V3.5` in direct 2-player comparison.
- Stop losing badly in 4-player mirrors against `V3.5 x3`.

Change:
- Kept the `V3.6` 4-player targeting idea.
- Gated the asymmetric 4-player enemy-selection bonuses so they only activate once the table has a meaningful power spread.
- In near-mirror 4-player lobbies, the bot now stays much closer to `V3.5` behavior.
- Saved this snapshot as [main_v3_6a_4p_adaptive_targeting.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v3_6a_4p_adaptive_targeting.py:1).

Local evidence:
- `main.py` vs `bots/main_v3_5_split_opening_lookahead.py` over 40 games and both seats
- Win rate: `65.0%`
- Average score diff: `0.00`
- Crash rate: `0%`
- `main.py` in 4-player games vs `bots/main_v3_5_split_opening_lookahead.py` over 40 games and all seats
- Win rate: `100.0%`
- Average placement: `1.00`
- Average score diff: `0.00`

Kaggle evidence:
- Not submitted yet in this log entry.

Decision:
- Keep as the active baseline for the next round of self-owned improvements.

Next:
- From this safer baseline, improve 4-player mixed fields again without losing the restored `V3.5 x3` mirror result.

## 2026-06-06 - V3.7 influence area control candidate

Goal:
- Move away from mirror-preserving tweaks and test a more general algorithmic improvement.
- Make target selection care more about whether the captured area is locally supportable.

Change:
- Added a lightweight local influence map over nearby planets and fleets.
- Added a midgame `influence_adjustment` so neutral and enemy targets are scored by local area control, not only raw value and travel time.
- Saved this branch as [main_v3_7_influence_area_control.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v3_7_influence_area_control.py:1).

Local evidence:
- `main.py` vs `bots/main_v3_2_berserker_rush.py` over 40 games and both seats
- Win rate: `65.0%`
- Average score diff: `+6690.05`
- Crash rate: `0%`
- `main.py` vs `bots/main_v2_9_highprod_recapture.py` over 40 games and both seats
- Win rate: `65.0%`
- Average score diff: `+3005.43`
- Crash rate: `0%`
- `main.py` in 4-player games vs `bots/main_v2_9_highprod_recapture.py`, `bots/v2_7_highprod_pressure.py`, and `random` over 40 games and all seats
- Win rate: `35.0%`
- Average placement: `2.15`
- Average score diff: `-4822.18`
- `main.py` in 4-player games vs `bots/main_v3_5_split_opening_lookahead.py` over 40 games and all seats
- Win rate: `60.0%`
- Average placement: `1.73`
- Average score diff: `+334.27`

Decision:
- Promising as a general algorithmic branch, but not yet a clear overall replacement for `V3.6a`.
- Kept as a saved candidate and restored `main.py` to the safer `V3.6a` baseline.

Next:
- Keep pursuing general improvements, but judge them on mixed-field average placement first, not only head-to-head wins.

## 2026-06-06 - V3.8 endgame ROI candidate

Goal:
- Improve late-game score retention with a more explicit ROI check.
- Reduce hopeless late neutral grabs and deep attacks that cannot pay back before turn 500.

Change:
- Added `endgame_roi_adjustment` based on remaining turns, arrival time, setup delay, and approximate payback.
- Applied stronger late-game penalties to low-ROI neutral captures and deep enemy attacks.
- Saved this branch as [main_v3_8_endgame_roi.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v3_8_endgame_roi.py:1).

Local evidence:
- `main.py` vs `bots/main_v3_2_berserker_rush.py` over 40 games and both seats
- Win rate: `70.0%`
- Average score diff: `+8815.90`
- Crash rate: `0%`
- `main.py` vs `bots/main_v2_9_highprod_recapture.py` over 40 games and both seats
- Win rate: `65.0%`
- Average score diff: `+3112.88`
- Crash rate: `0%`
- `main.py` in 4-player games vs `bots/main_v2_9_highprod_recapture.py`, `bots/v2_7_highprod_pressure.py`, and `random` over 40 games and all seats
- Win rate: `32.5%`
- Average placement: `2.30`
- Average score diff: `-4742.60`
- `main.py` in 4-player games vs `bots/main_v3_5_split_opening_lookahead.py` over 40 games and all seats
- Win rate: `80.0%`
- Average placement: `1.30`
- Average score diff: `-17.00`

Decision:
- Rejected as the active baseline.
- The ROI heuristic likely cut too many playable late-game actions, especially in 4-player fields.
- Restored `main.py` to [main_v3_6a_4p_adaptive_targeting.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v3_6a_4p_adaptive_targeting.py:1).

Next:
- Try a narrower endgame rule next time, such as only filtering late low-production neutrals, instead of applying a broad ROI penalty to both neutral and enemy targets.

## 2026-06-06 - V3.9 effective capture and anti-snipe hold

Goal:
- Move the self-owned bot toward `hairate`-style state awareness without a full rewrite.
- Stop underpricing targets whose garrison will grow or receive incoming fleets before our arrival.
- Avoid neutral captures that are immediately vulnerable to enemy re-snipes.

Change:
- Added `projected_capture_garrison` and `effective_capture_plan`.
- Opening and normal target selection now use projected arrival-time capture cost.
- Added a lightweight `capture_holds_against_snipe` veto for neutral targets.

Local evidence:
- `main.py` vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and both seats
- Win rate: `75.0%`
- Average score diff: `+7747.75`
- Crash rate: `0%`
- `main.py` vs `bots/main_v3_2_berserker_rush.py` over 8 games and both seats
- Win rate: `100.0%`
- Average score diff: `+14087.62`
- Crash rate: `0%`
- `main.py` in 4-player games vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and all seats
- Win rate: `100.0%`
- Average placement: `1.00`
- Average score diff: `+708.50`
- `main.py` vs `bots/hairate.py` over 16 games and both seats
- Win rate: `0.0%`
- Average survival turn improved from about `103` to `153`, but average score diff worsened to `-4745.44`
- `main.py` in 4-player games vs `bots/hairate.py` over 8 games and all seats
- Win rate: `0.0%`
- Average score diff improved from about `-1786.50` to `-1334.88`

Decision:
- Keep as a constructive step, not a solved version.
- The bot survives longer and remains strong against self-owned baselines, but still needs search/defense depth to threaten `hairate`.

Next:
- Add a small candidate-action search or stronger emergency defense pass before doing more scalar threshold tuning.

## 2026-06-06 - V3.10 narrow emergency defense

Goal:
- Improve V3.9 by saving valuable planets that are projected to fall soon.
- Avoid repeating the older V2.3/V2.4 failure mode where generic defense slowed expansion too much.

Change:
- Added `projected_defense_need` to detect near-term planet losses from incoming fleets.
- Added `build_emergency_defense_needs` with strict urgency/value gating.
- Added `find_emergency_defense_target` before ordinary reinforcement and attacks.
- Updated `main.py` title to `V3.10 Effective Defense Agent`.

Local evidence:
- `main.py` vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and both seats
- Win rate: `75.0%`
- Average score diff: `+6352.38`
- Crash rate: `0%`
- `main.py` vs `bots/main_v3_2_berserker_rush.py` over 8 games and both seats
- Win rate: `100.0%`
- Average score diff: `+16767.50`
- Crash rate: `0%`
- `main.py` vs `bots/hairate.py` over 8 games and both seats
- Win rate: `0.0%`
- Average survival turn: `158.50`
- Average score diff: `-4230.25`
- `main.py` in 4-player games vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and all seats
- Win rate: `100.0%`
- Average placement: `1.00`
- Average score diff: `+700.00`
- `main.py` in 4-player games vs `bots/hairate.py` over 8 games and all seats
- Win rate: `0.0%`
- Average survival turn: `140.38`
- Average score diff: `-1194.38`

Decision:
- Keep. It preserves the self-owned baseline wins and improves survival/score gap against `hairate`.

Next:
- Move from single-action heuristics toward small candidate-action search, especially for coordinated attacks and counter-snipes.

## 2026-06-06 - V3.11 conservative coalition pressure

Goal:
- Address a structural gap where the bot only captured targets that one source planet could afford.
- Add limited multi-source attacks without recreating the old over-defense/overcommitment failures.

Change:
- Added `try_coalition_capture` for high-production targets that require 2-3 contributors.
- Added `coalition_target_base_value` and strict gates for production, timing, travel time, sun safety, and cost.
- Tightened the first version after a regression against `V3.2`, especially for early enemy-planet coalitions.
- Updated `main.py` title to `V3.11 Coalition Pressure Agent`.

Local evidence:
- After tightening, `main.py` vs `bots/main_v3_2_berserker_rush.py` over 8 games and both seats
- Win rate: `100.0%`
- Average score diff: `+16767.25`
- Crash rate: `0%`
- `main.py` vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and both seats
- Win rate: `75.0%`
- Average score diff: `+6474.38`
- Crash rate: `0%`
- `main.py` vs `bots/hairate.py` over 8 games and both seats
- Win rate: `0.0%`
- Average survival turn: `158.50`
- Average score diff: `-4230.25`
- `main.py` in 4-player games vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and all seats
- Win rate: `100.0%`
- Average placement: `1.00`
- Average score diff: `+700.00`

Decision:
- Keep as a safe incremental improvement, but not a `hairate` breakthrough.
- The first looser coalition pass was more explosive but caused a `V3.2` regression, so the stricter version is the active line.

Next:
- Build counter-snipe or shallow candidate-action search. That is more likely to move the `hairate` matchup than further coalition threshold tuning.

## 2026-06-06 - V3.12 reactive counter-snipe

Goal:
- Add a `hairate`-style tactical response where enemy fleets that soften/capture neutrals can be punished.
- Improve 2-player pressure without hurting the safer 4-player baseline.

Change:
- Added `predicted_enemy_capture_surplus` for enemy fleets headed to neutral planets.
- Added `find_counter_snipe` to launch after an enemy capture if our delayed recapture is cheap, timely, and likely to hold.
- Counter-snipes run after emergency defense and coalition checks, before ordinary expansion.
- Updated `main.py` title to `V3.12 Counter-Snipe Agent`.

Local evidence:
- `main.py` vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and both seats
- Win rate: `100.0%`
- Average score diff: `+16860.62`
- Crash rate: `0%`
- `main.py` vs `bots/main_v3_2_berserker_rush.py` over 8 games and both seats
- Win rate: `100.0%`
- Average score diff: `+16809.25`
- Crash rate: `0%`
- `main.py` vs `bots/hairate.py` over 8 games and both seats
- Win rate: `0.0%`
- Average survival turn: `156.38`
- Average score diff: `-4011.12`
- `main.py` in 4-player games vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and all seats
- Win rate: `100.0%`
- Average placement: `1.00`
- Average score diff: `+700.00`
- `main.py` in 4-player games vs `bots/hairate.py` over 8 games and all seats
- Win rate: `0.0%`
- Average survival turn: `140.38`
- Average score diff: `-1194.38`

Decision:
- Keep. This is the clearest recent improvement against self-owned baselines and modestly improves the 2-player `hairate` gap.

Next:
- Add shallow action search/forward scoring. Counter-snipe helps tactics, but `hairate` still wins through broader search and coordinated timing.

## 2026-06-06 - V3.13 frontline reserve, forward-score rejected

Goal:
- Continue from V3.12 and test whether light forward-style scoring improves target choice.
- If forward scoring fails, keep only tactical/safety changes that preserve the `hairate` gap.

Change:
- Tried a lightweight `forward_action_adjustment` that estimated post-capture production payoff and local pressure.
- Rejected and removed that forward scoring because it improved self-owned matchups but worsened the 2-player `hairate` gap.
- Added `nearby_enemy_reserve_bonus` for valuable frontline planets near strong enemy sources.
- Updated `reserve_for_planet` to include this narrow frontline reserve.
- Updated `main.py` title to `V3.13 Frontline Reserve Agent`.

Local evidence:
- Rejected forward scoring trial:
- `main.py` vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and both seats improved to `100.0%`, but `main.py` vs `bots/hairate.py` worsened to around `-4292` to `-4361` average score diff.
- Active V3.13:
- `main.py` vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and both seats
- Win rate: `100.0%`
- Average score diff: `+16935.62`
- Crash rate: `0%`
- `main.py` vs `bots/main_v3_2_berserker_rush.py` over 8 games and both seats
- Win rate: `100.0%`
- Average score diff: `+16776.75`
- Crash rate: `0%`
- `main.py` vs `bots/hairate.py` over 8 games and both seats
- Win rate: `0.0%`
- Average survival turn: `156.38`
- Average score diff: `-4011.12`
- `main.py` in 4-player games vs `bots/main_v3_6a_4p_adaptive_targeting.py` over 8 games and all seats
- Win rate: `100.0%`
- Average placement: `1.00`
- Average score diff: `+700.00`
- `main.py` in 4-player games vs `bots/hairate.py` over 8 games and all seats
- Win rate: `0.0%`
- Average survival turn: `140.38`
- Average score diff: `-1194.38`

Decision:
- Keep the frontline reserve.
- Reject the naive forward-score adjustment. Future search should be action-simulation based, not another broad scalar bonus.

Next:
- Build a truly shallow candidate-action search with explicit no-action baseline and conflict handling, or focus on diagnosing `hairate` loss replays by seed.

## 2026-06-06 - V4.0 planner branch foundation

Goal:
- Start a separate self-owned planner branch instead of continuing scalar V3 heuristic tuning.
- Preserve `main.py` as the stable baseline and implement projection, safe drain, capture floor, candidate greedy, and regroup in a new file.

Change:
- Added [main_v4_0_planner.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_0_planner.py:1).
- Added `PlannerConfig` with 2P/4P presets.
- Added `project_planet_states`, `safe_drain`, `capture_floor`, attack/defense candidate generation, greedy selection, and short-distance regroup.
- Kept implementation independent from `hairate2/orbit_lite`; those remain reference/evaluation code only.

Local evidence:
- `bots/main_v4_0_planner.py` vs `main.py` over 8 games and both seats
- Win rate: `100.0%`
- Average score diff: `+7869.75`
- Crash rate: `0%`
- `bots/main_v4_0_planner.py` vs `bots/hairate.py` over 8 games and both seats
- Win rate: `0.0%`
- Average survival turn: `135.62`
- Average score diff: `-1990.62`
- `bots/main_v4_0_planner.py` vs `bots/hairate2.py` over 8 games and both seats
- Win rate: `0.0%`
- Average survival turn: `146.88`
- Average score diff: `-5349.75`
- `bots/main_v4_0_planner.py` in 4-player games vs `main.py` over 8 games and all seats
- Win rate: `100.0%`
- Average placement: `1.00`
- Average score diff: `+13167.50`
- `bots/main_v4_0_planner.py` in 4-player games vs `bots/hairate.py` over 8 games and all seats
- Win rate: `25.0%`
- Average placement: `1.75`
- Average score diff: `+209.62`
- `bots/main_v4_0_planner.py` in 4-player games vs `bots/hairate2.py` over 8 games and all seats
- Win rate: `0.0%`
- Average survival turn: `83.50`
- Average score diff: `-683.50`

Decision:
- Keep as the active experimental branch.
- V4.0 already beats the current stable `main.py` locally and improves the 4P `hairate2` gap substantially.
- 2P still needs counter-snipe/frontline reserve transfer and better opening pressure.

Next:
- Implement V4.1 by candidate-izing V3 counter-snipe and improving 2P opening pressure without hurting 4P survival.

## 2026-06-06 - V4.1/V4.2/V4.3 follow-up probes

Goal:
- Continue the V4 planner branch while keeping token/runtime cost low.
- Move V3 tactics into planner candidates and classify `hairate2` losses before broad tuning.

Change:
- Added [main_v4_1_counter_snipe.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_1_counter_snipe.py:1), which turns reactive counter-snipe into normal greedy candidates.
- Added [main_v4_2_4p_shaping.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_2_4p_shaping.py:1), a rejected 4P target-shaping probe.
- Added [main_v4_3_guarded.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_3_guarded.py:1), a rejected higher-reserve probe.
- Added [analyze_losses.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/analyze_losses.py:1), a lightweight loss classifier for `opening_loss`, `early_defense_collapse`, `defense_collapse`, `bad_overexpand_or_low_economy`, and late losses.

Local evidence:
- `main_v4_1_counter_snipe.py` vs `main_v4_0_planner.py` over 6 games and both seats
- Win rate: `83.3%`
- Average placement: `1.17`
- Crash rate: `0%`
- `main_v4_1_counter_snipe.py` vs `bots/hairate2.py` over seeds 0-2 and both seats
- Same result as V4.0 on those seeds: average score diff `-1425.67`, average survival `96.00`
- `main_v4_2_4p_shaping.py` in 4-player games vs `bots/hairate2.py`
- Rejected: worsened average score diff to `-1206.50` versus V4.0's `-683.50` on the comparable small 4P check.
- `main_v4_3_guarded.py` vs `bots/hairate2.py` over seeds 0-2 and both seats
- Rejected: score diff improved to `-860.00`, but survival fell to `76.67` and it only scored `33.3%` vs V4.0.
- `analyze_losses.py` on V4.1 vs `hairate2` seeds 0-2 classified losses as:
- `bad_overexpand_or_low_economy=2`
- `defense_collapse=2`
- `early_defense_collapse=2`

Decision:
- Keep V4.1 as the active experimental branch.
- Reject V4.2 and V4.3 as active branches.
- Use `analyze_losses.py` to guide the next improvement instead of broad scalar tuning.

Next:
- Build a targeted V4.4 defense candidate improvement: rescue threatened high-production planets earlier, instead of globally increasing reserve.

## 2026-06-06 - V4.4 targeted defense probes

Goal:
- Use the `analyze_losses.py` finding that V4 losses vs `hairate2` are often early/normal defense collapses.
- Improve defense without globally raising reserves, because the previous guarded reserve probe hurt the V4 mirror.

Change:
- Added [main_v4_4_targeted_defense.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_4_targeted_defense.py:1), which sizes defense candidates from projected loss severity.
- Added [main_v4_4b_selective_defense.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_4b_selective_defense.py:1), which gates the stronger defense to high-production or urgent planets.

Local evidence:
- `main_v4_4_targeted_defense.py` vs `main_v4_1_counter_snipe.py` over 6 games and both seats
- Win rate: `33.3%`
- Average score diff: `+179.33`
- `main_v4_4_targeted_defense.py` vs `bots/hairate2.py` over seeds 0-2 and both seats
- Average score diff improved from V4.1's `-1425.67` to `-1243.67`, but survival fell to `93.33`
- `main_v4_4b_selective_defense.py` vs `main_v4_1_counter_snipe.py` over 6 games and both seats
- Win rate: `66.7%`
- Average score diff: `-986.67`
- `main_v4_4b_selective_defense.py` vs `bots/hairate2.py` over seeds 0-2 and both seats
- Average score diff improved slightly to `-1235.67`, but survival fell to `88.00`

Decision:
- Keep V4.1 as the active general branch.
- Keep V4.4b as a saved specialist candidate, not active, because it improves the `hairate2` score gap but hurts the V4.1 mirror and survival.

Next:
- Do not keep increasing defense weights. Next efficient step is parameter search around V4.1/V4.4b knobs, especially `reserve_margin`, `roi_threshold`, and `horizon`.

## 2026-06-06 - V4.5 parameter sweep foundation

Goal:
- Start the automatic parameter-search part of the V4 plan without cloning many full bot files.
- Allow temporary local weakness if it reveals useful long-term search directions.

Change:
- Added [sweep_v4_params.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/sweep_v4_params.py:1), which imports a V4 planner module, swaps `PlannerConfig` values in-process, and runs local Orbit Wars games.
- Added [main_v4_5_roi22.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_5_roi22.py:1), a wrapper branch that keeps V4.1 code but raises `roi_threshold` to `2.2`.
- Added [main_v4_5_growth_params.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_5_growth_params.py:1), a wrapper branch that keeps V4.1 code but lowers `roi_threshold` to `1.2` and raises `max_targets` to `14`.

Local evidence:
- `sweep_v4_params.py --preset small` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- `base`: average score diff `-1053.00`, survival `89.00`
- `guarded`: average score diff `-715.00`, survival `81.00`
- `growth`: average score diff `-640.50`, survival `73.00`
- `sweep_v4_params.py --preset roi` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- `roi2.2`: average score diff `-785.50`, survival `81.50`
- Direct wrapper check vs `main.py` over seed 0 and both seats:
- `main_v4_5_roi22.py`: `100.0%` win rate, average diff `+2353.00`
- `main_v4_5_growth_params.py`: `50.0%` win rate, average diff `-2344.00`
- Direct wrapper check vs `bots/hairate2.py` over seeds 0-1 and both seats:
- `main_v4_5_roi22.py`: average score diff `-785.50`, survival `81.50`
- `main_v4_5_growth_params.py`: average score diff `-640.50`, survival `73.00`
- Loss classification vs `bots/hairate2.py`:
- `main_v4_5_roi22.py`: `bad_overexpand_or_low_economy=2`, `early_defense_collapse=2`
- `main_v4_5_growth_params.py`: `bad_overexpand_or_low_economy=1`, `early_defense_collapse=2`, `opening_loss=1`

Decision:
- Keep V4.1 as the active general branch.
- Keep `main_v4_5_roi22.py` as the safer parameter candidate because it remains stable vs `main.py`.
- Keep `main_v4_5_growth_params.py` as an aggressive research branch only. It improves score gap vs `hairate2` on the tiny sample but introduces an opening-loss failure mode and loses a seat vs `main.py`.

Next:
- Implement an opening/floor guard inside the planner rather than only changing scalar parameters.
- Target the repeated `early_defense_collapse` label with a projected-loss defense gate, while preserving the `growth` branch's ability to reduce score gap.

## 2026-06-06 - V4.6 opening/defense guard probe

Goal:
- Convert the V4.5 loss labels into an actual planner-side rule.
- Reduce `opening_loss` and `early_defense_collapse` without rewriting the planner.

Change:
- Added [main_v4_6_opening_defense_guard.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_6_opening_defense_guard.py:1), a V4.1 wrapper that patches:
- `score_candidate` for opening high-production preference and low-production expansion penalties.
- `build_defense_candidates` for stronger projected-loss rescue sizing on valuable early planets.

Local evidence:
- `main_v4_6_opening_defense_guard.py` vs `main.py` over seeds 0-1 and both seats:
- Win rate: `100.0%`
- Average score diff: `+6046.50`
- `main_v4_6_opening_defense_guard.py` vs `main_v4_1_counter_snipe.py` over seeds 0-1 and both seats:
- Win rate: `50.0%`
- Average score diff: `+401.00`
- `main_v4_6_opening_defense_guard.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-1102.75`
- Average survival: `91.50`
- Loss classification vs `bots/hairate2.py`:
- `bad_overexpand_or_low_economy=4`

Decision:
- Keep V4.6 as a useful diagnostic branch, not the active best branch.
- It successfully removes the `early_defense_collapse` label on this tiny sample, but worsens score gap vs the V4.5 parameter candidates.
- The remaining failure mode is now concentrated: V4 needs stronger opening economy / target generation, not more defense weight.

Next:
- Add an opening economy candidate path: during the first ~80 turns, prefer coordinated captures of production `>=4` neutrals even if they are slightly outside the normal ROI comfort zone.
- Keep the V4.5 `roi2.2` branch as the safer parameter baseline and use V4.6 only as evidence for defense gating.

## 2026-06-06 - V4.7 economy push probe

Goal:
- Test whether combining V4.6's defense guard with wider early target search can reduce the remaining `bad_overexpand_or_low_economy` failures.

Change:
- Added [main_v4_7_economy_push.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_7_economy_push.py:1), which reuses the V4.6 guard patches but widens 2P to `horizon=20`, `max_targets=16`, `max_actions=7`, `roi_threshold=1.15`.

Local evidence:
- `main_v4_7_economy_push.py` vs `main.py` over seeds 0-1 and both seats:
- Win rate: `100.0%`
- Average score diff: `+2082.50`
- `main_v4_7_economy_push.py` vs `main_v4_1_counter_snipe.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-8586.25`
- `main_v4_7_economy_push.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-983.50`
- Average survival: `93.50`
- Loss classification vs `bots/hairate2.py`:
- `bad_overexpand_or_low_economy=4`

Decision:
- Reject V4.7 as an active branch.
- It shows that simply widening the candidate set does not solve economy failures; V4 needs better opening source/target coordination, not just lower ROI and more actions.
- The safest current V4 parameter branch remains `main_v4_5_roi22.py`; the best tiny-sample hairate2 score-gap branch remains `main_v4_5_growth_params.py`, but it is unstable.

Next:
- Implement true multi-source opening capture or a target-level "commitment" planner so high-production neutrals can be taken reliably when no single source can safely pay the full capture floor.

## 2026-06-06 - V4 submit candidate activation

Goal:
- Turn the current safest V4 branch into a Kaggle-submittable single-file `main.py`.

Change:
- Backed up the previous `main.py` to [main_pre_v4_submission_backup.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_pre_v4_submission_backup.py:1).
- Added [main_v4_submit_candidate.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_submit_candidate.py:1), a flat single-file V4.1 planner with the V4.5 `roi_threshold=2.2` settings baked in.
- Copied that flat V4 submit candidate to [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1).

Local evidence:
- `py_compile` passed for `main.py` and `bots/main_v4_submit_candidate.py`.
- `main.py` vs `bots/main_pre_v4_submission_backup.py` over seeds 0-1 and both seats:
- Win rate: `100.0%`
- Average score diff: `+3020.50`
- `main.py` vs `bots/hairate.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-1679.50`
- `main.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-785.50`
- Average survival: `81.50`

Decision:
- `main.py` is now a valid V4 experiment submission file.
- This is not yet the final strongest long-term V4; it is the safest currently submittable V4 checkpoint.

Next:
- If leaderboard feedback is poor, restore [main_pre_v4_submission_backup.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_pre_v4_submission_backup.py:1) or continue V4 with multi-source opening capture before the next submission.

## 2026-06-06 - V4.8 multi-source opening probe

Goal:
- Test a true multi-source opening capture path so high-production neutrals can be taken when no single source can safely pay the full capture floor.

Change:
- Added [main_v4_8_multisource_opening.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_8_multisource_opening.py:1).
- Extended `Candidate` with `group_id`.
- Added `build_opening_combo_candidates`, which creates grouped `opening_combo` launches from multiple sources to one high-production neutral.
- Extended `greedy_select` so grouped candidates are selected atomically.

Local evidence:
- `py_compile` passed.
- `main_v4_8_multisource_opening.py` vs current V4 submit [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-3739.00`
- Average survival: `499.00`
- `main_v4_8_multisource_opening.py` vs `bots/main_v4_1_counter_snipe.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-4569.50`
- `main_v4_8_multisource_opening.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-1347.50`
- Average survival: `99.00`
- Loss classification vs `bots/hairate2.py`:
- `bad_overexpand_or_low_economy=4`

Decision:
- Reject V4.8 as an active branch.
- Multi-source opening capture changes the game shape and can extend survival, but it currently worsens score gap because the newly captured economy is not defended/regrouped well enough.
- Keep the current V4 submit [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) unchanged.

Next:
- The next aligned improvement is not more opening capture. It is post-capture consolidation: immediately reserve/regroup toward freshly captured high-production neutrals and avoid launching follow-up attacks from them until they stabilize.

## 2026-06-06 - V4.9 capture stabilization probe

Goal:
- Test post-capture consolidation without carrying over V4.8's worsening multi-source opening behavior.
- Add support moves toward high-production planets that were selected as attack/counter-snipe targets in the current turn.

Change:
- Added [main_v4_9_stabilize_capture.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_9_stabilize_capture.py:1).
- Added `build_capture_stabilize_candidates`, which creates short-range `regroup` support candidates for newly targeted high-production captures.

Local evidence:
- `py_compile` passed.
- `main_v4_9_stabilize_capture.py` vs current V4 submit [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over seeds 0-1 and both seats:
- Win rate: `100.0%`
- Average score diff: `0.00`
- Average survival: `499.00`
- `main_v4_9_stabilize_capture.py` vs `bots/main_v4_1_counter_snipe.py` over seeds 0-1 and both seats:
- Win rate: `50.0%`
- Average score diff: `-3763.00`
- `main_v4_9_stabilize_capture.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-785.50`
- Average survival: `81.50`
- Loss classification vs `bots/hairate2.py`:
- `bad_overexpand_or_low_economy=2`
- `early_defense_collapse=2`

Decision:
- Do not promote V4.9. It is effectively neutral vs the current submit candidate on the hairate2 micro-benchmark.
- Capture support candidates are too narrow to affect the current failure pattern.

Next:
- Revisit V4.6's projected-loss defense gate and merge only the useful early-defense portion into the safer V4.5 submit line, without the opening scoring penalties that worsened score gap.

## 2026-06-06 - V4.10 early-defense-only probe

Goal:
- Extract only the useful early-defense idea from V4.6, without V4.6's opening score penalties.
- Reduce the repeated `early_defense_collapse` label in the current V4 submit line.

Change:
- Added [main_v4_10_early_defense_only.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_10_early_defense_only.py:1).
- Added `projected_defense_need`.
- Replaced defense candidate sizing/scoring with a high-production early rescue gate.

Local evidence:
- `py_compile` passed.
- `main_v4_10_early_defense_only.py` vs current V4 submit [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over seeds 0-1 and both seats:
- Win rate: `50.0%`
- Average score diff: `-8.50`
- Average survival: `499.00`
- `main_v4_10_early_defense_only.py` vs `bots/main_v4_1_counter_snipe.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-6329.00`
- `main_v4_10_early_defense_only.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-785.50`
- Average survival: `81.50`
- Loss classification vs `bots/hairate2.py`:
- `bad_overexpand_or_low_economy=2`
- `early_defense_collapse=2`

Decision:
- Reject V4.10 as an active branch.
- The defense candidate tweak does not affect the hairate2 failure pattern, suggesting the issue is upstream: `project_planet_states` / incoming detection is not flagging the early collapse soon enough, or the defense candidate is not competitive in greedy selection.

Next:
- Improve the projection/arrival model used by defense: inspect fleet-to-planet ETA/target assignment and add a direct enemy-threat defense candidate based on current enemy planets/fleets, not only `first_loss_turn_by_id`.

## 2026-06-06 - V4.11 direct enemy-threat defense probe

Goal:
- Add defense candidates that do not depend on `projection.first_loss_turn_by_id`.
- Detect near-future danger directly from enemy planets and enemy fleets.

Change:
- Added [main_v4_11_direct_threat_defense.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_11_direct_threat_defense.py:1).
- Added `direct_enemy_threat`.
- Added `build_direct_threat_defense_candidates` and mixed it into the normal greedy candidate pool.

Local evidence:
- `py_compile` passed.
- `main_v4_11_direct_threat_defense.py` vs current V4 submit [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-7682.50`
- `main_v4_11_direct_threat_defense.py` vs `bots/main_v4_1_counter_snipe.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-2149.50`
- `main_v4_11_direct_threat_defense.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-817.00`
- Average survival: `81.00`
- Loss classification vs `bots/hairate2.py`:
- `bad_overexpand_or_low_economy=2`
- `opening_loss=2`

Decision:
- Reject V4.11 as an active branch.
- Direct threat defense is too sensitive and turns early-defense collapse into opening/economy loss.
- This confirms that V4's next improvement needs better opening economy and selective defense, not broad defensive pressure.

Next:
- Use sweep results to tune a narrower hybrid: keep `roi_threshold=2.2`, but only allow direct defense for production `>=4` planets after we have at least 3 owned planets or production `>=10`.

## 2026-06-06 - V4.12 selective hybrid defense probe

Goal:
- Narrow V4.11's over-sensitive direct defense so it only protects high-production planets after the opening economy is established.

Change:
- Added [main_v4_12_selective_hybrid_defense.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_12_selective_hybrid_defense.py:1).
- Added `my_economy_ready`, `selective_enemy_threat`, and `build_selective_hybrid_defense_candidates`.
- Gated direct defense to high-production planets, close threats, and post-minimum-economy states.

Local evidence:
- `py_compile` passed.
- `main_v4_12_selective_hybrid_defense.py` vs current V4 submit [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over seeds 0-1 and both seats:
- Win rate: `100.0%`
- Average score diff: `0.00`
- Average survival: `499.00`
- `main_v4_12_selective_hybrid_defense.py` vs `bots/main_v4_1_counter_snipe.py` over seeds 0-1 and both seats:
- Win rate: `50.0%`
- Average score diff: `-4696.50`
- `main_v4_12_selective_hybrid_defense.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-785.50`
- Average survival: `81.50`
- Loss classification vs `bots/hairate2.py`:
- `bad_overexpand_or_low_economy=2`
- `early_defense_collapse=2`

Decision:
- Do not promote V4.12. It is safe, but it is effectively neutral on the hairate2 benchmark.
- Direct defense has a narrow useful window: broad versions hurt economy, narrow versions rarely fire.

Next:
- Stop spending iterations on defensive candidates alone. Return to opening economy scoring/candidate generation, especially high-production neutral selection and avoiding production-1/2 distractions before turn 80.

## 2026-06-06 - V4.13 opening economy focus probe

Goal:
- Reduce `bad_overexpand_or_low_economy` by pushing early target selection toward production 4/5 neutrals and away from production 1/2 distractions.

Change:
- Added [main_v4_13_opening_economy_focus.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_13_opening_economy_focus.py:1).
- Added opening-aware `target_shortlist` scoring.
- Added opening economy bonuses/penalties in `score_candidate`.

Local evidence:
- `py_compile` passed.
- `main_v4_13_opening_economy_focus.py` vs current V4 submit [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-1691.00`
- `main_v4_13_opening_economy_focus.py` vs `bots/main_v4_1_counter_snipe.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-6858.00`
- `main_v4_13_opening_economy_focus.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-734.00`
- Average survival: `81.50`
- Loss classification vs `bots/hairate2.py`:
- `bad_overexpand_or_low_economy=2`
- `early_defense_collapse=2`

Decision:
- Do not promote V4.13.
- It slightly improves the tiny hairate2 score gap (`-785.50` to `-734.00`) but damages self-play/V4 comparisons badly.
- The direction has signal, but the scoring weights are too strong.

Next:
- Create a milder V4.14 opening economy branch: keep the production 4/5 boost, but reduce low-production penalties and avoid changing target shortlist too aggressively.

## 2026-06-06 - V4.14 mild opening economy probe

Goal:
- Keep the useful part of V4.13's opening economy signal while avoiding the self-play collapse caused by overly strong target-shortlist and low-production penalties.

Change:
- Added [main_v4_14_mild_opening_economy.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_14_mild_opening_economy.py:1).
- Kept `target_shortlist` unchanged.
- Added only mild opening bonuses in `score_candidate` for production 4/5 neutrals and small penalties for distant production 1/2 neutrals.

Local evidence:
- `py_compile` passed.
- `main_v4_14_mild_opening_economy.py` vs current V4 submit [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over seeds 0-1 and both seats:
- Win rate: `50.0%`
- Average score diff: `+6007.50`
- Average survival: `499.00`
- `main_v4_14_mild_opening_economy.py` vs `bots/main_v4_1_counter_snipe.py` over seeds 0-1 and both seats:
- Win rate: `50.0%`
- Average score diff: `+2021.50`
- `main_v4_14_mild_opening_economy.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-1456.50`
- Average survival: `105.75`
- Loss classification vs `bots/hairate2.py`:
- `bad_overexpand_or_low_economy=2`
- `early_defense_collapse=2`

Decision:
- Do not promote V4.14 to the submit line.
- It is a strong self-play/economy branch, but it worsens the hairate2 score gap. It likely captures more economy but cannot defend or convert it against stronger planner pressure.
- Keep V4.14 as a useful branch for self-play pressure and future learned-evaluator data, not as current submission.

Next:
- The useful next step is collecting per-seed feature data from V4.5/V4.13/V4.14 outcomes, because hand-tuned opening weights are now producing conflicting objectives. This is a good setup for the planned lightweight learned evaluator or at least a small rule-based score table.

## 2026-06-06 - V4 feature collection and V4.15 stabilizer probe

Goal:
- Move from hand-tuned weight guessing toward feature-guided V4 iteration.
- Understand why V4.13/V4.14 improve some signals but fail against `hairate2`.

Change:
- Added [collect_v4_features.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/collect_v4_features.py:1), which writes JSONL outcome features at turns `25, 50, 80, 120, 180, 260`.
- Collected [v4_features_hairate2_opening.jsonl](C:/Users/yuu98/Desktop/kaggle/orbit-wars/v4_features_hairate2_opening.jsonl:1) for current V4 submit, V4.13, and V4.14 vs `hairate2`.
- Added [main_v4_15_midgame_stabilizer.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_15_midgame_stabilizer.py:1), based on V4.14 with turn 65-125 attack dampening and high-production regroup support.

Feature findings:
- Current V4 submit vs `hairate2`: average final diff `-785.50`, survival `81.50`.
- V4.13 vs `hairate2`: average final diff `-734.00`, survival `81.50`; slightly better score gap, same failure labels.
- V4.14 vs `hairate2`: average final diff `-1456.50`, survival `105.75`; survives longer on seed 0 but loses much more by turn 120.
- V4.14 keeps production alive at turn 80 in seed 0, but that economy is gone by turn 120.

Local evidence for V4.15:
- `py_compile` passed.
- `main_v4_15_midgame_stabilizer.py` vs current V4 submit [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over seeds 0-1 and both seats:
- Win rate: `50.0%`
- Average score diff: `+4901.00`
- `main_v4_15_midgame_stabilizer.py` vs `bots/main_v4_1_counter_snipe.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-2343.00`
- `main_v4_15_midgame_stabilizer.py` vs `bots/hairate2.py` over seeds 0-1 and both seats:
- Win rate: `0.0%`
- Average score diff: `-1456.50`
- Average survival: `105.75`

Decision:
- Do not promote V4.15.
- Midgame stabilization did not change the `hairate2` outcome compared with V4.14; it likely fires too late or is not selected against the decisive pressure.
- Keep [collect_v4_features.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/collect_v4_features.py:1) as useful infrastructure for the planned lightweight evaluator.

Next:
- Build a tiny feature-guided score table rather than another broad rule: for early neutral attacks, use observed seed outcomes to prefer the V4.13-style mild score gap improvement without triggering V4.14's overlong doomed survival.

## 2026-06-07 - V4.33-V4.38 no-action and defense-collapse probes

Goal:
- Investigate the video-observed "doing nothing while losing" behavior without spending many evaluation runs.
- Use fixed large seeds `54661125,190734863,7777777` to avoid small-seed overfitting.

Changes:
- Added `--seed-list` and score-diff output to [analyze_losses.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/analyze_losses.py:1).
- Extended [debug_shadow_activity.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/debug_shadow_activity.py:1) to count attack, greedy, and optional micro-regroup activity.
- Added micro-regroup probes `V4.33` to `V4.36`; none were promoted.
- Added [main_v4_37_midgame_reserve.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_37_midgame_reserve.py:1), a midgame reserve probe.
- Added [main_v4_38_defense_horizon.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v4_38_defense_horizon.py:1), which extends defense/safe-drain projection horizon without extending attack horizon.

Findings:
- Direct shadow expansion remained mostly inert; direct neutral capture was blocked by ETA/sun or needed ships.
- Micro-regroup can fire, but it easily destabilizes the planner. `V4.33` produced candidates but worsened seed `190734863`; stricter variants either did not fire or still looked unsafe.
- Current submit [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) vs `hairate2` on large seeds classified as `defense_collapse=2`, `bad_overexpand=1`.
- `V4.37` slightly improved survival against `hairate2` but worsened score diff.
- `V4.38` beat current `main.py` in the light 2-seed self check, but still lost all 3 large seeds to `hairate2`; survival improved only slightly while score gap remained bad.

Decision:
- Do not promote `V4.33`-`V4.38`.
- The next useful direction is not broad regroup. Focus on preventing the turn 80-120 collapse with better projected defense/counter-pressure, while preserving enough economy pressure to avoid a worse score gap.
## 2026-06-07 - V5.2 comeback-only multi-source probe

Goal:
- Continue the V5 multi-source planner direction without destabilizing the V4.43/V5.1 baseline.
- Reframe multi-source capture as a narrow comeback tool instead of a broad extra attack layer.

Change:
- Added [main_v5_2_comeback_multisource.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v5_2_comeback_multisource.py:1), copied from V5.1.
- Restricted multi-source capture to 2P turns `105-260`.
- Limited it to high-production enemy planets only, max 2 sources, and positions where the bot is behind or near-behind.
- Added post-capture hold filtering via existing retake-risk logic and a higher score floor.
- Let the multi-source layer scan all targets, while preserving the normal attack shortlist.

Local evidence:
- `py_compile` passed with `C:\tmp\ow\Scripts\python.exe -B`.
- V5.2 vs V5.1 over 20 seeds, both seats:
- Win rate: `57.5%`
- Average score diff: `-524.27`
- Crash rate: `0%`
- V5.2 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over 20 seeds, both seats:
- Win rate: `87.5%`
- Average score diff: `0.00`
- Crash rate: `0%`
- V5.2 vs `hairate2` over 20 seeds, both seats:
- Win rate: `100%`
- Average score diff: `+26243.15`
- Crash rate: `0%`
- Loss classification vs current `main.py`:
- `endgame_waste=1`
- `late_or_endgame_loss=4`
- `win=35`

Decision:
- Do not promote V5.2.
- The narrow comeback gate appears too restrictive: behavior is effectively close to current `main.py` on the tested seeds.
- V5.1's multi-source layer still carries more local upside, but it is too broad and spends into bad positions.

Next:
- Build V5.3 around instrumentation first: count candidate generation and selected multi-source plans by rejection reason.
- Then loosen one gate at a time, likely starting with hold-margin/retake checks rather than returning to neutral targets or 3-source plans.

## 2026-06-07 - V5.3 and V5.4 hairate-focused defense probes

Goal:
- Move V5 toward the real target matchup, `bots/hairate.py`, instead of optimizing only around `hairate2` or V5 mirrors.
- Reduce the dominant failure modes seen in `analyze_losses.py`: `defense_collapse` and `bad_overexpand`.

Change:
- Added [main_v5_3_hairate_defense.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v5_3_hairate_defense.py:1).
- V5.3 added direct-threat defense candidates and a heavy early neutral expansion penalty.
- Added [main_v5_4_selective_hold.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v5_4_selective_hold.py:1).
- V5.4 kept the V5.1 core, imported stronger projected-defense sizing from the older selective-defense branch, and added only a light neutral-expansion guard.

Local evidence:
- `py_compile` passed for both V5.3 and V5.4.
- Current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) vs `bots/hairate.py` over 20 seeds and both seats:
- Win rate: `0.0%`
- Average score diff: `-2800.78`
- V5.3 vs `bots/hairate.py` over 20 seeds and both seats:
- Win rate: `0.0%`
- Average score diff: `-3060.30`
- V5.3 vs current `main.py` over 20 seeds and both seats:
- Win rate: `47.5%`
- Average score diff: `-1413.72`
- V5.3 loss classification vs `bots/hairate.py`:
- `bad_overexpand=28`
- `defense_collapse=6`
- `endgame_waste=2`
- `late_stall=2`
- `opening_loss=2`
- V5.4 vs `bots/hairate.py` over 20 seeds and both seats:
- Win rate: `0.0%`
- Average score diff: `-2508.57`
- V5.4 vs current `main.py` over 20 seeds and both seats:
- Win rate: `70.0%`
- Average score diff: `+401.30`
- V5.4 loss classification vs `bots/hairate.py`:
- `bad_overexpand=16`
- `defense_collapse=21`
- `endgame_waste=2`
- `late_stall=1`

Decision:
- Reject V5.3.
- Keep V5.4 as the current best V5 branch against `hairate`, but do not promote it to `main.py` yet.

Next:
- V5.4 showed that heavy direct-threat defense overcorrects into worse overexpand patterns, while selective projected defense improves the `hairate` score gap without collapsing against `main.py`.
- The next V5 step should target the remaining `defense_collapse` cases with a narrow urgent-defense add-on on top of V5.4, not another broad attack planner change.

## 2026-06-07 - V5.5 instrumentation branch

Goal:
- Start the long-term planner rewrite with observability first instead of another blind tuning pass.
- Collect per-turn evidence for why V5 refuses actions, overspends, or fails to produce useful defense candidates.

Change:
- Added [main_v5_5_instrumented.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v5_5_instrumented.py:1), based on V5.4.
- When `OW_V5_LOG` is set, the bot appends one JSON line per turn with:
- candidate counts
- rejection reasons for attack/defense/multi-source gating
- source budgets
- shortlist contents
- selected actions before and after multi-source/shadow layers
- Added [summarize_v5_debug.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/summarize_v5_debug.py:1) to aggregate the JSONL logs by reason and step bucket.

Local evidence:
- `py_compile` passed.
- One-sample run with `OW_V5_LOG` produced `80` JSONL rows and the summarizer worked.
- On that sample, the dominant zero-action reasons were `attack_eta_too_long`, `attack_needed_gt_budget`, and `attack_light_overexpand_penalty`.
- The same sample produced only `2` defense candidates across `80` rows, which suggests the next useful step is not more attack tuning but making urgent defense opportunities appear earlier and more often.

Decision:
- Keep V5.5 as a research-only branch.

Next:
- Use V5.5 logs on fixed `hairate` seeds to identify where projected defense opportunities are missing versus merely losing in greedy selection.

## 2026-06-07 - V5.6 urgent-hold defense probe

Goal:
- Act on the V5.5 finding that projected losses are common, but actual defense candidates are rare and often blocked by `budget_too_small` or `eta_too_late`.
- Add only a narrow reserve-breaking defense path instead of another broad planner rewrite.

Change:
- Added [main_v5_6_urgent_hold.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v5_6_urgent_hold.py:1), based on V5.4.
- Added `urgent_hold` candidates for planets projected to fall within `5-7` turns.
- Let those urgent holds use a limited fraction of raw source ships instead of only safe-drain budget.
- Selected at most `2` urgent holds before the normal planner, then handed the reduced budget to the baseline V5.4 planner.

Local evidence:
- `py_compile` passed.
- V5.5 fixed-seed instrumentation vs `hairate` over collapse-heavy seeds `1,2,5,6,9,14,15,16,17,19`:
- `Rows: 2309`
- `defense_loss_planets: 1841`
- `defense_candidates: 245`
- dominant defense failures: `defense_eta_too_late=3536`, `defense_budget_too_small=3081`
- V5.6 vs `bots/hairate.py` over 20 seeds and both seats:
- Win rate: `0.0%`
- Average score diff: `-2277.43`
- V5.4 baseline vs `bots/hairate.py` on the same suite:
- Average score diff: `-2508.57`
- V5.6 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over 20 seeds and both seats:
- Win rate: `55.0%`
- Average score diff: `+866.85`
- V5.6 loss classification vs `bots/hairate.py`:
- `bad_overexpand=14`
- `defense_collapse=23`
- `endgame_waste=2`
- `late_stall=1`

Decision:
- Keep V5.6 as the current strongest V5 branch against `hairate`.
- Do not promote yet; it still loses all tested games.

Next:
- The urgent-hold idea helped the score gap, but `defense_collapse` is still high.
- The next useful step is not broader reserve breaking; it is multi-source defense or earlier frontline relocation so threatened planets can actually be reached in time.

## 2026-06-07 - V5.7 and V5.8 pre-collapse defense probes

Goal:
- Continue from V5.6 by testing whether "one source cannot arrive in time" is the remaining bottleneck against `hairate`.
- Probe two next-step ideas:
- V5.7: 2-source emergency defense on a single threatened high-value planet
- V5.8: early frontline relay to pre-position ships before the collapse turn

Change:
- Added [main_v5_7_multihold.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v5_7_multihold.py:1), based on V5.6.
- V5.7 tries one narrow `multi_hold` defense before the regular planner.
- Added [main_v5_8_frontline_relay.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v5_8_frontline_relay.py:1), also based on V5.6.
- V5.8 adds one early/midgame `frontline_relay` from a low-pressure rear planet into a high-pressure high-production ally.

Local evidence:
- `py_compile` passed for both V5.7 and V5.8.
- V5.6 vs `bots/hairate.py` over 20 seeds and both seats:
- Average score diff: `-2277.43`
- V5.7 vs `bots/hairate.py` over 20 seeds and both seats:
- Average score diff: `-2317.55`
- V5.7 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over 20 seeds and both seats:
- Win rate: `55.0%`
- Average score diff: `+504.55`
- V5.7 loss classification vs `bots/hairate.py`:
- `bad_overexpand=15`
- `defense_collapse=22`
- `endgame_waste=1`
- `late_stall=2`
- V5.8 vs `bots/hairate.py` over 20 seeds and both seats:
- Average score diff: `-2873.95`
- V5.8 vs current `main.py` over 20 seeds and both seats:
- Win rate: `25.0%`
- Average score diff: `-2262.15`
- V5.8 loss classification vs `bots/hairate.py`:
- `bad_overexpand=22`
- `defense_collapse=16`
- `endgame_waste=1`
- `late_stall=1`

Decision:
- Reject V5.7.
- Reject V5.8.
- Keep V5.6 as the current best V5 branch against `hairate`.

Next:
- Simple pre-collapse defense additions are reaching diminishing returns.
- The next serious step should move away from layered patches and into a unified planner rewrite where defense, opening expansion, and repositioning compete in one action pool.

## 2026-06-07 - V6.0 unified action-pool prototype

Goal:
- Start the real planner rewrite instead of adding another V5 patch.
- Make relay, urgent hold, defense, counter-snipe, and attack compete in one shared greedy pool.

Change:
- Added [main_v6_0_unified_pool.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_0_unified_pool.py:1), based on V5.6.
- Kept the V5.6 evaluation helpers and shadow layer.
- Added `frontline_relay` candidate generation directly into the main planner branch.
- Removed the old "select urgent holds first, then run the normal planner" flow.
- Let `urgent_hold`, `frontline_relay`, `defense`, `counter_snipe`, and `attack` all enter one shared candidate pool and one shared `greedy_select`.
- Allowed only relay and urgent-hold generation to see reserve-broken budgets; normal attack and defense candidates still use safe-drain budgets.

Local evidence:
- `py_compile` passed.
- V6.0 vs `bots/hairate.py` over 20 seeds and both seats:
- Win rate: `0.0%`
- Average score diff: `-3061.30`
- Average survival turn: `130.05`
- V6.0 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over 20 seeds and both seats:
- Win rate: `35.0%`
- Average score diff: `-1278.42`
- V6.0 loss classification vs `bots/hairate.py`:
- `bad_overexpand=18`
- `defense_collapse=20`
- `endgame_waste=1`
- `late_stall=1`

Decision:
- Keep V6.0 as a research branch only.
- Do not promote it; this first unified version is clearly worse than V5.6.

Takeaway:
- The unified action-pool direction is still the right architectural move, but the first scoring/gating is too eager.
- `frontline_relay` plus reserve access created too much early/midgame overextension before the hold logic could pay it back.
- The next version should narrow relay sharply and make position-building conditional on a stronger map signal, not just local pressure gap.

Next:
- Build V6.1 around a stricter opening/frontier map:
- score outer-ring high-production lane control explicitly
- cap relay spending harder
- require stronger local support before non-defensive reserve breaking
- keep the unified pool, but separate "reserve-using structural moves" from ordinary tactical moves with a tighter admission gate

## 2026-06-07 - V6.1 frontier-gated unified pool

Goal:
- Keep the V6 unified planner shape, but remove the over-eager reserve relay behavior from V6.0.
- Make structural reserve use depend on a stronger frontier signal instead of a simple local pressure gap.

Change:
- Added [main_v6_1_frontier_gated.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_1_frontier_gated.py:1), based on V6.0.
- Added helper signals for:
- `orbital_ring_value`
- `friendly_support_count`
- `frontier_gate_score`
- Tightened `frontline_relay` so it now requires:
- shorter timing window
- larger pressure gap
- positive outer-ring gain
- at least one nearby friendly support planet
- stronger map-gate score before entering the unified pool
- Capped relay send size harder and reduced reserve-break access from `0.34/0.42` to `0.28/0.34` of source ships.

Local evidence:
- `py_compile` passed.
- V6.1 vs `bots/hairate.py` over 20 seeds and both seats:
- Win rate: `0.0%`
- Average score diff: `-2390.28`
- Average survival turn: `126.88`
- V6.0 baseline on the same suite:
- Average score diff: `-3061.30`
- V5.6 baseline on the same suite:
- Average score diff: `-2277.43`
- V6.1 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over 20 seeds and both seats:
- Win rate: `55.0%`
- Average score diff: `+777.00`
- V6.1 loss classification vs `bots/hairate.py`:
- `bad_overexpand=13`
- `defense_collapse=22`
- `early_defense_collapse=2`
- `endgame_waste=2`
- `late_stall=1`

Decision:
- Keep V6.1 as the best V6 research branch so far.
- Do not promote it yet; it still trails V5.6 against `hairate`.

Takeaway:
- The unified planner can be stabilized.
- The frontier-gated relay cut the V6.0 overexpansion problem materially and restored positive performance against current `main.py`.
- The remaining gap to `hairate` is now less about reckless expansion and more about early-to-mid hold quality.

Next:
- Build V6.2 around unified early hold strength:
- strengthen `urgent_hold` and normal `defense` scoring inside the same pool
- explicitly reward preserving newly won high-production planets through the first counter window
- consider a small multi-source hold / handoff action before re-enabling broader structural relocation

## 2026-06-07 - Fixed hairate benchmarks and planner search tooling

Goal:
- Move beyond ad hoc hand-tuning by adding a repeatable fixed benchmark and a generic config search script.
- Make it easy to compare V6 branches under the exact same `hairate` seed set.

Change:
- Added fixed benchmark specs:
- [benchmarks/hairate_fixed_2p.json](C:/Users/yuu98/Desktop/kaggle/orbit-wars/benchmarks/hairate_fixed_2p.json:1)
- [benchmarks/hairate_focus_2p.json](C:/Users/yuu98/Desktop/kaggle/orbit-wars/benchmarks/hairate_focus_2p.json:1)
- Added [search_planner_params.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/search_planner_params.py:1)
- The search script:
- loads a planner module dynamically
- rewrites `CONFIG_2P` / `CONFIG_4P` through `PlannerConfig`
- expands repeated `--set key=v1,v2,...` dimensions into a cartesian product
- runs every variant on a fixed benchmark spec
- ranks variants by average score diff, then win rate and survival
- can emit JSONL summaries for later analysis

Smoke test:
- `py_compile` passed for `search_planner_params.py`.
- Ran:
- `C:\tmp\ow\Scripts\python.exe -B .\search_planner_params.py --agent bots\main_v6_1_frontier_gated.py --benchmark benchmarks\hairate_focus_2p.json --set roi_threshold=2.0,2.2 --set reserve_margin=2,3 --limit 3`
- Result on the focused `hairate` benchmark:
- `base`: `diff=-2080.85`
- `roi_threshold=2.0 | reserve_margin=2`: `diff=-2201.00`
- `roi_threshold=2.0 | reserve_margin=3`: `diff=-2684.40`
- In that small sample, `base` remained best.

Takeaway:
- We now have a formal `hairate` benchmark layer and a reusable tuning loop.
- From here, parameter experiments can be batched and compared without changing the bot code by hand each time.

Next:
- Use the focused benchmark for fast iteration inside a branch.
- Use the broad benchmark before promoting any candidate.
- After V6.2 exists, run targeted sweeps over:
- `roi_threshold`
- `reserve_margin`
- `defense_horizon`
- `max_actions`

## 2026-06-07 - V6.2 early-hold push

Goal:
- Push the unified planner further toward `hairate` by valuing early hold more aggressively.
- Protect newly won high-production planets through the first enemy counter window.

Change:
- Added [main_v6_2_early_hold.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_2_early_hold.py:1), based on V6.1.
- Added `hold_priority_bonus` and used it in:
- defense scoring
- urgent-hold scoring
- high-production attack scoring
- Let high-production attacks send extra hold margin when budget allowed.
- Tightened/filtered high-production attacks by rejecting some captures that still looked too easy to retake.
- Let normal `defense` candidate generation see reserve-side budgets inside the unified planner.
- Made urgent-hold reserve breaking slightly more permissive.

Local evidence:
- `py_compile` passed.
- V6.2 vs `bots/hairate.py` over 20 seeds and both seats:
- Win rate: `0.0%`
- Average score diff: `-2643.90`
- Average survival turn: `126.45`
- V6.1 baseline on the same suite:
- Average score diff: `-2390.28`
- V6.2 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over 20 seeds and both seats:
- Win rate: `37.5%`
- Average score diff: `-810.45`
- V6.1 baseline vs current `main.py`:
- Win rate: `55.0%`
- Average score diff: `+777.00`
- V6.2 loss classification vs `bots/hairate.py`:
- `bad_overexpand=19`
- `defense_collapse=19`
- `endgame_waste=2`

Decision:
- Reject V6.2.
- Keep V6.1 as the best current V6 branch.

Takeaway:
- The architectural target was right, but the implementation was too broad.
- "Early hold" cannot be improved just by adding more hold margin and more reserve access across the board.
- In this form, the planner over-commits into expensive positions and recreates the overexpansion problem.

Next:
- Go narrower than V6.2:
- keep V6.1 as the base
- only add hold-aware attack sizing on very high-value targets
- use the new focused `hairate` benchmark plus parameter search before broad evaluation

## 2026-06-07 - V6.3 selective hold-aware attack with focused benchmark gating

Goal:
- Recover the useful part of V6.2 without reopening broad overcommit.
- Use the new "upper-tier style" workflow: first test on the fixed focused `hairate` benchmark, then promote the best setting to broad evaluation.

Change:
- Added [main_v6_3_selective_hold_attack.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_3_selective_hold_attack.py:1), based on V6.1.
- Added a narrow `selective_hold_target` gate:
- only early/midgame
- only very high-value captures (`prod 5`)
- requires nearby friendly support
- tighter neutral timing gate
- Only those targets can receive extra hold-aware attack sizing.
- Added a modest score bonus for hold-safe selective captures instead of broad hold inflation.

Focused benchmark search:
- Ran [search_planner_params.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/search_planner_params.py:1) on [benchmarks/hairate_focus_2p.json](C:/Users/yuu98/Desktop/kaggle/orbit-wars/benchmarks/hairate_focus_2p.json:1):
- `base`: `diff=-1978.15`
- `roi_threshold=2.1 | reserve_margin=2`: `diff=-2071.85`
- `roi_threshold=2.2 | reserve_margin=2`: `diff=-2082.45`
- `roi_threshold=2.1 | reserve_margin=3`: `diff=-2412.75`
- `roi_threshold=2.2 | reserve_margin=3`: `diff=-2430.60`
- Best focused result was the base V6.3 setting, so that setting was promoted to broad testing.

Local evidence:
- `py_compile` passed.
- V6.3 vs `bots/hairate.py` on the focused benchmark:
- Average score diff: `-2078.35`
- V6.1 baseline on the earlier focused sample:
- Average score diff: `-2080.85`
- V6.3 vs `bots/hairate.py` over 20 seeds and both seats:
- Win rate: `0.0%`
- Average score diff: `-2279.40`
- V6.1 baseline on the same broad suite:
- Average score diff: `-2390.28`
- V5.6 baseline on the same broad suite:
- Average score diff: `-2277.43`
- V6.3 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over 20 seeds and both seats:
- Win rate: `55.0%`
- Average score diff: `+929.95`
- V6.3 loss classification vs `bots/hairate.py`:
- `bad_overexpand=12`
- `defense_collapse=23`
- `early_defense_collapse=2`
- `endgame_waste=2`
- `late_stall=1`

Decision:
- Keep V6.3 as the current best V6 branch.
- It is still not ready to replace the best overall baseline, but it nearly matches V5.6 against `hairate` while being structurally closer to the final unified planner direction.

Takeaway:
- The narrow hold-aware attack idea works better than broad early-hold inflation.
- Overexpansion improved slightly again, but the main remaining wall is still `defense_collapse`.
- The new search/benchmark workflow successfully prevented us from promoting a worse parameter setting.

Next:
- Use V6.3 as the new research base.
- Explore one narrow next step:
- either selective multi-source hold on threatened `prod 5` planets
- or earlier same-pool defense candidate generation for collapse-heavy seeds
- Run focused benchmark first, then broad benchmark only for the winner

## 2026-06-07 - V6.4 and V6.5 defensive probes under focused benchmark gating

Goal:
- Continue growing the unified planner while staying disciplined about promotion.
- Test two narrow defense-collapse ideas, but gate both through the focused `hairate` benchmark before spending time on broad evaluation.

### V6.4 selective multihold

Change:
- Added [main_v6_4_selective_multihold.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_4_selective_multihold.py:1), based on V6.3.
- Added a very narrow `multi_hold` rescue:
- 2-player only
- `prod 5` planets only
- projected loss within `6` turns
- at most 2 sources
- appended before normal multi-capture

Focused benchmark result:
- Search over `roi_threshold={2.1,2.2}` and `reserve_margin={2,3}` on [benchmarks/hairate_focus_2p.json](C:/Users/yuu98/Desktop/kaggle/orbit-wars/benchmarks/hairate_focus_2p.json:1):
- `base`: `diff=-1978.15`
- same as V6.3 base on the same focused set

Decision:
- Reject V6.4.
- The selective multihold probe did not improve the focused benchmark and likely did not activate meaningfully on the target seeds.

### V6.5 priority defense

Change:
- Added [main_v6_5_priority_defense.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_5_priority_defense.py:1), also based on V6.3.
- Added `priority_defense` candidates:
- only for collapsing `prod 5` planets
- only within `4-6` turns
- may use limited reserve through the unified pool

Focused benchmark result:
- Search over `roi_threshold={2.1,2.2}` and `reserve_margin={2,3}` on the same focused benchmark:
- `base`: `diff=-2002.15`
- worse than V6.3 base `diff=-1978.15`

Decision:
- Reject V6.5.

Takeaway:
- Narrowing the branch and using the focused benchmark first saved time and prevented broad runs on weak candidates.
- The immediate wall is probably not "one more defense patch."
- V6.3 remains the current best V6 branch:
- broad `hairate` diff `-2279.40`
- broad `main.py` diff `+929.95`

Next:
- Keep V6.3 as the active research base.
- The next promising angle is earlier shape control rather than another late rescue:
- attack shortlist quality
- support-aware opening capture ordering
- or defense candidate generation that reacts before the projected loss window gets short

## 2026-06-07 - Shot audit for sun loss and likely aim waste

Goal:
- Check whether visible "shots drifting into the sun" or other wasted launches are a major real bottleneck.
- Measure this from replay data instead of guessing from a few watched games.

Change:
- Added [audit_shots.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/audit_shots.py:1).
- The audit replays fixed seeds, reads each recorded `action`, matches launched fleets, then classifies each disappearance as:
- `sun_loss`
- `out_of_bounds`
- `planet_hit`
- `unknown_loss`
- It also keeps a rough `target_guess` from launch angle so we can estimate likely wrong-planet hits.

Local evidence:
- Ran `main.py` vs `bots/hairate.py` on seeds `0-9`, both seats:
- launches: `1496`
- `sun_loss=14` (`0.9%`)
- `out_of_bounds=113` (`7.6%`)
- `planet_hit=1364` (`91.2%`)
- `target_hit_guess=825` (`55.1%`)
- `wrong_planet_hit_guess=539` (`36.0%`)
- Ran [main_v6_3_selective_hold_attack.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_3_selective_hold_attack.py:1) on the same suite:
- launches: `1356`
- `sun_loss=13` (`1.0%`)
- `out_of_bounds=89` (`6.6%`)
- `planet_hit=1246` (`91.9%`)
- `target_hit_guess=696` (`51.3%`)
- `wrong_planet_hit_guess=550` (`40.6%`)

Takeaway:
- Sun loss is real, but it is not the dominant failure mode in these samples.
- The larger waste bucket is fleets that run out of bounds, plus a sizable amount of likely wrong-planet contact under the current rough target-guess heuristic.
- This suggests shot quality is worth improving, but the first payoff is probably not just stricter sun safety; it is better target intent / aiming fidelity for selected attacks.

Note:
- `wrong_planet_hit_guess` is heuristic, not exact ground truth; it uses the nearest angular target at launch as the guessed intent.

## 2026-06-07 - Friendly rotating-planet aim fix

Goal:
- Fix the most actionable shot-quality issue after the audit.
- Reduce `out_of_bounds` launches caused by aiming at the current position of rotating friendly planets during defense, relay, and regroup moves.

Finding:
- The upgraded shot audit on [main_v6_3_selective_hold_attack.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_3_selective_hold_attack.py:1) over seeds `0-9`, both seats vs `bots/hairate.py` showed:
- `out_of_bounds=89` (`6.6%`)
- `out_kind_friendly=57` (`64.0%` of out_of_bounds)
- `out_rot_rotating=67` (`75.3%` of out_of_bounds)
- This strongly suggested that many wasted shots were friendly-targeted actions aimed at stale positions on rotating planets.

Change:
- Added [main_v6_6_friendly_aim_fix.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_6_friendly_aim_fix.py:1), based on V6.3.
- Updated these friendly-targeted candidate builders to use predicted arrival-time positions instead of current positions:
- `build_defense_candidates`
- `build_urgent_hold_candidates`
- `build_frontline_relay_candidates`
- `build_regroup_candidates`
- Added a small `validate_intercept_window(...)` check so these actions are dropped when the straight-line flight is unlikely to meet the rotating target within a sampled arrival window.

Local evidence:
- `py_compile` passed.
- Shot audit on `V6.6` over seeds `0-9`, both seats vs `bots/hairate.py`:
- launches: `1553`
- `sun_loss=13` (`0.8%`)
- `out_of_bounds=48` (`3.1%`)
- `planet_hit=1472` (`94.8%`)
- `out_kind_friendly=11` (`22.9%` of out_of_bounds)
- `out_rot_rotating=20` (`41.7%` of out_of_bounds)
- Compared with `V6.3` on the same suite:
- `out_of_bounds`: `89 -> 48`
- `out_kind_friendly`: `57 -> 11`
- `out_rot_rotating`: `67 -> 20`
- V6.6 vs `bots/hairate.py` over 20 seeds and both seats:
- Win rate: `2.5%`
- Average score diff: `-2060.10`
- V6.3 baseline on the same broad suite:
- Average score diff: `-2279.40`
- V6.6 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1) over 20 seeds and both seats:
- Win rate: `72.5%`
- Average score diff: `+1300.50`
- V6.3 baseline vs current `main.py`:
- Win rate: `55.0%`
- Average score diff: `+929.95`

Decision:
- Promote V6.6 as the new best current research branch.

Takeaway:
- Shot-quality cleanup was worth doing.
- Sun loss was not the big lever, but fixing rotating friendly-planet aiming materially reduced wasted launches and improved both `hairate` and `main.py` results.

## 2026-06-07 - Search baseline fix and attack re-aim probe

Goal:
- Clean up the tuning tool so benchmark sweeps compare against each bot's real baseline.
- Probe the next shot-quality hypothesis: attack candidates may also be aiming with the wrong ship-speed assumption.

Change:
- Fixed [search_planner_params.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/search_planner_params.py:1) so sweeps now start from the bot module's actual `CONFIG_2P` / `CONFIG_4P`, not from a generic `PlannerConfig()` default.
- Extended [audit_shots.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/audit_shots.py:1) to break out `wrong_planet_hit_guess` by:
- target kind
- rotating/static
- ship bucket
- target production
- Added [main_v6_7_attack_reaim.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_7_attack_reaim.py:1), based on V6.6.
- V6.7 recomputes attack intercepts with the actual final `send` size and validates the intercept window before firing.

Audit evidence:
- V6.6 wrong-hit heuristic over seeds `0-9`, both seats vs `bots/hairate.py`:
- `wrong_planet_hit_guess=630` (`40.6%`)
- split: `enemy=310`, `neutral=260`, `friendly=60`
- mostly `static=425` rather than `rotating=205`
- mostly medium fleets: `16-31 ships = 219`, `8-15 ships = 175`
- V6.7 over the same suite:
- `wrong_planet_hit_guess=536` (`40.5%`)
- but `sun_loss` worsened: `13 -> 26`
- and `out_of_bounds` worsened: `48 -> 61`

Match evidence:
- V6.7 vs `bots/hairate.py` over 20 seeds and both seats:
- Average score diff: `-2221.07`
- V6.6 baseline:
- Average score diff: `-2060.10`
- V6.7 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1):
- Win rate: `50.0%`
- Average score diff: `+340.82`
- V6.6 baseline vs current `main.py`:
- Win rate: `72.5%`
- Average score diff: `+1300.50`

Decision:
- Keep the search baseline fix.
- Reject V6.7.
- Keep V6.6 as the current best branch.

Takeaway:
- The next ROI was not "recompute every attack with final send size."
- The earlier fix was the good one: stale aiming on rotating friendly targets.
- The remaining wrong-hit bucket appears to be mostly enemy and neutral attacks on static targets, so the next useful investigation should focus there rather than friendly-target motion.

## 2026-06-07 14:37 V6.8 attack lane filter + evaluate stdout fix

Changes:
- Added [main_v6_8_attack_lane_filter.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_8_attack_lane_filter.py:1), based on V6.6.
- Added `path_has_intercept_conflict(...)` to reject attack candidates whose route appears to pass through another planet before the intended target.
- Fixed [evaluate.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/evaluate.py:1) so single-worker runs no longer silence their own summary output.

Quick audit:
- V6.8 vs `bots/hairate.py`, seeds `0-9`, both seats:
- `launches=1369`
- `sun_loss=14` (`1.0%`)
- `out_of_bounds=52` (`3.8%`)
- `wrong_planet_hit_guess=568` (`41.5%`)
- Compared with V6.6:
- `out_of_bounds` worsened from `48` to `52`
- `wrong_planet_hit_guess` worsened from `40.6%` to `41.5%`

Quick match check on seeds `0-3`, both seats:
- V6.8 vs `bots/hairate.py`: `0/8` wins, average score diff `-1948.38`
- V6.6 baseline vs `bots/hairate.py`: `1/8` wins, average score diff `-1142.25`
- V6.8 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1): `6/8` wins, average score diff `+1144.25`
- V6.6 baseline vs current `main.py`: `5/8` wins, average score diff `-99.62`

Decision:
- Reject V6.8 as the next main research branch.
- Keep V6.6 as the best current branch.

Takeaway:
- A simple route-conflict filter on attack shots is too blunt in its current form.
- It can improve some local `main.py` matchups while still clearly hurting the `hairate` objective.

## 2026-06-07 14:45 V6.9 attack ambiguity penalty

Changes:
- Added [main_v6_9_attack_ambiguity_penalty.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_9_attack_ambiguity_penalty.py:1), based on V6.6.
- Instead of hard-rejecting suspicious attack paths, added `attack_ambiguity_penalty(...)` to downscore attacks whose heading overlaps with a nearer planet lane.
- Kept the change narrow: attack scoring only, no planner-wide structural changes.

Quick gate results, seeds `0-3`, both seats:
- V6.9 vs `bots/hairate.py`: `2/8` wins, average score diff `+107.75`
- V6.6 baseline vs `bots/hairate.py`: `1/8` wins, average score diff `-1142.25`
- V6.9 vs current [main.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/main.py:1): `3/8` wins, average score diff `+91.25`
- V6.6 baseline vs current `main.py`: `5/8` wins, average score diff `-99.62`

Audit on the same short suite vs `bots/hairate.py`:
- `launches=886`
- `out_of_bounds=36` (`4.1%`)
- `wrong_planet_hit_guess=287` (`32.4%`)
- This is a noticeable drop in guessed wrong hits relative to the earlier V6.6 long audit (`40.6%`), though with a shorter sample and a slight increase in `out_of_bounds`.

Broader check, seeds `0-9`, both seats:
- V6.9 vs `bots/hairate.py`: `2/20` wins, average score diff `-1383.55`
- V6.6 baseline vs `bots/hairate.py`: `1/20` wins, average score diff `-1878.20`
- V6.9 vs current `main.py`: `11/20` wins, average score diff `+1596.65`
- V6.6 baseline vs current `main.py`: `13/20` wins, average score diff `+1705.15`

Decision:
- Keep V6.9 as the new best `hairate`-focused V6 branch.
- Do not promote it as a general replacement yet, because it gives back a little against current `main.py`.

Takeaway:
- A soft ambiguity penalty works much better than the earlier hard lane filter.
- This is the kind of result-oriented V6.6-line change we want: narrow, measurable, and actually better against the main target.

## 2026-06-07 15:11 Research loop automation

Changes:
- Added [run_research_loop.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/run_research_loop.py:1).
- The runner evaluates a candidate bot against configured opponents, runs optional shot audit, writes `results.jsonl`, `run.json`, and a compact `summary.md`.
- Added `--baseline` support so candidate-vs-baseline deltas are computed automatically on the same seeds/seats/opponents.
- Updated [.gitignore](C:/Users/yuu98/Desktop/kaggle/orbit-wars/.gitignore:1) to ignore generated `research_runs/` outputs and temporary evaluation files.

Smoke checks:
- `py_compile` passed.
- `smoke_v69_gate` ran V6.9 on the gate suite:
- vs `bots/hairate.py`: `2/8` wins, average score diff `+107.75`
- vs current `main.py`: `3/8` wins, average score diff `+91.25`
- shot audit summary was written under `research_runs/smoke_v69_gate/`.
- `smoke_v610_vs_v69` ran V6.10 against V6.9 baseline:
- hairate delta: win `-12.5%`, average diff `-1093.62`
- main delta: win `+0.0%`, average diff `+0.00`

Decision:
- Use `run_research_loop.py` as the default gate for new V6.9-line candidates.
- This should reduce Codex turns spent on repeated command construction, result gathering, and manual comparison.

## 2026-06-07 15:24 V6.11 prod1 strict and V6.12 attack confidence

Changes:
- Added [main_v6_11_prod1_strict.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_11_prod1_strict.py:1), based on V6.9.
- V6.11 aggressively gated `prod1` neutral attacks by support distance and enemy pressure, plus an extra score penalty.
- Added [main_v6_12_attack_confidence_bonus.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_12_attack_confidence_bonus.py:1), also based on V6.9.
- V6.12 kept the ambiguity penalty and added a soft confidence bonus for attacks with cleaner lanes, closer support, high production, and static targets.

V6.11 gate result (`v611_vs_v69_gate`):
- vs `bots/hairate.py`: `0/8` wins, average score diff `-1922.00`
- baseline V6.9 on same suite: `2/8` wins, average score diff `+107.75`
- vs current `main.py`: `4/8` wins, average score diff `+1480.38`

Decision for V6.11:
- Reject.
- It reduced some low-value neutral noise, but it clearly cut too much attacking power against the main target.

V6.12 gate result (`v612_vs_v69_gate`):
- vs `bots/hairate.py`: `2/8` wins, average score diff `-411.88`
- vs current `main.py`: `8/8` wins, average score diff `+3280.50`

V6.12 broader result (`v612_vs_v69_standard`):
- vs `bots/hairate.py`: `2/20` wins, average score diff `-1694.65`
- baseline V6.9 on same suite: `2/20` wins, average score diff `-1383.55`
- vs current `main.py`: `19/20` wins, average score diff `+2671.25`
- baseline V6.9 on same suite: `11/20` wins, average score diff `+1596.65`

Focused benchmark check (`v612_vs_v69_hairate_focus`):
- `hairate_focus_2p`: `0/20` wins, average score diff `-2120.65`

Decision for V6.12:
- Do not replace V6.9 as the `hairate`-focused branch.
- Keep V6.12 as an interesting generalist candidate because it is much stronger against current `main.py`.

Takeaway:
- The `prod1` strict gate was too blunt.
- The confidence-bonus idea is directionally useful, but in its current form it improves local/general matchups more than the hard target benchmark.

## 2026-06-07 16:02 opening experiments V6.13-V6.15

Changes:
- Added [main_v6_13_opening_bias.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_13_opening_bias.py:1).
- V6.13 only added a light opening bias in shortlist ranking.
- Added [main_v6_14_opening_layer.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_14_opening_layer.py:1).
- V6.14 introduced a much harder opening-only layer with explicit neutral filtering.
- Added [main_v6_15_opening_params.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_15_opening_params.py:1).
- V6.15 backed off from hard filters and instead used `hairate`-style opening parameters: neutral travel cap, low-prod soft skip, static/support bonuses, and early enemy de-prioritization.

V6.13 result:
- Gate vs `bots/hairate.py`: unchanged from V6.9 (`2/8`, diff `+107.75`)
- `hairate3_ladder_2p`: slightly worse than V6.9 (`-1919.68` vs `-1906.92`)
- Decision: reject; too weak to matter.

V6.14 result:
- Gate vs `bots/hairate.py`: `0/8`, diff `-2442.62`
- Gate vs current `main.py`: `0/8`, diff `-2874.38`
- `hairate3_ladder_2p`: `1/60`, diff `-1974.68`
- Decision: reject; opening layer was far too aggressive and cut good options.

V6.15 result:
- Gate vs `bots/hairate.py`: `2/8`, diff `-87.50`
- Gate vs current `main.py`: `1/8`, diff `-2104.25`
- `hairate3_ladder_2p`: `2/60`, diff `-1841.48`
- Compared with V6.9 on `hairate3_ladder_2p`, V6.15 improved average diff (`-1841.48` vs `-1906.92`).
- But it badly regressed against current `main.py`.

Takeaway:
- The opening weakness is real.
- A light opening bias is too weak, and a hard opening layer is too destructive.
- V6.15 is the first opening experiment that helped one hard benchmark (`hairate3_ladder_2p`), but it is still not a promotion candidate because the regression against `main.py` is too large.

## 2026-06-07 15:31 hairate2 benchmark setup

Context:
- Checked whether `bots/hairate2.py` should be included as another regular benchmark target.
- `bots/hairate2.py` imports repo-local `bots/orbit_lite` plus external `torch`.

Environment check:
- In the current `C:\\tmp\\ow` environment, importing `torch` failed with `ModuleNotFoundError`.
- That means `hairate2` is not currently a fair always-on benchmark in this local setup, because it cannot run as intended without additional dependencies.

Changes:
- Added [hairate2_fixed_2p.json](C:/Users/yuu98/Desktop/kaggle/orbit-wars/benchmarks/hairate2_fixed_2p.json:1).
- Added [hairate2_focus_2p.json](C:/Users/yuu98/Desktop/kaggle/orbit-wars/benchmarks/hairate2_focus_2p.json:1).
- Both benchmark files document that `torch` is required.

Decision:
- `hairate2` should be part of the benchmark set eventually.
- But for now, `hairate` remains the primary trustworthy strong-opponent benchmark in this environment.

## 2026-06-07 16:31 V6.16-V6.19 aggressive opening/pressure experiments

Context:
- User wanted a bigger experiment, not only small manual scoring tweaks.
- The target was to preserve the `main.py` 90%+ local win-rate requirement while continuing the longer `hairate`/`hairate3` research line.

Changes:
- Added [main_v6_16_aggressive_opening_confidence.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_16_aggressive_opening_confidence.py:1).
- V6.16 combined V6.12's confidence attack layer with a much stronger `hairate`-style opening neutral shape layer.
- Added [main_v6_17_soft_opening_aggression.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_17_soft_opening_aggression.py:1).
- V6.17 softened V6.16 by removing the hard opening neutral travel filter and reducing opening neutral bonuses/penalties.
- Added [main_v6_18_early_enemy_pressure.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_18_early_enemy_pressure.py:1).
- V6.18 left neutral opening alone and instead boosted early pressure on enemy high-production planets.
- Added [main_v6_19_soft_opening_pressure_hybrid.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_19_soft_opening_pressure_hybrid.py:1).
- V6.19 combined V6.17's soft opening shaping with a stronger early high-production enemy pressure bonus.

Results vs V6.12 baseline:
- V6.16 gate: `hairate` unchanged at `2/8`, diff improved by `+72.25`; `main.py` collapsed to `4/8`, so reject as too disruptive.
- V6.17 gate: `hairate` and `main.py` identical to V6.12 on gate; `hairate3_ladder_2p` diff `-1829.02`.
- V6.18 gate: `hairate` unchanged at `2/8`, diff improved by `+43.50`; `main.py` stayed `8/8`; `hairate3_ladder_2p` worsened to `-1858.62`.
- V6.19 gate: `hairate` unchanged at `2/8`, diff improved by `+43.50`; `main.py` stayed `8/8`; `hairate3_ladder_2p` matched V6.17 at `-1829.02`.
- V6.19 standard: `hairate` `2/20`, average diff `-1618.75` (`+75.90` vs V6.12); `main.py` `19/20`, average diff `+2568.55`.

Decision:
- Reject V6.16 despite hard-benchmark movement because it violates the `main.py` stability requirement.
- Keep V6.17 as a safe opening-shape branch.
- Keep V6.19 as the best current generalist branch from this batch: it preserves `main.py` 95% on standard while improving `hairate` average diff vs V6.12.
- V6.9 remains better than V6.19 for pure `hairate` average diff, but V6.19 is much stronger than V6.9 against current `main.py`.

Takeaway:
- Hard copying `hairate` opening filters is dangerous.
- Soft opening shaping plus early enemy production pressure is a more stable direction.
- The next likely high-value step is an automated parameter sweep over the V6.19 constants instead of another hand-tuned single branch.

## 2026-06-07 15:36 hairate3 check

Context:
- Added a new local opponent candidate: [hairate3.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/hairate3.py:1).

Dependency check:
- `hairate3.py` is pure Python in this repo layout.
- Unlike `hairate2.py`, it does not require `torch`.
- So it is easy to run and can be benchmarked immediately.

Quick strength check, seeds `0-3`, both seats:
- [main_v6_9_attack_ambiguity_penalty.py](C:/Users/yuu98/Desktop/kaggle/orbit-wars/bots/main_v6_9_attack_ambiguity_penalty.py:1) vs `bots/hairate3.py`: `8/8` wins, average score diff `+23269.25`
- `bots/hairate.py` vs `bots/hairate3.py`: `8/8` wins, average score diff `+915.50`
- `bots/hairate3.py` vs `bots/hairate.py`: `0/8` wins, average score diff `-915.50`

Interpretation:
- `hairate3` does not currently look like a stronger primary benchmark than `hairate`.
- It may still be useful as an additional regression opponent, but not as the main "strong-opponent" target.

Changes:
- Added [hairate3_fixed_2p.json](C:/Users/yuu98/Desktop/kaggle/orbit-wars/benchmarks/hairate3_fixed_2p.json:1).
- Added [hairate3_focus_2p.json](C:/Users/yuu98/Desktop/kaggle/orbit-wars/benchmarks/hairate3_focus_2p.json:1).
