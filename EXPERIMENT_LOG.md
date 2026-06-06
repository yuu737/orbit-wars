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
