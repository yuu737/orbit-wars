# Recent Orbit Wars Experiment Log

Updated: 2026-06-23

This file summarizes the recent `sample` / `hairate` experiments that were not
covered by the older `EXPERIMENT_LOG.md`.

## Current Development Baseline

Use `sample110_4p_oneply_coord_from109` as the main development baseline going forward.

Reason:
- 4P:
  - `sample110` is currently the strongest known 4P line among our own code.
  - It contains the true-one-ply / high-production coord-followup direction that actually moved win/loss results more than the earlier A/B/C selector-style changes.
  - It should be treated as the reference behavior for future 4P work unless a new branch clearly beats it on fresh seed tests.
- 2P:
  - `sample8` itself is public-code based and remains very strong in the real/public environment.
  - The 2P behavior inside `sample110` is also a strong practical baseline.
  - Losses in 2P often come from same-family/public-code opponents that have small improvements over `sample8`, especially behavior changes around turn 50/100, plus genuinely high-level opponents.
  - Do not replace the 2P baseline just because a small local public-code block looks bad; new 2P changes must avoid breaking the strong sample8/sample110 feel.
- Submission-building caution:
  - Avoid dynamic folder wrappers when possible. They repeatedly changed behavior through `orbit_lite` import/cache effects.
  - Prefer one loaded package/root `main.py` with explicit 2P/4P branches.

Current known weakness:
- 4P still has a major top-player problem: even `sample110` can be erased around turn 40 before forming a stable base.
- Future 4P improvements should prioritize surviving and stabilizing against top-tier early pressure, not only improving average score diff against public/local bots.

## Competition Overview

Orbit Wars is a Kaggle real-time strategy environment for 2 or 4 players.

- Players start with one home planet and launch fleets to capture neutral or enemy planets.
- The map is a 100x100 continuous 2D board with a destructive sun at the center.
- Planets can be static or orbiting; orbiting planets require future-position prediction.
- Owned planets produce ships each turn; higher-production planets are strategically critical.
- Fleets travel in straight lines, can hit planets, leave the map, or be destroyed by crossing the sun.
- The game lasts up to 500 turns. The winner is determined by total ships on planets plus ships in fleets at the end, or by elimination.
- 2P and 4P are strategically different:
  - 2P is closer to a direct duel where timing, counter-capture, and midgame response matter.
  - 4P rewards avoiding wasteful non-leader fights, building a stable outer/side production base, and surviving early multi-player pressure.

## Historical Submission Context

Historical public leaderboard references from earlier submissions:

- `sample7_4p_sample8_2p_submit_v3.zip`: `1204.1`
- `sample7.zip`: `1121.7`
- `sample8.zip`: `1117.2`
- `hairate5.zip`: about `1109`
- `hairate7.zip`: about `1119`
- `V10.2.zip`: about `1189`
- older `hairate2.zip` had large public variance, including one high score around `1296`

Current safe local baseline:

- `sample11_s8_baseline_2p_s7_4p`
- 2P uses `sample8`
- 4P uses `sample7`
- This is the clean baseline for new experiments.

Current 4P research candidate:

- `sample44_4p_script_portfolio_planner`
- Based on `sample36_4p_third_mode_lane_anchor`.
- Adds a public-top-bot-inspired strategy portfolio: `s7_stable`, `s8_burst`, `lane_anchor`, `winner_outer_domain`, and `enemy_domain_block`.
- The goal is not survival; it is to create first-place routes with thick outer domains when the initial board supports that plan.
- Latest 15-seed pool vs `sample7 + sample8 + hairate5`: `11W / 0D / 4L`, avg place `1.27`.
- Reference: `sample36` was `10W / 1D / 4L` on the same pool.
- Remaining hard losses: `2000001`, `5421622`, `9874600002`, `12000002`.

Useful `sample44` check:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent sample44_4p_script_portfolio_planner\main.py --opponent sample7\main.py --opponent sample8\main.py --opponent bots\hairate5.py --seed-list 2000000,2000001,2000002,5421622,5421623,5421624,56000000,56000001,56000002,9874600000,9874600001,9874600002,12000000,12000001,12000002 --workers 5
```

## Important Strategic Conclusions

### 1. Response Search Did Not Fit `sample8`

Branches:

- `sample9_s8_response_2p_s7_4p`
- `sample10_s8_targeted_response_2p_s7_4p`

Result:

- Broad enemy-response re-scoring hurt `sample8`.
- It reduced win rate in 2P same-seed tests.
- The likely reason is that `sample8` already has a strong, sharp launch policy; broad down-penalties disrupt its winning timing.

Decision:

- Freeze response-search-on-sample8 experiments for now.
- Do not keep adding wider 2-ply penalties to `sample8`.

### 2. Small Score Bonuses Are Usually Not Enough

Branches:

- `sample12_2p_opening_route_commit_s7_4p`
- `sample13_2p_phase_controller_s7_4p`
- `sample14_2p_counter_capture_candidates_s7_4p`
- `sample15_s8_2p_s7_4p_anchor`

Result:

- These branches mostly behaved like the baseline in short local gates.
- They did not clearly improve win rate.

Decision:

- Keep as references.
- Do not promote.
- Future work should change action generation / planning state more deeply, not just add target bonuses.

### 3. `sample6_17` Is Not a Direct Replacement

Reference file:

- `sample6_17.gz`

What it contains:

- A self-contained `main.py` plus `orbit_lite`
- Dynamic ROI / strength-aware configuration
- Late-game suppression
- No `sample8`-style multi-size tiers

Observed:

- Direct 2P test vs `sample8` was weak.
- It is useful as a source of ideas, not as a full replacement.

Decision:

- Do not replace `sample8` or `sample7` with `sample6_17`.
- Consider cherry-picking dynamic phase ideas later.

### 4. 4P Needs Route / Anchor / Cluster Thinking

Relevant branches:

- `hairate32_4p_home_cluster_planner.py`
- `hairate38_safe_strategy_selector.py`
- `hairate41_4p_opening_route_planner.py`
- `hairate42_4p_anchor_planner.py`
- `sample18_planned_anchor_route`

Observed from replays:

- Strong 4P winners build a thick outer or side cluster.
- They do not scatter thin planets everywhere.
- They create an anchor, build ships there, then expand to large neutral/enemy areas.

Important replay pattern:

- Winner first secures a nearby production cluster.
- Then builds a thick source.
- Then expands through central or outer high-value planets.
- Losing bots often take planets but fail to hold a connected, thick production zone.

Decision:

- The right long-term direction is still 4P route/anchor/cluster planning.
- However, the first `sample18` implementation did not clearly improve local results.
- It should remain a research branch, not a submission branch.

### 5. 2P Discovery: `sample7` Can Beat `sample8` on Some Blocks

Important local comparison:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent sample7\main.py --opponent sample8\main.py --games 10 --both-seats --workers 10 --seed-start 5522554122
```

Result:

- `sample7` vs `sample8`: `11/20`, `55.0%`

Reverse / related:

- `sample8` vs `sample7`: `9/20`, `45.0%`
- `sample7` vs `sample7`: roughly symmetric, `5/10` in short same-bot check

Decision:

- For 2P, do not assume `sample8` is always stronger.
- `sample7` has a 2P style that can beat `sample8` on this block.
- A future 2P selector or unified agent should consider using `sample7`-style behavior.

### 6. Folder-Based Selector Can Change Behavior

Observed with:

- `sample20_2p_s7_anchor_route_s7_4p`

Direct engine result:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent sample20_2p_s7_anchor_route_s7_4p\sample8\main.py --opponent sample8\main.py --games 10 --both-seats --workers 10 --seed-start 5522554122
```

Result:

- `11/20`, `55.0%`

Selector wrapper result:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent sample20_2p_s7_anchor_route_s7_4p\main.py --opponent sample8\main.py --games 10 --both-seats --workers 10 --seed-start 5522554122
```

Result:

- `10/20`, `50.0%`

Interpretation:

- The folder-based `main.py` selector is risky.
- Loading two different `orbit_lite` packages dynamically can subtly change behavior.
- For a serious submission, prefer a single integrated `main.py` instead of runtime folder switching.

Decision:

- Do not rely on folder-based selector for final high-score submission unless fully validated.
- If combining `sample7` and `sample8`, build a single-file or single-package integrated submission.

## Branch Status

### Safe / Useful

- `sample11_s8_baseline_2p_s7_4p`
  - Current clean baseline.
  - 2P=`sample8`, 4P=`sample7`.

- `sample7`
  - Strong 4P public component.
  - Also surprisingly useful in 2P on some seed blocks.

- `sample8`
  - Strong 2P public component.
  - But not always stronger than `sample7` locally.

### Research Only

- `sample12_2p_opening_route_commit_s7_4p`
- `sample13_2p_phase_controller_s7_4p`
- `sample14_2p_counter_capture_candidates_s7_4p`
- `sample15_s8_2p_s7_4p_anchor`
- `sample17_s8_dynamic_roi_late_2p_s7_4p`
- `sample18_planned_anchor_route`
- `sample20_2p_s7_anchor_route_s7_4p`

### Frozen / Do Not Promote

- `sample9_s8_response_2p_s7_4p`
- `sample10_s8_targeted_response_2p_s7_4p`
- `sample19_2p_anchor_route_s7_4p`

Reasons:

- Response search hurt `sample8`.
- 2P anchor route on `sample8` hurt vs `sample7`.
- Anchor route as implemented did not produce the desired 55% target.

## Recommended Next Direction

### Near-Term

Build a robust 2P experiment around `sample7` behavior:

- Baseline: `sample7` 2P logic
- Goal: keep `55%` vs `sample8`
- Improve: raise performance vs `sample7` itself
- Avoid: broad score penalties and broad response search

Most promising next 2P idea:

- Counter-capture / contested-neutral planner
- Predict the enemy's first high-value neutral capture
- Add explicit candidates for:
  - arrive just after enemy capture
  - defend own newly captured anchor
  - abandon bad race and choose alternate route

### Medium-Term

Build a single integrated submission instead of folder selector:

- One package / one `orbit_lite`
- Explicit 2P and 4P config/logic branches inside the same loaded module
- Avoid dynamic clearing/reloading of `orbit_lite`

### 4P Direction

Continue route/anchor work, but with stronger diagnostics:

- Record selected anchor id
- Record whether anchor target was actually launched toward
- Record anchor ships over time
- Compare replay behavior before judging win rate

4P improvements should target:

- building thick production clusters
- avoiding thin scattered planets
- timing expansion from anchor to large planets
- detecting enemy safe outer cluster only after self-plan is stable

## Current Practical Recommendation

For immediate public submission:

- Use the latest proven public candidate, not `sample18/19/20`.
- `sample7_4p_sample8_2p_submit_v3.zip` remains the most proven mixed submission from this batch.

For development:

- Next branch should not be another `sample8 + response search`.
- Next branch should be either:
  - `sample21_2p_s7_counter_capture`
  - or an integrated single-main version of `sample7/sample8` to avoid selector artifacts.

## 2026-06-19: 4P Single-Package Selector + Oracle Dataset Direction

Goal:

- Improve 4P first-place rate against a Kaggle-like local pool.
- Stop relying on folder-based selectors.
- Use initial board features to choose between strong `sample7` and `sample8` behaviors.
- Move toward an oracle selector, then only add a third strategy if data shows both `sample7` and `sample8` fail on the same board type.

### New Tools

- `dump_initial_boards.py`
  - Dumps raw initial boards, features, and an HTML preview.
  - Useful when the user wants to inspect a seed visually.

- `cluster_initial_boards.py`
  - Fast large-scale initial board clustering.
  - A 1000-seed run around `90000000` found three broad board families:
    - `cheap-dense`
    - `near-rich`
    - `balanced`

- `analyze_4p_losses.py`
  - Parses evaluate logs or explicit seed lists.
  - Writes loss-focused initial features, selected mode, board bucket, and HTML preview.
  - Treats `diff=0` as draw.

- `build_4p_oracle_dataset.py`
  - Evaluates candidate bots on identical seed blocks.
  - Writes `oracle_results.csv`, `oracle_features.csv`, `oracle_summary.md`, and `oracle_rules.json`.
  - Now supports partial/interrupted runs:
    - completed logs are skipped on resume,
    - `--collect-only` rebuilds outputs from existing logs,
    - `Ctrl+C` still writes partial outputs from completed logs.

### Branches

- `sample30_4p_cluster_selector`
  - Dynamic folder selector.
  - Failed as a final direction because loading sub-agents dynamically changed behavior.
  - Do not promote.

- `sample31_4p_single_engine_cluster_modes`
  - First attempt at single-engine config switching.
  - Reproduced sample8 wins on `9874600005/9874600006`.
  - Too broad/weak overall; moved on.

- `sample32_singlefile_s7_s8_selector`
  - Important promising branch.
  - Uses one `orbit_lite` package, with `sample7`-style stable config and `sample8`-style burst config.
  - No dynamic folder import.
  - Observed local pool results:
    - `56000000`: `6W / 1D / 3L`
    - `12000000`: `4W / 6L`
    - `9874600000`: `5W / 5L`
    - `5421622`: `2W / 2D / 6L`
    - `2000000`: `6W / 4L`

- `sample33_singlefile_contested_selector`
  - Added reserve-based contested mode.
  - Did not improve `5421622`; same 2-win result and worse-looking losses.
  - Freeze. Do not retry reserve-only contested mode.

- `sample34_singlefile_s7_s8_selector_tuned`
  - Tightened selector to avoid sending some `sample7`-winning seeds to `sample8`.
  - Needs full evaluation.

- `sample35_singlefile_oracle_selector`
  - Current oracle-ready branch.
  - Same single-package approach as `sample34`.
  - Reads optional `oracle_rules.json`.
  - Without `oracle_rules.json`, falls back to tuned selector behavior.
  - Smoke passed on `9874600005/9874600006`: `2W/2`.

### Key Results and Interpretation

- `sample8` can be very strong on specific 4P board types, especially fast-snowball/outer-high boards.
- `sample7` remains safer on many blocks and should be the fallback when selector confidence is low.
- `5421622` is a hard block:
  - direct `sample7`: `3W / 2D / 5L`
  - direct `sample8`: `1W / 2D / 7L`
  - `sample32`: `2W / 2D / 6L`
  - reserve contested mode did not help.
- `9874600000` improved under `sample32`:
  - `5W / 5L`
  - sample8-winning seeds `9874600005/9874600006` were preserved.

### Current Decision

- Continue with `sample32/sample34/sample35`, not `sample33`.
- Build oracle dataset before inventing another third mode.
- Convert only high-confidence board buckets into `sample35/oracle_rules.json`.
- If both `sample7` and `sample8` lose the same bucket repeatedly, then implement `sample36_4p_third_mode`.

### Current Working Commands

Build/resume oracle dataset:

```powershell
C:\tmp\ow\Scripts\python.exe build_4p_oracle_dataset.py --blocks 2000000,5421622,56000000,9874600000,12000000 --games 10 --workers 5
```

Collect existing partial logs:

```powershell
C:\tmp\ow\Scripts\python.exe build_4p_oracle_dataset.py --out-dir <existing_research_run_dir> --blocks 2000000,5421622,56000000,9874600000,12000000 --games 10 --workers 5 --collect-only
```

Analyze losses from a pasted/saved evaluate log:

```powershell
C:\tmp\ow\Scripts\python.exe analyze_4p_losses.py --eval-log <evaluate_output.txt> --agent sample32_singlefile_s7_s8_selector\main.py
```

Evaluate current oracle-ready branch:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent sample35_singlefile_oracle_selector\main.py --opponent sample7\main.py --opponent sample8\main.py --opponent bots\hairate5.py --games 10 --workers 10 --seed-start 9874600000
```

## 2026-06-17: `sample21_2p_s7_counter_capture_s7_4p`

Goal:

- Restart 2P from `sample7` instead of `sample8`.
- Add only candidate-generation style contested neutral / counter-capture ideas.
- Probe a possible weakness of orbit_lite-like bots that make similar opening moves for the first ~20 turns.

Implementation:

- Copied `sample7`.
- Added 2P-only early contested neutral target augmentation.
- Added a small 2P-only early bonus against thin enemy productive anchors.
- 4P remains `sample7` behavior.

Short results:

- Syntax: passed.
- 4P smoke vs `hairate5`, seed `12000000`, 1 game: crash 0, win 1/1.
- 2P vs `sample8`, seeds `5522554122-5522554125`, both seats: `2/8`.
- Baseline `sample7` vs `sample8` on the same seeds: `1/8`.
- 2P vs `sample7`, same seeds, both seats: `3/8`.
- Baseline `sample7` vs `sample7` on same seeds: `4/8`.

Conclusion:

- The idea created at least one extra win vs `sample8`, so the orbit_lite mirror weakness is real enough to study.
- It hurts `sample7` mirror performance, so it is not a promotion candidate.
- Next attempt should not be another small bonus. Use a stronger stateful opening policy: detect the first contested neutral, then choose one of `race`, `delayed retake`, or `alternate route`, instead of merely adding targets to the normal shortlist.

## 2026-06-19: `sample45_4p_domain_control_planner`

Goal:

- Build a 4P bot that plays for first by creating its own outer production factory.
- Keep `sample44`'s strong script portfolio, but add a stateful `domain_factory` layer that tracks anchor/core/support/expansion planets.
- Avoid turning this into a survival bot: the factory layer should support a winning outer route, not stop early expansion.

Implementation:

- Copied `sample44_4p_script_portfolio_planner`.
- Added `domain_factory` config fields and state in `ProducerLiteMemory`.
- Added `_make_domain_factory_plan`, `_append_domain_factory_targets`, and `_apply_domain_factory_adjustment`.
- Aligned `domain_factory` with `winner_outer_domain`: winner path selects the lane, then factory logic reinforces that same lane after the early claim phase.
- `winner_outer_domain` now enables domain factory only conservatively after step 62, with very light drain/off-lane penalties.

Results:

- Syntax: passed.
- 6-seed smoke vs `sample7 + sample8 + hairate5`: `3W / 0D / 3L`.
- 15-seed fixed pool vs `sample7 + sample8 + hairate5`: `11W / 0D / 4L`, avg place `1.27`, avg diff `+2857.80`.
- This matches `sample44`'s win count (`11W / 0D / 4L`) while adding a safer outer-factory layer.

Failed sub-idea:

- Forcing chain-cluster losses (`2000001`, `5421622`) into `domain_factory` was tested.
- It slightly improved `2000001`'s loss margin but hurt `5421622`, so the condition was reverted.

Decision:

- `sample45` is a valid research/submission candidate, but not a clear promotion over `sample44` yet.
- Next real improvement needs a dedicated strategy for the hard third-needed seeds: `2000001`, `5421622`, `9874600002`, `12000002`.
- The most promising next direction is not stronger holding, but an early high-prod anchor race/retake planner that can win contested heavy anchors without stalling expansion.
## 2026-06-19 sample50_4p_halite_pw_adapted

- Purpose: test Planet Wars / Halite-style 4P shape control on top of `sample48_4p_sample_opening_domain_factory`.
- External-code policy: no third-party source code copied. Ideas were reimplemented locally:
  - prefer outer planets in 4P,
  - penalize center fights,
  - reduce attacks into enemy pressure.
- Initial version applied the shape adjustment from step 0 and was too disruptive:
  - 6-seed smoke: 1W / 0D / 5L.
  - It broke `125693095`, which `sample48` won.
- Revised version delays shape adjustment until `step >= 50` and weakens the bonus/penalty:
  - 6-seed smoke: 2W / 0D / 4L.
  - This matches `sample48` on the same quick pool, so it is not an upgrade yet.
- Interpretation:
  - The Halite/Planet Wars idea is plausible, but a broad center/outer score adjustment is still too blunt.
  - Next high-upside direction should be a strategy-level opening/domain script, not a global target-shape correction.

## 2026-06-19 sample51_4p_outer_lane_sequence_planner

- Purpose: move beyond global center/outer tweaks and implement a stateful action-sequence planner.
- Parent: `sample48_4p_sample_opening_domain_factory`.
- Main idea:
  - choose one outer lane from the opening,
  - claim cheap/near lane planets,
  - consolidate anchor/support planets,
  - project from the matured lane into expansion targets.
- Implementation:
  - added `outer_lane_plan` to memory,
  - added `_make_outer_lane_sequence_plan`,
  - added `_append_outer_lane_sequence_targets`,
  - added `_apply_outer_lane_sequence_adjustment`.
- Important iteration:
  - applying lane bonuses from step 0 was too destructive: 6-seed smoke fell to `1W / 0D / 5L`.
  - keeping the selected plan from step 0 but delaying score/target intervention until step 34 restored the quick-pool win count.
- Results:
  - Syntax: passed.
  - 6-seed quick pool vs `sample7 + sample8 + hairate5`: `2W / 0D / 4L`, matching `sample48` win count.
  - 15-seed fixed pool vs `sample7 + sample8 + hairate5`: `10W / 0D / 5L`, matching `sample48` win count.
  - Avg diff on 15-seed pool: `+1671.67` vs `sample48` `+1590.93`.
- Budget-reservation update:
  - Added a dedicated lane greedy pass before normal greedy selection.
  - Lane modes reserve `1` wave during claim/anchor and `2` waves during project.
  - The reserved wave debits `source_budget`, and taken lane targets are removed from the normal greedy pass.
  - Applying this to every mode broke `125693095`, so budget reservation is now limited to `winner_outer_domain`, `domain_factory`, and `lane_anchor` modes.
  - 6-seed quick pool after limiting budget reservation: `2W / 0D / 4L`.
  - 15-seed fixed pool after limiting budget reservation: `10W / 0D / 5L`, avg diff `+1667.67`.
- Interpretation:
  - This is not a promotion over `sample44/sample45`.
  - It is a working implementation of the intended outer-lane sequence concept and lane budget reservation.
  - The current sequence still fails the hard seeds (`2000001`, `5421622`, `9874600002`, `12000002`), so the missing piece is probably not budget ownership alone.
  - Next high-upside step: build an explicit opening action queue for the first 2-3 captures in selected lane modes, using capture floor sizing, rather than letting scorer/greedy infer the sequence from bonuses.

## 2026-06-19 sample52_4p_outer_lane_rollout_planner

- Purpose: implement a higher-level forward-planning idea instead of another broad score tweak.
- Parent: `sample51_4p_outer_lane_sequence_planner`.
- External algorithm inspiration, reimplemented locally:
  - Rolling Horizon / forward-planning: compare short action sequences before choosing the first action.
  - Planet Wars-style forward evaluation: value future ownership/production, not only immediate capture value.
  - RTS portfolio thinking: choose a strategic script/route, then let tactical scorer handle launch details.
- Implementation:
  - Added `outer_lane_rollout_horizon`, `outer_lane_rollout_candidates`, `outer_lane_queue_width`.
  - `_make_outer_lane_sequence_plan` now generates multiple outer lane candidates instead of choosing the first best heuristic lane.
  - Added `_outer_lane_rollout_score`, a cheap deterministic rollout that estimates:
    - capture order,
    - approximate travel time,
    - enemy fastest contest ETA,
    - production compounding over ~60 turns,
    - whether the lane becomes thick enough to be a factory.
  - The selected plan stores a concrete `queue` of target planets.
  - Target shortlist and reserved-budget pass prioritize this queue.
  - Initial version reserved budget too early and broke `1661282750`; budget reservation is now delayed until `step >= 55`.
- Results:
  - Syntax: passed.
  - 6-seed quick pool: `2W / 0D / 4L`.
  - 15-seed fixed pool: `10W / 0D / 5L`, avg diff `+1392.00`.
- Interpretation:
  - This is a real architectural step: it now compares candidate action sequences with a forward-looking heuristic.
  - It is not yet a promotion candidate because win count did not improve over `sample48/sample51`.
  - The next likely bottleneck is that rollout picks a queue, but actual capture sizing/timing is still delegated to normal scorer. To finish the idea, the queue needs explicit capture-floor launch actions for the first 1-2 captures, not just target/score/budget priority.

## 2026-06-19 sample53_4p_sector_factory_rollout

- Purpose: address replay losses where `sample51/sample52` selected a contested heavy anchor before it had safe support.
- Parent: `sample52_4p_outer_lane_rollout_planner`.
- Main idea:
  - make lane selection support-first,
  - reward safe cheap/support planets before the anchor,
  - penalize heavy-anchor rushes when `support_before_anchor` is thin,
  - penalize lanes whose anchor overlaps an enemy initial sphere,
  - keep the rollout queue as cheap support -> anchor -> expansion.
- Implementation:
  - Added sector-factory knobs:
    - `sector_min_support_before_anchor`,
    - `sector_support_first_bonus`,
    - `sector_safe_support_bonus`,
    - `sector_anchor_rush_penalty`,
    - `sector_enemy_overlap_penalty`,
    - `sector_support_radius`.
  - `_make_outer_lane_sequence_plan` now computes `support_before_anchor` and scores it above raw anchor size.
  - Anchor is separated from `support` in the plan and introduced from anchor/project phases rather than being treated as early support.
  - `_apply_outer_lane_sequence_adjustment` and `_outer_lane_candidate_mask` now handle the anchor as its own lane target.
- Results:
  - Syntax: passed.
  - Hard loss set (`514234942,951681560,2000001,5421622,9874600002,12000002`): `0W / 0D / 6L`.
    - Same result as `sample52`, so the branch did not newly break these losses, but also did not solve them.
  - 15-seed fixed pool: `10W / 0D / 5L`, avg diff `+1425.07`.
- Interpretation:
  - This is not a promotion candidate yet.
  - It preserves fixed-pool win count while changing the lane objective in the intended direction.
  - The remaining loss type is probably not solved by scoring/target priority alone.
  - Next high-upside step: implement either explicit capture-floor launch actions for the first support queue targets, or an enemy safe-factory denial layer that claims boundary neutral planets before a rival completes their outer factory.

## 2026-06-19 sample54_4p_queue_executor

- Purpose: test the next step after `sample53`: make the first 1-2 support queue targets explicit capture-floor sized candidates instead of relying only on target bonuses.
- Parent: `sample53_4p_sector_factory_rollout`.
- Main idea:
  - Keep support-first sector selection.
  - Add `_outer_lane_queue_executor_candidates`.
  - For the first live queue targets, estimate ETA, gather `capture_floor` at that ETA, and create dedicated launch candidates sized to actually capture.
  - Feed those candidates into the same source-budget-aware greedy pass.
- Results:
  - Syntax: passed.
  - Hard loss set (`514234942,951681560,2000001,5421622,9874600002,12000002`): `0W / 0D / 6L`.
    - Some losses became slightly less explosive (`2000001`, `9874600002` best opponent score decreased), but no win conversion.
  - 15-seed fixed pool: `10W / 0D / 5L`, avg diff `+1106.07`.
- Interpretation:
  - Explicit queue capture sizing did not improve win count.
  - This suggests the hard losses are less about first-target under-sizing and more about choosing a sector that still allows a rival safe factory to complete.
  - Next major branch should focus on conservative enemy safe-factory denial: claim boundary neutral planets near the rival's lane, not direct attacks on owned planets.
## 2026-06-21 2P selector line: sample70-sample73

- Context:
  - User moved focus from 4P to 2P.
  - Current idea: use `sample8` by default because it is strong in live-like results, but route known losing opening shapes to `sample7`.
  - User preference: judge by win/loss first; large score margin is secondary.
- `sample70_2p_s8_s7_heavy_selector`:
  - Parent: `sample8` + `sample7` selector, with 4P delegated to `sample69_4p_orbit_lite_response_gate`.
  - Rule: default `sample8`, switch to `sample7` for heavy/high-value neutral opening lanes.
  - Result on user-provided sample8-vs-sample7 losing seeds improved from roughly `3W/9L` to `6W/6L`.
  - Result vs `bots/hairate2.py`, 40 seeds both seats from `95446545645`: `56W/24L`, same win count as `sample8` but lower average diff.
- `sample71_2p_s8_s7_loss_selector`:
  - Parent: `sample70`.
  - Added broader `sample7` gates for close prod5 route, low-mass high-production trap, and mixed heavy value lanes.
  - Result vs `bots/hairate2.py`, same 40 seeds both seats: `62W/18L`.
  - This is the current best 2P selector candidate from this line.
- `hairate5` check:
  - `bots/hairate5.py` vs `bots/hairate2.py`, same 40 seeds both seats: `42W/38L`; not good as a default.
  - However, it wins several `sample71` loss cases:
    - `95446545654` both seats,
    - `95446545667` both seats,
    - `95446545683` both seats.
  - Direct targeted check on these three seeds both seats: `hairate5` = `6W/0L`.
- `sample72_2p_s8_s7_h5_selector`:
  - Parent: `sample71`.
  - Added a third mode, `h5`, using a copied `hairate5` plus copied `orbit_lite`.
  - Added narrow h5 rescue gates for low-mass prod3 cluster, far heavy prod5 anchor, and compact double prod4.
  - Problem: wrapper-mode `h5` did not reproduce direct `hairate5`.
  - Targeted check on `95446545654,95446545667,95446545683` both seats:
    - direct `hairate5`: `6W/0L`;
    - `sample72` h5-enabled wrapper: `0W/6L`.
  - Likely cause: `orbit_lite` package/module-state collision when `sample8`, `sample7`, copied `hairate5`, and opponent `hairate2` coexist in the same evaluation process.
  - Safety decision: keep h5 rescue code in `sample72`, but disable it with `_ENABLE_H5_RESCUE = False`.
  - With h5 disabled, targeted check matches the `sample71` behavior on those seeds (`1W/5L`), so it is safe but not an improvement.
- `sample73_2p_s8_s7_h5_external_selector`:
  - Purpose: local-only diagnostic to test whether using the original `bots/hairate5.py` path reproduces direct `hairate5` better than copied/subfolder h5.
  - Not submission-ready because it reaches outside its own folder.
  - Targeted check on `95446545654,95446545667,95446545683` both seats: `0W/6L`.
  - This means the failure is not just the copied folder layout; selector-wrapper execution itself changes `hairate5` behavior.
- `sample74_2p_s8_s7_h5_isolated_selector`:
  - Renamed copied `hairate5/orbit_lite` to `h5_orbit_lite` and rewrote imports to isolate h5 from sample7/sample8/opponent `orbit_lite`.
  - Direct isolated h5 (`sample74/.../hairate5/main.py`) still wins targeted three seeds both seats: `6W/0L`.
  - Selector wrapper with h5 rescue enabled still loses targeted three seeds both seats: `0W/6L`.
  - Therefore package-name collision alone is not the issue.
- `sample75_h5_min_wrapper`:
  - Minimal wrapper that only loads isolated h5 via `importlib` and calls it.
  - Targeted result: `0W/6L`.
  - Preloading h5 at module import instead of first action did not fix it.
  - This suggests h5 is sensitive to being called through an imported submodule/function wrapper.
- `sample76_2p_s8_s7_h5_inlined_selector` / `sample78_h5_inlined_forced` / `sample79_2p_s8_s7_h5_patch_agent_selector`:
  - Tried inlining h5 into root `main.py` and either:
    - aliasing original h5 `agent` as `_h5_agent`,
    - forcing all calls to `_h5_agent`,
    - or patching the original h5 `agent` body with selector logic.
  - Targeted h5 rescue still produced `0W/6L`.
  - A related step-reset bug was found: some observations lack `step`; treating missing step as 0 can reset mode every turn. Fixed in sample79, but it did not solve h5 reproduction.
- `sample77_h5_root_direct` and `sample80_h5_preread_only`:
  - `sample77`: root-level isolated h5 only, no selector: targeted `6W/0L`.
  - `sample80`: same h5 with harmless pre-read of `planets/fleets` before original h5 logic: targeted `6W/0L`.
  - So reading observations is not the issue; the failing pattern is wrapping/replacing the h5 `agent` execution path.
- Decision:
  - Do not integrate `hairate2` as a mode; it is primarily the benchmark opponent here.
  - `hairate5` has useful niche behavior, but only after module isolation is solved.
  - Current practical 2P candidate remains `sample71_2p_s8_s7_loss_selector`.

## 2026-06-21 sample81_4p_lane_anchor_promotions_from69

- Context:
  - User reviewed current 4P mode outcomes and found:
    - `lane_anchor` looked most stable on the small sample,
    - `s7_stable` was still the main mode but also contained many losing seeds,
    - `s8_burst` appeared only on the losing side in the current sample.
  - Request: start from `sample69` and promote some `s7_stable` loss shapes into `lane_anchor`.
- Parent:
  - `sample69_4p_orbit_lite_response_gate`
- Change:
  - Added two extra `lane_anchor` promotions inside `_choose_4p_mode(...)`.
  - Rule 1:
    - moderate chain, real outer anchor, real support density, midrange enemy distance,
    - intended for “outer factory exists but old lane gate was too strict” boards.
  - Rule 2:
    - sparse `~20` planet boards with a smaller but still meaningful outer anchor,
    - intended for light outer-lane boards that were falling back to `s7_stable`.
- Mode-change check on the user-provided win/loss seed set:
  - promoted losses from `s7_stable -> lane_anchor`:
    - `363175246`
    - `306239195`
    - `49433498`
    - `1488820789`
  - already-`lane_anchor` loss stayed `lane_anchor`:
    - `1796441431`
  - non-promoted `s7_stable` losses remained unchanged:
    - `2012339891`, `315400519`, `754928456`, `915403528`, `927652245`,
      `2112295433`, `1846002515`, `1309011149`
  - one prior `s7_stable` win also moved to `lane_anchor`:
    - `1301457313`
  - `s8_burst` loss example stayed `s8_burst`:
    - `1426214042`
- Interpretation:
  - This is a real “more lane-anchor from sample69” branch, not a broad rewrite.
  - It hits several requested `s7_stable` loss boards, but it is not cleanly loss-only:
    one current win seed also gets re-routed.
  - Needs real match evaluation before promotion.

## 2026-06-21 sample82/83/84 forced 4P mode sweep from sample69

- Context:
  - User wanted a clean ablation:
    - disable the normal 4P selector,
    - force one mode only,
    - and compare behavior separately on the current `win seeds` and `loss seeds`.
  - Goal:
    - check whether current failures are mainly `lane_anchor`-type boards,
    - or whether `enemy_domain_block` / `s8_burst` are actually rescuing some cases.
- Agents:
  - `sample82_4p_force_lane_anchor_from69`
  - `sample83_4p_force_enemy_domain_block_from69`
  - `sample84_4p_force_s8_burst_from69`
- Parent:
  - `sample69_4p_orbit_lite_response_gate`
- Change:
  - Patched `_choose_4p_mode(...)` in each branch to immediately return exactly one mode:
    - sample82: `lane_anchor`
    - sample83: `enemy_domain_block`
    - sample84: `s8_burst`
  - No other planner logic was changed.
- Eval command shape:
  - opponents fixed to:
    - `sample7/main.py`
    - `sample8/main.py`
    - `bots/hairate5.py`
  - output filtered with:
    - `Select-String 'Wins|Losses|Average placement|Average score diff|seed='`
- Win-seed group (`1284299523,1301457313,230622801,1218569221,718279990,638588263,676586300,27311426,736262817,1964410425`):
  - `sample69`: `4W/6L`, avg diff `-1716.90`, avg placement `1.70`
  - `sample82 force_lane_anchor`: `4W/6L`, avg diff `-2006.20`, avg placement `1.70`
  - `sample83 force_enemy_domain_block`: `2W/8L`, avg diff `-2507.40`, avg placement `1.90`
  - `sample84 force_s8_burst`: `2W/8L`, avg diff `-1662.50`, avg placement `1.90`
- Loss-seed group (`2012339891,363175246,315400519,1796441431,754928456,915403528,927652245,1426214042,306239195,2112295433,49433498,1488820789,1846002515,1309011149`):
  - `sample69`: `6W/8L`, avg diff `-1672.00`, avg placement `1.57`
  - `sample82 force_lane_anchor`: `8W/6L`, avg diff `-810.14`, avg placement `1.43`
  - `sample83 force_enemy_domain_block`: `8W/6L`, avg diff `-822.64`, avg placement `1.43`
  - `sample84 force_s8_burst`: `8W/6L`, avg diff `-163.00`, avg placement `1.50`
- Interpretation:
  - The current selector is clearly doing useful work on the present win-seed group:
    forced `enemy_domain_block` and forced `s8_burst` both collapse there.
  - On the present loss-seed group, all three forced modes improve over baseline `sample69`.
  - The strongest rescue by win/loss count is:
    - `lane_anchor` and `enemy_domain_block`: both `8W/6L`
  - `s8_burst` also recovers to `8W/6L` on the loss group, but still looks dangerous as a global default because it was poor on the win group.
  - This points toward the next real change:
    - keep mixed-mode behavior for the good boards,
    - but expand the routing into `lane_anchor` / `enemy_domain_block` on specific `sample69` loss shapes,
    - rather than replacing the whole selector with any one forced mode.

## 2026-06-21 sample85/sample86 lane-anchor promotion follow-up

- Context:
  - User proposed that the forced-mode sweep implies `lane_anchor` is the only mode that can improve losing seeds without hurting winning seeds.
  - Rechecked current working-tree `sample69` on the same 24 seed set because prior grouped summaries were inconsistent.
- Baseline recheck:
  - `sample69_4p_orbit_lite_response_gate`
  - Seeds:
    - win group: `1284299523,1301457313,230622801,1218569221,718279990,638588263,676586300,27311426,736262817,1964410425`
    - loss group: `2012339891,363175246,315400519,1796441431,754928456,915403528,927652245,1426214042,306239195,2112295433,49433498,1488820789,1846002515,1309011149`
  - Result on all 24 seeds:
    - `12W/12L`
    - avg diff `-810.46`
    - avg placement `1.58`
- `sample85_4p_lane_default_from69`:
  - Parent: `sample69`.
  - Change:
    - `s8_burst` routes fall back to `lane_anchor`.
    - final `s7_stable` fallback changed to `lane_anchor`.
    - narrow existing `winner_outer_domain` / `enemy_domain_block` rules left intact.
  - Mode effect:
    - most `s7_stable` boards become `lane_anchor`.
  - Result:
    - win group: `4W/6L`, avg diff `-2006.20`, avg placement `1.70`
    - loss group: `8W/6L`, avg diff `-810.14`, avg placement `1.43`
    - combined: `12W/12L`
  - Interpretation:
    - broad lane-default did not improve win count versus current `sample69`.
    - It also worsened diff, mainly due to large losses on some boards.
- Seed-level lane comparison:
  - Lane helped:
    - `638588263`
    - `315400519`
    - `1846002515`
  - Lane hurt:
    - `676586300`
    - `927652245`
    - `1309011149`
  - Therefore the right move was targeted promotion, not global default.
- `sample86_4p_targeted_lane_rescue_from69`:
  - Parent: `sample69`.
  - Change:
    - Added three narrow `_choose_4p_mode(...)` rules before the older loss gates.
    - Only these seeds changed mode in the 24-seed probe:
      - `638588263: s7_stable -> lane_anchor`
      - `315400519: s7_stable -> lane_anchor`
      - `1846002515: s7_stable -> lane_anchor`
    - The known lane-regression seeds stayed on their original route.
  - Result on all 24 seeds:
    - `15W/9L`
    - avg diff `-282.58`
    - avg placement `1.42`
  - Interpretation:
    - This is the strongest current 4P branch in this local targeted set.
    - It supports the general claim that `lane_anchor` is useful, but only as a targeted rescue, not as a global default.
    - Next check should be broader fixed/random seeds to see whether the three rules overfit.
- Overfit/hit-rate check:
  - Checked 100 fresh seeds starting at `700000000` without running full matches.
  - Compared initial `_choose_4p_mode(...)` between `sample69` and `sample86`.
  - `sample69` modes:
    - `lane_anchor`: 7
    - `s7_stable`: 79
    - `s8_burst`: 13
    - `winner_outer_domain`: 1
  - `sample86` modes:
    - identical counts.
  - Changed seeds:
    - `0/100`
  - Hit rate:
    - `0.0%`
  - Interpretation:
    - The three added lane rescue rules are too narrow for broad generalization.
    - `sample86` is valuable evidence that lane rescues can work, but the current rules are overfit to the discovered seed shapes.
    - Next branch should generalize the causal pattern behind those three rules instead of tightening numeric seed fingerprints.

## 2026-06-21 sample87/sample88 generalized lane-anchor catch-all

- Context:
  - User pointed out that `sample86` had near-zero hit rate because each rescue rule used 10+ simultaneous feature gates.
  - Goal:
    - replace seed-fingerprint rules with broader causal lane-anchor rules,
    - target 10-25% hit rate on fresh 4P seeds,
    - keep the `15W/9L` result on the known 24-seed set if possible.
- `sample87_4p_general_lane_catchall_from69`:
  - Parent: `sample69`.
  - Inserted after existing `winner_outer_domain` rules and before `burst_score`.
  - Rule:
    - `outer_anchor >= 55`
    - `support_density >= 10`
    - `chain >= 42`
    - `enemy_dist >= 45`
  - Hit-rate check on 100 seeds starting at `700000000`:
    - changed `64/100` seeds
    - hit rate `64.0%`
  - Interpretation:
    - Too broad; likely to behave like the unsuccessful broad lane-default experiment.
- Condition scan:
  - Tested several variants against the same 100 seed hit-rate set plus the six known local comparison seeds:
    - lane-helped: `638588263`, `315400519`, `1846002515`
    - lane-hurt: `676586300`, `927652245`, `1309011149`
  - Best practical scan result:
    - `cF_AorB`
    - hit rate `18.0%`
    - catches all three lane-helped seeds
    - excludes the three known lane-hurt seeds
- `sample88_4p_general_lane_two_shape_from69`:
  - Parent: `sample69`.
  - Inserted after existing `winner_outer_domain` rules and before `burst_score`.
  - Rule A: low/high65, cheap-rich outer lane:
    - `outer_anchor >= 58`
    - `support_density >= 11`
    - `chain >= 42`
    - `enemy_dist >= 45`
    - `high65 <= 6`
    - `mid_cheap >= 4`
    - `near_prod >= 7`
  - Rule B: expensive but strong outer anchor/support cluster:
    - `outer_anchor >= 78`
    - `support_density >= 22`
    - `chain >= 46`
    - `enemy_dist >= 58`
    - `high65 >= 8`
    - `near_prod <= 9`
    - `2 <= mid_cheap <= 5`
- Hit-rate check:
  - 100 seeds starting at `700000000`
  - `sample69` modes:
    - `lane_anchor`: 7
    - `s7_stable`: 79
    - `s8_burst`: 13
    - `winner_outer_domain`: 1
  - `sample88` modes:
    - `lane_anchor`: 25
    - `s7_stable`: 63
    - `s8_burst`: 11
    - `winner_outer_domain`: 1
  - Changed:
    - `18/100`
  - Hit rate:
    - `18.0%`
  - Changed seeds:
    - `700000000,700000002,700000006,700000007,700000018,700000019,700000024,700000032,700000035,700000040,700000045,700000047,700000059,700000060,700000091,700000092,700000093,700000097`
- Known 24-seed eval:
  - `sample69`: `12W/12L`, avg diff `-810.46`, avg placement `1.58`
  - `sample86`: `15W/9L`, avg diff `-282.58`, avg placement `1.42`
  - `sample88`: `15W/9L`, avg diff `-332.71`, avg placement `1.42`
  - Interpretation:
    - `sample88` keeps the key win-count improvement while being much less seed-fingerprint-like than `sample86`.
- Fresh hit-seed eval on the 18 changed seeds:
  - `sample69`: `5W/13L`, avg diff `-779.67`, avg placement `1.72`
  - `sample88`: `6W/12L`, avg diff `-648.39`, avg placement `1.67`
  - Interpretation:
    - The generalized lane catch-all is mildly positive on fresh hit seeds.
    - It is not a huge jump, but it is directionally correct and far more general than `sample86`.
  - Current judgment:
  - `sample88` is the best current lane-selector branch:
    - same known-set win count as `sample86`,
    - 18% fresh hit rate,
    - positive result on the fresh hit subset.
  - Next check should be a broader random 40/80 full evaluation versus `sample69`.

## 2026-06-21 sample89/sample90/sample91 early orbit response gate

- Context:
  - `sample88` generalized lane selector did not beat `sample69` on the user's random100:
    - `sample69`: `40W/2D/58L`, avg diff `-448.84`, avg placement `1.59`
    - `sample88`: `39W/1D/60L`, avg diff `-867.45`, avg placement `1.63`
  - Decision:
    - stop broad selector promotion for now,
    - test improvement A: enable the existing `orbit_response_gate` earlier so 4P modes can avoid/discount captures that will be retaken soon.
- Parent:
  - `sample69_4p_orbit_lite_response_gate`
- Shared implementation:
  - Added config fields:
    - `orbit_response_ramp_end`
    - `orbit_response_ramp_min`
  - `_apply_orbit_response_gate(...)` now scales `(bonus - penalty)` by a step-based ramp:
    - weak at `orbit_response_start`,
    - full strength at `orbit_response_ramp_end`.
  - Selector logic is unchanged from `sample69`.
- Branches:
  - `sample89_4p_early_response_gate_from69`
    - `orbit_response_start=25`
    - `orbit_response_ramp_end=80`
    - `orbit_response_ramp_min=0.35`
  - `sample90_4p_response_gate_step15_from69`
    - `orbit_response_start=15`
    - `orbit_response_ramp_end=80`
    - `orbit_response_ramp_min=0.35`
  - `sample91_4p_response_gate_step25_stronger_from69`
    - `orbit_response_start=25`
    - `orbit_response_ramp_end=80`
    - `orbit_response_ramp_min=0.60`
- Known 24-seed eval:
  - `sample69`: `12W/12L`, avg diff `-810.46`, avg placement `1.58`
  - `sample89`: `12W/12L`, avg diff `-804.50`, avg placement `1.58`
  - `sample90`: `13W/11L`, avg diff `-611.21`, avg placement `1.54`
  - `sample91`: `12W/12L`, avg diff `-804.50`, avg placement `1.58`
  - Interpretation:
    - step 15 weak ramp is the only promising version on the targeted set.
- User random100 eval on the same `random100_seedlist.txt` used for `sample69/sample88`:
  - `sample90`: `43W/2D/55L`, avg diff `-343.90`, avg placement `1.56`
  - Baseline from same seed list:
    - `sample69`: `40W/2D/58L`, avg diff `-448.84`, avg placement `1.59`
  - Delta:
    - `+3W`
    - `-3L`
    - avg diff `+104.94`
    - avg placement `+0.03` better
  - Current judgment:
  - `sample90` is the strongest current 4P branch over `sample69`.
  - This supports improvement A: earlier response gate is useful when started very early but ramped gently.
  - Next reasonable checks:
    - another independent random100,
    - fixed15/known trouble seeds,
    - then consider improvement C: include in-flight fleet ships in leader power.

## 2026-06-21 sample92 leader-weighted competitive score

- Context:
  - User requested improvement B:
    - change `competitive_score = my_net_delta - sum(opponent_net_delta)`
    - to weight the current leader more heavily and nonleaders less heavily in 4P.
  - Starting point:
    - `sample90_4p_response_gate_step15_from69`
- Branch:
  - `sample92_4p_leader_weighted_score_from90`
- Implementation:
  - `orbit_lite/planner_core.py`
    - Added optional `leader_weights: Tensor | None` to `competitive_score(...)`.
    - Added optional `leader_weights` passthrough to `score_candidates(...)`.
    - Existing behavior is preserved when `leader_weights is None`.
  - `main.py`
    - Added config:
      - `enable_leader_weighted_score_4p`
      - `leader_score_weight=1.30`
      - `nonleader_score_weight=0.85`
    - Added `_leader_score_weights(...)` using existing `_leader_owner_by_power(...)`.
    - Enabled leader weighting in `CONFIG_4P`.
- Syntax:
  - `py_compile` passed for `main.py` and `orbit_lite/planner_core.py`.
- Known 24-seed eval:
  - `sample90`: `13W/11L`, avg diff `-611.21`, avg placement `1.54`
  - `sample92`: `0W/24L`, avg diff `-3041.25`, avg placement `2.08`
- Interpretation:
  - This direct global opponent-net weighting is catastrophically bad.
  - The failure is too large to tune around with small weight changes.
  - Likely issue:
    - globally weighting all opponent net deltas inside `competitive_score` distorts candidate selection too broadly,
    - not just "attack leader instead of weak player."
  - Do not adopt `sample92`.
  - Safer future variant:
    - leave `competitive_score` untouched,
    - add a small target-owner adjustment after candidate scoring:
      - bonus only when target owner is the current leader,
      - penalty only when target owner is a nonleader enemy,
      - do not globally reweight all opponent future deltas.

## 2026-06-22 sample93 leader power includes fleets

- Context:
  - User requested improvement C:
    - current `_leader_owner_by_power` used only planet ships plus production,
    - add in-flight fleet ships,
    - add a small planet-count bonus in midgame.
  - Starting point:
    - `sample90_4p_response_gate_step15_from69`
- Branch:
  - `sample93_4p_leader_power_fleets_from90`
- Implementation:
  - `main.py`
    - `_leader_owner_by_power(...)` now includes:
      - enemy planet ships,
      - enemy fleet ships from `obs.f_owner` / `obs.f_ships`,
      - production score `prod * 13.0`,
      - `planet_count * 5.0` after step 80.
    - Self player remains excluded from leader selection.
- Syntax:
  - `py_compile` passed for `main.py`.
- Known 24-seed eval:
  - `sample90`: `13W/11L`, avg diff `-611.21`, avg placement `1.54`
  - `sample93`: `13W/11L`, avg diff `-611.21`, avg placement `1.54`
- Interpretation:
  - Local change is safe on known 24 seeds.
  - No measurable gain on this set, likely because leader identity did not change often enough to affect selected actions.
  - Keep as a low-risk candidate, but current best practical baseline remains `sample90` unless broader random evaluation shows improvement.

## 2026-06-22 sample94/sample95 midgame reset, terminal, safe reserve

- Context:
  - User requested improvements D/E/F:
    - D: lightweight midgame mode validity check,
    - E: 4P terminal phase should stay cautious and keep defensive regroup/response,
    - F: expand safe neutral reservation from 1 to 2-3 when safe.
- Branches:
  - `sample94_4p_midgame_terminal_safe_from90`
  - `sample95_4p_midgame_safe_light_from90`
- `sample94` implementation:
  - Based on `sample90`.
  - Added `winner_path` reset after step 70 if:
    - anchor is already enemy-owned, or
    - at least half of planned targets are enemy-owned.
  - Added 4P terminal phase:
    - `terminal_4p_roi_threshold=1.3`,
    - `terminal_4p_enable_regroup=True`,
    - `orbit_response_turn_limit=500`.
  - Enabled `terminal_phase_turns=40` for the main 4P mode configs.
  - Added top-director safe neutral reservation count:
    - `top_director_reserve_count`,
    - prod >= 2 required for reserved safe neutral.
- `sample94` status:
  - `py_compile` passed.
  - Known 24-seed eval:
    - `13W/11L`, avg diff `-611.21`, avg placement `1.54`
  - This is exactly identical to `sample90` on the known 24 seeds.
  - Interpretation:
    - the combined D/E/F changes are behaviorally neutral on the current known set,
    - no measurable gain, no measurable regression.
- `sample95` implementation:
  - Based again on `sample90`.
  - Keeps the low-risk parts:
    - winner_path reset after step 70,
    - prod >= 2 filter for top-director safe neutral reserve,
    - top-director reserve count can be 2,
    - `orbit_response_turn_limit=500`.
  - Does not enable terminal regroup across all modes.
- Syntax:
  - `py_compile` passed for `sample95`.
- Mini eval:
  - Seeds: `1301457313,2012339891,1796441431,1426214042`
  - Result: `2W/2L`, avg diff `-96.75`, avg placement `1.50`.
  - Same win/loss pattern as known `sample90` results on these seeds.
- Interpretation:
  - `sample95` is safe enough to test further.
  - `sample94` is too heavy/risky as a combined D/E/F variant.
  - Current practical best remains `sample90` until `sample95` beats it on known 24 or random100.

## 2026-06-22 sample95 Kaggle submission package

- Package:
  - `sample95_4p_midgame_safe_light_submit.zip`
- Structure:
  - root `main.py`
  - root `params.json`
  - root `orbit_lite/`
- Reason:
  - The flat structure matches `sample57_4p_top_director_submit.zip` and preserves the same import behavior as the source folder.
  - Avoids wrapper-loader behavior differences seen with nested submit packages.
- Verification:
  - `py_compile` passed for `submission_builds/sample95_submit_flat/main.py`.
  - `__file__`-less exec check passed: root `agent` is callable.
  - Mini 4-seed comparison against the source folder matched exactly.
    - Seeds: `1301457313,2012339891,1796441431,1426214042`
    - Source `sample95`: `2W/2L`, avg diff `-96.75`, avg placement `1.50`
    - Submit flat `sample95`: `2W/2L`, avg diff `-96.75`, avg placement `1.50`
    - Per-seed score/diff/length/survival all matched.

## 2026-06-22 sample90 Kaggle submission package

- Package:
  - `sample90_4p_response_gate_step15_submit_flat.zip`
- Structure:
  - root `main.py`
  - root `params.json`
  - root `orbit_lite/`
- Reason:
  - Rebuilt using the same flat submit structure as `sample95`.
  - This avoids nested wrapper import behavior and matches the source folder's module layout.
- Verification:
  - `py_compile` passed for `submission_builds/sample90_submit_flat_v2/main.py`.
  - `__file__`-less exec check passed: root `agent` is callable.
  - Mini 4-seed comparison against the source folder matched exactly.
    - Seeds: `1301457313,2012339891,1796441431,1426214042`
    - Source `sample90`: `2W/2L`, avg diff `-96.75`, avg placement `1.50`
    - Submit flat `sample90`: `2W/2L`, avg diff `-96.75`, avg placement `1.50`
    - Per-seed score/diff/length/survival all matched.

## 2026-06-22 Kaggle submit zip recipe

- Use the flat submit structure for orbit_lite-based bots.
- Do not use a nested wrapper unless absolutely necessary.
- Required zip contents:
  - root `main.py`
  - root `params.json` if the bot reads it
  - root `orbit_lite/`
- Do not include:
  - `__pycache__/`
  - old inner zip files such as `sample69.zip`
  - `candidate_metadata.json` unless explicitly needed by runtime
- Why flat is preferred:
  - It matches the source folder's import behavior.
  - It avoids wrapper-loader differences from `sys.path` and `orbit_lite` module caching.
  - It also avoids Kaggle raw-python issues where `__file__` may be undefined in a root wrapper.
- PowerShell recipe:

```powershell
$src = 'C:\Users\yuu98\Desktop\kaggle\orbit-wars\sample90_4p_response_gate_step15_from69'
$stage = 'C:\Users\yuu98\Desktop\kaggle\orbit-wars\submission_builds\sample90_submit_flat_v2'
$out = 'C:\Users\yuu98\Desktop\kaggle\orbit-wars\sample90_4p_response_gate_step15_submit_flat.zip'

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null

Copy-Item "$src\main.py" $stage
Copy-Item "$src\params.json" $stage
Copy-Item "$src\orbit_lite" $stage -Recurse

Get-ChildItem "$stage\orbit_lite" -Recurse -Directory -Filter '__pycache__' |
  Remove-Item -Recurse -Force

if (Test-Path $out) { Remove-Item $out -Force }
Push-Location $stage
Compress-Archive -Path main.py, params.json, orbit_lite -DestinationPath $out
Pop-Location
```

- Structure check:

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::OpenRead($out).Entries |
  Select-Object FullName, Length
```

- `__file__`-less load check:

```powershell
@'
import os
path = r'C:\Users\yuu98\Desktop\kaggle\orbit-wars\submission_builds\sample90_submit_flat_v2\main.py'
code = open(path, 'r', encoding='utf-8').read()
env = {'__name__': '__main__'}
os.chdir(r'C:\Users\yuu98\Desktop\kaggle\orbit-wars\submission_builds\sample90_submit_flat_v2')
exec(compile(code, path, 'exec'), env)
print('agent' in env, callable(env.get('agent')))
'@ | C:\tmp\ow\Scripts\python.exe -
```

- Behavior match check:

```powershell
$mini = '1301457313,2012339891,1796441431,1426214042'

C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 `
  --agent sample90_4p_response_gate_step15_from69\main.py `
  --opponent sample7\main.py `
  --opponent sample8\main.py `
  --opponent bots\hairate5.py `
  --seed-list $mini --workers 2 |
  Select-String 'Wins|Losses|Crash rate|Average placement|Average score diff|seed='

C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 `
  --agent submission_builds\sample90_submit_flat_v2\main.py `
  --opponent sample7\main.py `
  --opponent sample8\main.py `
  --opponent bots\hairate5.py `
  --seed-list $mini --workers 2 |
  Select-String 'Wins|Losses|Crash rate|Average placement|Average score diff|seed='
```

- Adoption rule:
  - The submit build must match the source folder on the mini seeds.
  - At minimum, compare `Wins/Losses`, `Average score diff`, `Average placement`, and each per-seed `score/diff/length/survival`.

## 2026-06-22 sample96: orbit_lite Wave1 influence + intent from sample90

- Parent:
  - `sample90_4p_response_gate_step15_from69`
- New folder:
  - `sample96_4p_influence_intent_from90`
- Goal:
  - Move beyond parameter/selector tweaks by adding low-risk board understanding to orbit_lite.
  - 4P only. 2P behavior is untouched by config flags.
- Implemented:
  - `enable_influence_map_4p`
    - Computes a per-planet control score from friendly/enemy planet ships and production with distance decay.
    - Gives small bonuses for safe neutral targets and own planets under pressure.
    - Penalizes neutral/enemy targets deep inside enemy influence.
  - `enable_fleet_intent_4p`
    - Uses observed enemy fleet position, angle, ships, and speed to estimate likely target planets.
    - Penalizes attacking/capturing targets that enemy fleets appear to be converging on.
    - Gives a small defense bonus to own targets under predicted fleet intent pressure.
- Safety:
  - Added as score adjustment after existing lane/domain/block/winner/top-director adjustments and before orbit response gate.
  - Weak coefficients in v1 to avoid large score-shape breakage.
- Checks:
  - `py_compile`: PASS.
  - Mini 4 seeds `1301457313,2012339891,1796441431,1426214042`:
    - `2W/2L`, crash `0.0%`, average placement `1.50`, average score diff `-280.25`.
    - Same W/L as sample90 on this mini set, but worse score diff than sample90's earlier `-96.75`.
- Not run:
  - Known24 was intentionally stopped due time cost.
  - Random100 not run.
- Current read:
  - sample96 is not broken, but v1 does not yet prove improvement.
  - Next useful cheap step is coefficient ablation on a tiny seed set:
    - influence only
    - intent only
    - both weaker

## 2026-06-22 sample97: Wave1 dynamic adaptation from sample96

- Parent:
  - `sample96_4p_influence_intent_from90`
- New folder:
  - `sample97_4p_wave1_dynamic_from96`
- Goal:
  - Finish Wave1 by adding a low-risk dynamic strategy adaptation layer.
  - Keep sample96's influence map and fleet intent prediction.
- Implemented:
  - `_owner_power_snapshot`
    - Counts per-owner planet ships, production, planet count, and fleet ships.
  - `_maybe_reconsider_4p_mode`
    - After step 60, only reconsider high-variance modes:
      - `winner_outer_domain`
      - `enemy_domain_block`
      - `top_director`
      - `s8_burst`
    - If the opening failed to produce territory by midgame, fall back to `s7_stable`.
    - `s8_burst` also falls back to `s7_stable` after step 90.
  - `_apply_dynamic_4p_config`
    - If ahead, slightly raises ROI and keeps regroup/response active.
    - If behind, slightly lowers ROI and keeps regroup active.
  - 2P remains unchanged.
- Checks:
  - `py_compile`: PASS.
  - 8 seed comparison:
    - seeds: `1301457313,230622801,2012339891,315400519,1796441431,1426214042,754928456,1846002515`
    - sample90: `4W/4L`, crash `0.0%`, average placement `1.62`
    - sample96: `4W/4L`, crash `0.0%`, average placement `1.62`
    - sample97: `4W/4L`, crash `0.0%`, average placement `1.62`
- Current read:
  - Wave1 additions did not reduce win count on this 8 seed check.
  - No win gain yet.
  - Since user prioritizes win count over diff, sample97 remains a valid experimental continuation, but sample90 is still the proven submit baseline.

## 2026-06-22 sample98: Wave2 Phase1 coordinated follow-up

- Parent:
  - `sample97_4p_wave1_dynamic_from96`
- New folder:
  - `sample98_4p_coord_phase1_from97`
- Goal:
  - Add a low-risk post-greedy version of multi-source/coordinated attack without rewriting `_greedy_select`.
- Implemented:
  - `enable_coord_followup_4p`
  - `_plan_coord_followups`
    - Runs after normal greedy waves and before regroup.
    - Adds at most one extra follow-up launch from leftover budget.
    - Step window: 60-150.
    - Only targets already selected by normal greedy.
    - Requires high-value targets: enemy target or neutral target with production >= 4.
    - Uses existing `intercept_angle` for engine-faithful aiming.
    - Debits leftover source budget before regroup.
- Checks:
  - `py_compile`: PASS.
  - Submit flat build load check without `__file__`: PASS (`agent_callable=True`).
- 8 seed eval:
  - v1 before tightening: `3W/5L`, crash `0.0%`, average placement `1.75`.
  - It flipped `1426214042` from loss to win, but lost known wins `1796441431` and `754928456`.
  - Tightened v2 parameters afterward:
    - `coord_start_turn=60`
    - `coord_max_extra=1`
    - higher source reserve
    - lower send fraction/cap
    - `coord_min_target_prod=4`
  - v2 8 seed eval was not completed because it was intentionally interrupted.
- Submit artifact:
  - `sample98_4p_coord_phase1_submit_flat.zip`
  - Structure: root `main.py`, root `params.json`, root `orbit_lite/`.
- Current read:
  - ZIP is structurally submit-ready.
  - Performance is not proven; sample90 remains the proven baseline.

## 2026-06-22 sample99: s8_burst-only coordinated follow-up

- Parent:
  - `sample98_4p_coord_phase1_from97`
- New folder:
  - `sample99_4p_s8_coord_only_from98`
- Change:
  - Disabled coordinated follow-up in global `CONFIG_4P`.
  - Enabled coordinated follow-up only for `CONFIG_4P_S8_BURST`.
  - `top_director`, `s7_stable`, `lane_anchor`, `winner_outer_domain`, and `enemy_domain_block` do not use the follow-up.
- Reason:
  - sample98 all-mode follow-up hurt stable/lane wins.
  - The 8 seed probe showed the follow-up mainly rescued the `s8_burst` loss seed `1426214042`.
- 8 seed check:
  - seeds: `1301457313,230622801,2012339891,315400519,1796441431,1426214042,754928456,1846002515`
  - sample90/sample97 baseline: `4W/4L`
  - sample99: `5W/3L`, crash `0.0%`, average placement `1.50`
- Known24 user run:
  - `14W/10L`
  - crash `0.0%`
  - average score diff `-494.38`
  - average placement `1.50`
  - average game length `220.58`
  - average survival turn `185.83`
- Comparison:
  - sample90 known24 was `13W/11L`, average score diff `-611.21`, average placement `1.54`.
  - sample99 is +1W on known24 and slightly better placement/diff.
- Current read:
  - sample99 is the best known 4P branch in this line.
  - It should replace sample98 as the Wave2 submit candidate.

## 2026-06-22 4P diagnostic tooling

- Added:
  - `tools/diagnose_4p.py`
  - `tools/selector_probe_4p.py`
- Purpose:
  - For each 4P seed, show:
    - result / placement / diff
    - final 1st-4th standings with bot names and scores
    - primary agent selector mode
    - compact initial selector features
- Example:

```powershell
C:\tmp\ow\Scripts\python.exe tools\diagnose_4p.py `
  --agent sample99_4p_s8_coord_only_from98\main.py `
  --opponent sample7\main.py `
  --opponent sample8\main.py `
  --opponent bots\hairate5.py `
  --seed-list '1426214042,1301457313' `
  --show-features `
  --quiet-env-noise
```

- Sample output:

```text
seed=1426214042 result=W place=1 diff=2189 mode=s8_burst length=215 survival=215
  standings: 1:sample99_4p_s8_coord_only_from98(seat0,score=2189) | ...
  features: enemy_dist=76.3 n25_prod=12.0 n45_prod=22.0 ...
```

- Fast selector-only probe:
  - Does not run games.
  - Uses `env.reset(4)` only, then prints the primary agent's step-0 selector mode and compact board features.
  - Useful for random100 mode hit-rate checks and seed clustering.

```powershell
C:\tmp\ow\Scripts\python.exe tools\selector_probe_4p.py `
  --agent sample99_4p_s8_coord_only_from98\main.py `
  --seed-list '1426214042,1301457313,754928456' `
  --quiet-env-noise
```

- CSV version:

```powershell
C:\tmp\ow\Scripts\python.exe tools\selector_probe_4p.py `
  --agent sample99_4p_s8_coord_only_from98\main.py `
  --seed-list '1426214042,1301457313' `
  --csv `
  --quiet-env-noise
```

## 2026-06-22 4P experiment toolbox notes

### 1. Full per-seed diagnosis

- Tool:
  - `tools/diagnose_4p.py`
- Use when:
  - Need W/L/D, placement, score diff, final standings, selected mode, and initial features.
  - Best for a small set of important seeds because it runs full games.
- Example:

```powershell
$seeds = '466860781,1365116226,1151776762,1779716096'

C:\tmp\ow\Scripts\python.exe tools\diagnose_4p.py `
  --agent sample99_4p_s8_coord_only_from98\main.py `
  --opponent sample7\main.py `
  --opponent sample8\main.py `
  --opponent bots\hairate5.py `
  --seed-list $seeds `
  --show-features `
  --quiet-env-noise
```

- Output includes:
  - `seed`
  - `result`
  - `place`
  - `diff`
  - `mode`
  - `length`
  - `survival`
  - final `standings`
  - compact initial `features`

### 2. Fast selector / feature probe

- Tool:
  - `tools/selector_probe_4p.py`
- Use when:
  - Need only selected mode and initial board features.
  - Does not run the game; uses only `env.reset(4)`.
  - Best for random100/random200 mode hit-rate checks.
- Compact text:

```powershell
$seedList = 1..200 | ForEach-Object { Get-Random -Minimum 1 -Maximum 2147483647 }

C:\tmp\ow\Scripts\python.exe tools\selector_probe_4p.py `
  --agent sample99_4p_s8_coord_only_from98\main.py `
  --seed-list ($seedList -join ',') `
  --quiet-env-noise
```

- CSV:

```powershell
C:\tmp\ow\Scripts\python.exe tools\selector_probe_4p.py `
  --agent sample99_4p_s8_coord_only_from98\main.py `
  --seed-list ($seedList -join ',') `
  --csv `
  --quiet-env-noise > selector_random200_sample99.csv
```

### 3. Forced-mode bots

- Purpose:
  - Determine whether selector chose the wrong mode or the selected mode is best but still loses.
- Created from sample99:
  - `sample100_force_s7_from99` -> always `s7_stable`
  - `sample101_force_s8_from99` -> always `s8_burst`
  - `sample102_force_lane_from99` -> always `lane_anchor`
  - `sample103_force_enemy_block_from99` -> always `enemy_domain_block`
- All are sample99 except `_choose_4p_mode` returns a fixed mode.

### 4. Forced-mode standings on loss seeds

- Use `diagnose_4p.py` rather than `evaluate.py` when inspecting per-seed ranking.

```powershell
$loss = '466860781,1365116226,1151776762,1779716096,1880614615,1281573619,1599225194,608151314'

Write-Host "=== force s7_stable ==="
C:\tmp\ow\Scripts\python.exe tools\diagnose_4p.py `
  --agent sample100_force_s7_from99\main.py `
  --opponent sample7\main.py `
  --opponent sample8\main.py `
  --opponent bots\hairate5.py `
  --seed-list $loss `
  --show-features `
  --quiet-env-noise

Write-Host "=== force s8_burst ==="
C:\tmp\ow\Scripts\python.exe tools\diagnose_4p.py `
  --agent sample101_force_s8_from99\main.py `
  --opponent sample7\main.py `
  --opponent sample8\main.py `
  --opponent bots\hairate5.py `
  --seed-list $loss `
  --show-features `
  --quiet-env-noise

Write-Host "=== force lane_anchor ==="
C:\tmp\ow\Scripts\python.exe tools\diagnose_4p.py `
  --agent sample102_force_lane_from99\main.py `
  --opponent sample7\main.py `
  --opponent sample8\main.py `
  --opponent bots\hairate5.py `
  --seed-list $loss `
  --show-features `
  --quiet-env-noise

Write-Host "=== force enemy_domain_block ==="
C:\tmp\ow\Scripts\python.exe tools\diagnose_4p.py `
  --agent sample103_force_enemy_block_from99\main.py `
  --opponent sample7\main.py `
  --opponent sample8\main.py `
  --opponent bots\hairate5.py `
  --seed-list $loss `
  --show-features `
  --quiet-env-noise
```

### 5. Normal aggregate evaluation

- Use `evaluate.py` for win-rate summaries.
- Keep output filtered during iterative work:

```powershell
$all = '1284299523,1301457313,230622801,1218569221,718279990,638588263,676586300,27311426,736262817,1964410425,2012339891,363175246,315400519,1796441431,754928456,915403528,927652245,1426214042,306239195,2112295433,49433498,1488820789,1846002515,1309011149'

C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 `
  --agent sample99_4p_s8_coord_only_from98\main.py `
  --opponent sample7\main.py `
  --opponent sample8\main.py `
  --opponent bots\hairate5.py `
  --seed-list $all `
  --workers 8 |
  Select-String 'Wins|Losses|Crash rate|Average placement|Average score diff|seed='
```

### 6. Submit zip recipe

- Current submit candidate:
  - `sample99_4p_s8_coord_only_submit_flat.zip`
- Required structure:
  - root `main.py`
  - root `params.json`
  - root `orbit_lite/`
- Always verify:
  - `py_compile`
  - zip entry structure
  - `__file__`-less load check with `agent_callable=True`

## 2026-06-22 sample104: Wave3 mini-rollout v1

Parent:
- `sample99_4p_s8_coord_only_from98`

New bot:
- `sample104_4p_mini_rollout_from99`

Submit artifact:
- `sample104_4p_mini_rollout_submit_flat.zip`

Purpose:
- Wave3 first pass.
- Add a lightweight 4P mini-rollout style candidate rescoring layer without replacing the existing sample99 executor.
- The goal is to reject attacks/captures that look good greedily but collapse after local arrivals, source stripping, or quick enemy response.

Implemented behavior:
- Added `_apply_mini_rollout_adjustment(...)`.
- Enabled only for 4P, step 20-160.
- Re-scores top 24 already-scored candidates after orbit response gate and before greedy selection.
- Uses existing forecast data instead of new full simulation:
  - target garrison margin at ETA
  - friendly/enemy arrivals in the next 10 turns after ETA
  - current nearby enemy planet response pressure
  - source strip risk
  - non-leader enemy attack penalty

Config:
- `enable_mini_rollout_4p=True`
- `mini_rollout_start=20`
- `mini_rollout_turn_limit=160`
- `mini_rollout_top_k=24`
- `mini_rollout_window=10`
- `mini_rollout_hold_bonus=0.22`
- `mini_rollout_fail_penalty=0.55`
- `mini_rollout_source_strip_penalty=0.22`
- `mini_rollout_nonleader_enemy_penalty=0.18`

Checks:
- `py_compile`: PASS
- Submit load check: `agent_callable=True`

Small evaluation:
- 8-seed sanity set:
  - seeds: `1301457313,230622801,2012339891,315400519,1796441431,1426214042,754928456,1846002515`
  - result: `5W/3L`, crash 0.0%, average placement 1.50, average score diff +66.75
  - same W/L as sample99 on this set.
- loss8 probe:
  - seeds: `466860781,1365116226,1151776762,1779716096,1880614615,1281573619,1599225194,608151314`
  - result: `1W/7L`, crash 0.0%, average placement 1.88, average score diff -1622.88
  - picked up seed `1151776762`.

Current read:
- sample104 does not regress the first sanity 8.
- It may rescue at least one loss-seed shape, but it is not proven as a replacement for sample99 yet.
- Next decision should use known24 or a fresh random set, depending on time.

## 2026-06-22 orbit_lite essential-improvement plan: experiment summary

Source plan:
- Attachment: `orbit_lite 本質的改善計画 — sample90 以降`

Plan diagnosis:
- A-F tuning showed that simple parameter/selector changes are nearing a ceiling.
- Main structural limits identified:
  - depth-0 evaluation: actions are scored as if opponents do not respond,
  - no wave interaction: later wave values are not re-scored after earlier selections,
  - L=1 candidate structure: coordinated multi-source attacks are not native,
  - turn-0 strategy lock: mode rarely changes with midgame state,
  - unused information: fleet angle/source/history are only partially used.
- Core lesson from contest-style bots:
  - shallow lookahead plus a good evaluation function is more important than piling on many depth-0 heuristics.

### A-F results before the Wave plan

- A: early `orbit_response_gate`
  - Implemented in `sample89`, `sample90`, `sample91`.
  - Best result: `sample90_4p_response_gate_step15_from69`.
  - Known24: `13W/11L`, avg diff `-611.21`, avg placement `1.54`.
  - Random100: `43W/2D/55L`, avg diff `-343.90`, avg placement `1.56`.
  - Verdict: useful and adopted as the new baseline after sample69.

- B: leader-weighted `competitive_score`
  - Implemented in `sample92_4p_leader_weighted_score_from90`.
  - Result: catastrophic on known24 (`0W/24L` in the recorded experiment).
  - Verdict: changing the score root directly is dangerous; do not continue this form.

- C: leader power includes fleets
  - Implemented in `sample93_4p_leader_power_fleets_from90`.
  - Known24: same as sample90, `13W/11L`, avg diff `-611.21`, avg placement `1.54`.
  - Verdict: safe but no measurable gain on the known set.

- D/E/F: midgame reset, 4P terminal config, safe reserve expansion
  - Implemented in `sample94_4p_midgame_terminal_safe_from90` and `sample95_4p_midgame_safe_light_from90`.
  - `sample94`: same known24 result as sample90.
  - `sample95`: mini tests matched sample90 behavior; submit package was created and source/zip matched on mini check.
  - Verdict: not a practical improvement over sample90.

- lane_anchor selector expansion
  - Known24 improved in some targeted tests, but random100 later regressed:
    - sample69 random100: `40W`, `58L`, avg diff `-448.84`, placement `1.59`.
    - sample88 random100: `39W`, `60L`, avg diff `-867.45`, placement `1.63`.
  - Verdict: seed-specific selector rules overfit easily.

### Wave results

- Wave1 alpha/beta/epsilon: influence map, fleet intent, dynamic adaptation
  - `sample96_4p_wave1_influence_intent_from90`
  - `sample97_4p_wave1_dynamic_from96`
  - 8-seed check:
    - sample90: `4W/4L`, crash `0.0%`, average placement `1.62`
    - sample97: `4W/4L`, crash `0.0%`, average placement `1.62`
  - Verdict: did not reduce win count, but did not prove a clear gain. Kept as a valid continuation but not adopted by itself.

- Wave2 gamma Phase 1: post-greedy coordinated follow-up
  - `sample98_4p_coord_phase1_from97`
  - all-mode coordination hurt stable/lane cases.
  - Refined into `sample99_4p_s8_coord_only_from98`, enabling coordination only for `s8_burst`.
  - 8-seed sanity:
    - sample90/sample97 baseline: `4W/4L`
    - sample99: `5W/3L`, crash `0.0%`, average placement `1.50`
  - Known24:
    - sample90: `13W/11L`, avg diff `-611.21`, placement `1.54`
    - sample99: `14W/10L`, avg diff `-494.38`, placement `1.50`
  - Verdict: real improvement. sample99 replaced sample90 as the best proven 4P branch in this line.

- Wave3 delta v1: lightweight mini-rollout candidate rescoring
  - `sample104_4p_mini_rollout_from99`
  - Added `_apply_mini_rollout_adjustment(...)` as top-k candidate re-score after response gate and before greedy.
  - 8-seed sanity:
    - sample104: `5W/3L`, crash `0.0%`, avg placement `1.50`, avg diff `+66.75`
    - same win count as sample99 on that set.
  - loss8 probe:
    - sample104: `1W/7L`, crash `0.0%`, avg placement `1.88`, avg diff `-1622.88`
    - picked up seed `1151776762`.
  - Direct table vs current bots on loss8, from user run:
    - sample104 won 4 of 8 seeds,
    - sample99 won 2,
    - sample90 won 1,
    - `6_22_3/6_22_4` won 1.
  - Verdict: sample104 is promising and may be stronger than sample99 on the loss8 shape, but needs known24/random validation before replacing sample99 as the main submit candidate.

### External 6_22 family check

- User tested `6_22`, `6_22_2`, `6_22_3`, `6_22_4` against each other on loss8:
  - `6_22_3` looked strong inside that family.
- Follow-up against current bots showed this was mostly an internal-family effect:
  - on loss8 with `sample99`, `sample104`, `sample90` in the table:
    - sample104: 4 wins,
    - sample99: 2 wins,
    - sample90: 1 win,
    - 6_22_3/4: 1 win.
- Verdict:
  - Do not rush to embed 6_22_3/4 into the selector.
  - sample104 is currently the more relevant branch to validate.

### Current strategic read

- Proven baseline sequence:
  - sample69 -> sample90 -> sample99.
- Current promising branch:
  - sample104 from sample99.
- Most useful structural directions:
  - Wave2-style targeted coordination works when narrowly scoped.
  - Wave3-style mini-rollout is the right conceptual direction, but v1 is still a lightweight score adjustment rather than full depth-2 search.
- Lower-value directions:
  - direct leader-weighted competitive score,
  - broad lane_anchor selector expansion,
  - broad terminal/midgame/safe-reserve parameter changes.

### Next suggested step

- Before Wave4:
  - validate sample104 against sample99 on known24 or a small fresh random set.
- Wave4 should not be another parameter tweak.
- Wave4 should be one of:
  - true multi-source candidate generation for high-value targets only, or
  - a small action-script rollout that compares 2-3 candidate scripts, not just score corrections.

## 2026-06-22 sample105: per-player best-response mini-rollout

Parent:
- `sample104_4p_mini_rollout_from99`

New bot:
- `sample105_4p_per_player_response_from104`

Goal:
- Keep the sample104 mini-rollout structure, but replace the flat enemy pressure estimate with a per-player best-response approximation.
- Instead of summing every enemy planet with a flat fraction, each enemy player contributes the best reachable response source to the candidate target.

Implementation:
- Added config:
  - `one_ply_response_frac=0.55`
  - `one_ply_response_min_ships=8.0`
  - `one_ply_response_eta_cap=12.0`
- Replaced `local_enemy` inside `_apply_mini_rollout_adjustment(...)`.
- The rest of sample104 scoring/greedy behavior is unchanged.

Checks:
- `py_compile`: PASS

Evaluation:
- Sanity 8:
  - sample105: `4W/4L`, crash `0.0%`, avg diff `-390.75`, placement `1.62`
  - sample104 was previously recorded as `5W/3L` on the same sanity shape, so this was not immediately adopted.
- Random10 direct comparison:
  - sample105: `4W/6L`, avg diff `-337.30`, placement `1.60`
  - sample104: `4W/6L`, avg diff `-301.80`, placement `1.60`
  - Same win/loss pattern.
- Random40 direct comparison:
  - sample104: `17W/3D/20L`, avg diff `+542.35`, placement `1.50`
  - sample105: `17W/3D/20L`, avg diff `+543.02`, placement `1.50`
  - Same win/loss/draw pattern across all 40 seeds.
  - Only small score/game-length changes on a few seeds; no result flip.

Current read:
- sample105 does not meaningfully change behavior on random40.
- It is not worse on the random40 set, but it also does not prove improvement.
- The per-player best-response calculation is probably not crossing enough score/greedy thresholds to change selected actions.
- This suggests the next Wave should not only change a penalty estimate; it should alter action selection more directly, e.g. true multi-source candidate generation or script-level rollout.

## 2026-06-22 sample107/sample108: 4P one-ply candidate-selection experiments

Parent:
- `sample104_4p_mini_rollout_from99`

### sample107: one-ply candidate reservation

New bot:
- `sample107_4p_oneply_candidate_from104`

Goal:
- Make one-ply affect actual action selection, not just score adjustment.
- Before normal greedy, reserve at most one high-confidence candidate after estimating each enemy player's best response.
- The reserved candidate consumes source budget and removes that target from normal greedy.

Implementation:
- Added `enable_one_ply_candidate_4p`.
- Added `_plan_one_ply_candidate_reserve(...)`.
- Looks at top-k scored candidates, estimates per-player response, computes `future_hold`, and reserves one action if it passes the one-ply threshold.

Result:
- Initial check on seeds `1121763242,1359411720`:
  - `0W/2L`, crash `0.0%`.
- Disabling the one-ply reserve in a debug copy restored sample104 behavior:
  - `1W/1L`, crash `0.0%`.
- Verdict:
  - The reservation approach is too intrusive.
  - It disrupts the early/midgame greedy plan and can turn a sample104 win into a loss.
  - Keep as a negative result; do not use as submit candidate.

### sample108: one-ply toxic-target filter

New bot:
- `sample108_4p_oneply_filter_from104`

Goal:
- Keep the one-ply concept, but avoid preemptive reservation.
- Instead, make one-ply change the candidate set by hard-filtering toxic targets.

Implementation:
- Based on `sample105_4p_per_player_response_from104`.
- Strengthened the existing mini-rollout/per-player response:
  - `mini_rollout_top_k=36`
  - `mini_rollout_hold_bonus=0.72`
  - `mini_rollout_fail_penalty=2.2`
  - `mini_rollout_source_strip_penalty=0.55`
  - `one_ply_response_frac=0.65`
- Added hard filter:
  - if a neutral/enemy target has `prod >= 2` and `future_hold < -6`, set local score to `-inf`.

Result:
- Same two-seed sanity:
  - `1W/1L`, crash `0.0%`, avg placement `1.50`.
  - Keeps the `1121763242` win that sample107 lost.
- Verdict:
  - This is the safer one-ply action-selection direction.
  - It changes candidate selection through hard toxic-target removal without stealing the first normal greedy wave.
  - Needs random/known comparison against sample104 before judging.

Random20 comparison against sample104:
- Seeds included:
  - `98834007,1522561038,460985805,1543026641,719921714,808936432,1193654019,1207918184,1661603053,595398685,267117348,1375371788,203644404,459407471,1407436312,460368473,734500522,1616261939,1680314452,860430514`
- sample104:
  - `5W/0D/15L`, crash `0.0%`, avg diff `-1670.15`, placement `1.80`
- sample108:
  - `5W/0D/15L`, crash `0.0%`, avg diff `-1644.90`, placement `1.75`
- Seed flips:
  - `1661603053`: sample104 win -> sample108 loss
  - `1616261939`: sample104 loss -> sample108 win
- Read:
  - Win count did not improve, but behavior is not identical.
  - sample108 slightly improved placement/diff on this set, but the main signal is weak.
  - Since the set itself is hard for both bots, sample108 is not adoptable yet; it remains an experimental branch for analyzing flipped seeds.

## sample109 true one-ply response selection

Date:
- 2026-06-23

Artifact:
- `sample109_4p_true_oneply_from104`

Goal:
- Implement a serious 4P one-ply candidate evaluator:
  1. hypothetically apply our candidate launch,
  2. add each enemy's best immediate response launch to that target,
  3. recompute exact sparse garrison flow delta,
  4. rescore/re-rank the normal greedy candidates from the response-included state.

Implementation:
- Based on `sample104_4p_mini_rollout_from99`.
- Added `enable_true_one_ply_4p`.
- Added `_build_enemy_best_response_launches(...)`.
  - For each enemy, chooses one owned source with best response pressure to the candidate target.
  - Uses ship fraction, source reserve, eta cap, and target production to model a plausible counter.
- Added `_apply_true_one_ply_rescore(...)`.
  - Evaluates top candidate set with `sparse_launch_flow_delta`.
  - Uses `LaunchSet` containing our launches plus enemy response launches.
  - Scores by `one_ply_net = my_net_delta - weighted_enemy_net_delta`, with leader upweighted.
  - Strong version uses `true_one_ply_base_weight=0.45` and `true_one_ply_net_scale=0.085`, so the one-ply result can genuinely reorder candidates.

Sanity:
- `py_compile`: pass.
- Crash rate in all tests below: `0.0%`.

Small flip test:
- Seeds: `1207918184,1616261939,1121763242,1359411720`
- sample104:
  - `1W/0D/3L`, avg diff `-1213.25`, placement `1.75`
- sample109:
  - `1W/0D/3L`, avg diff `-1833.50`, placement `1.75`
- Flips:
  - `1616261939`: sample104 loss -> sample109 win
  - `1121763242`: sample104 win -> sample109 loss
- Read:
  - Behavior is clearly changed, not just cosmetic.

Known loss8 test:
- Seeds: `466860781,1365116226,1151776762,1779716096,1880614615,1281573619,1599225194,608151314`
- sample104:
  - `1W/0D/7L`, avg diff `-1622.88`, placement `1.88`
- sample109:
  - `4W/0D/4L`, avg diff `-265.75`, placement `1.50`
- Wins rescued by sample109:
  - `466860781`
  - `1365116226`
  - `1599225194`
- Read:
  - This is the strongest evidence that true one-ply can rescue counterattack-prone losses.
  - The set is known/hard, so this is a promising mechanism signal rather than submission proof.

Random8 test:
- Seeds: `220367059,248542608,1721204295,1386380957,289059467,1872093716,1068866810,1948340546`
- sample104:
  - `2W/0D/6L`, avg diff `+916.12`, placement `1.75`
- sample109:
  - `3W/0D/5L`, avg diff `-524.12`, placement `1.62`
- Flips:
  - `248542608`: sample104 loss -> sample109 win
  - `1068866810`: sample104 loss -> sample109 win
  - `1872093716`: sample104 big win -> sample109 loss
- Read:
  - Win count improved on this small random set, but it sacrificed a huge sample104 win.
  - sample109 is a high-impact experimental branch. It should be compared on random40+ before submission consideration.

## sample110 one-ply-gated high-prod coord followup

Date:
- 2026-06-23

Artifact:
- `sample110_4p_oneply_coord_from109`

Goal:
- Continue the plan after true one-ply:
  - permit `coord_followup` outside the old narrow `s8_burst` usage,
  - but only for high-production targets,
  - and only if the followup itself passes a one-ply response check.

Implementation:
- Based on `sample109_4p_true_oneply_from104`.
- Enabled coord followup in the base 4P config so all major 4P modes can use it.
- Added coord one-ply gate:
  - target must be high-production (`prod >= 4` after tightening),
  - after the normal selected wave plus candidate followup, add each enemy's best response to that same target,
  - recompute exact sparse garrison flow delta,
  - accept the followup only if response-included net remains positive enough.
- Strong but filtered coord settings:
  - `coord_max_extra=2`
  - `coord_send_frac=0.36`
  - `coord_send_cap=42`
  - `coord_one_ply_min_net=5`
  - `coord_one_ply_response_ratio_cap=1.20`

Sanity:
- `py_compile`: pass.
- Crash rate in tests below: `0.0%`.

Loss8 comparison:
- Seeds: `466860781,1365116226,1151776762,1779716096,1880614615,1281573619,1599225194,608151314`
- sample104:
  - `1W/0D/7L`, avg diff `-1622.88`, placement `1.88`
- sample109:
  - `4W/0D/4L`, avg diff `-265.75`, placement `1.50`
- sample110:
  - `5W/0D/3L`, avg diff `+483.75`, placement `1.38`
- New rescue vs sample109:
  - `1880614615`
- Read:
  - Coord followup is doing real work on the known counterattack-prone loss set.

Previous random8:
- Seeds: `220367059,248542608,1721204295,1386380957,289059467,1872093716,1068866810,1948340546`
- sample109:
  - `3W/0D/5L`, avg diff `-524.12`, placement `1.62`
- sample110 first strong pass:
  - `3W/0D/5L`, avg diff `-530.12`, placement `1.62`
- Read:
  - No win-count regression on that small random set.

Fresh random10:
- Seeds: `1195925204,1517082655,79585238,1706423789,857214887,125496488,769180429,464343471,875834072,2121827991`
- sample109:
  - `2W/0D/8L`, avg diff `-1126.70`, placement `1.80`
- sample110 first strong pass:
  - `1W/0D/9L`, lost `857214887`
- sample110 tightened gate:
  - `2W/0D/8L`, avg diff `-1353.40`, placement `1.80`
  - `857214887` restored to a win.
- Read:
  - Tightened one-ply-positive gate avoids the obvious random10 regression while retaining the loss8 gain.
  - Diff remains worse than sample109 on this random10, so sample110 is not proven as a submit replacement yet.
  - Mechanism is strong: use it for more random40/random100 testing.

## sample112 fixed S7 baseline

Date:
- 2026-06-23

Artifact:
- `sample112_4p_s7_fixed_from110`

Goal:
- Before pushing selector-unification further, test the pure baseline:
  - keep sample110's one-ply/coord implementation available,
  - force the 4P selector to always return `s7_stable`.

Implementation:
- Copied from `sample110_4p_oneply_coord_from109`.
- `_choose_4p_mode()` now returns `s7_stable` immediately for all non-fallback 4P boards.
- 2P behavior unchanged.

Sanity:
- `py_compile`: pass.
- Probe on sample seeds confirmed mode is always `s7_stable`.

Fresh random30 comparison set:
- Seeds: `1657384936,1351844563,1351611807,53351919,457793527,1190946047,1890565501,1767874900,1625436819,1798067820,1182650381,786534061,560842328,1319582963,1438880892,1851779903,1209263198,1271979858,1458181989,200181723,638633100,1414997908,1771745689,1397683760,1297486259,157338764,1670733138,165725081,1714320719,651533336`
- sample110:
  - `13W/0D/17L`, avg diff `-651.37`, placement `1.60`
- sample112 fixed S7:
  - `12W/0D/18L`, avg diff `-803.63`, placement `1.63`

Read:
- Fixed S7 is stable but below sample110 on this set.
- The lost extra win is consistent with sample110's selector picking a non-S7 mode on at least one useful board, especially the prior s8-burst rescue around `165725081`.
- Pure selector removal is not currently better; selector should stay, or unified mode needs a stronger internal replacement for the useful burst/lane cases.

## 2P one-ply transfer from sample110

Date:
- 2026-06-23

Goal:
- Test whether the 4P one-ply / response-gate work can improve 2P, using `sample110_4p_oneply_coord_from109` as parent.
- Keep 4P behavior structurally unchanged; only add 2P enable switches and 2P config variants.

Artifacts:
- `sample113_2p_oneply_s8_from110`
  - 2P true one-ply enabled from step 10.
  - 2P high-prod coord followup enabled from step 20.
  - Strong version intended to visibly change behavior.
- `sample114_2p_oneply_gate_no_coord_from113`
  - Same as sample113 but 2P coord disabled.
- `sample115_2p_oneply_light_gate_from114`
  - 2P one-ply made much lighter: original score mostly preserved, only obvious bad responses penalized.
- `sample116_2p_comeback_oneply_from115`
  - 2P one-ply only after step 45 and only when enemy production/power is ahead.

Sanity:
- `py_compile`: pass for tested variants.
- Crash rate in all 2P tests below: `0.0%`.

Set1 vs `sample8`:
- Seeds: `1456205561,83554485,1084008127,260721277,419543473,2103738413,2124103539,1523877248,199088422,1250205891`
- sample110 baseline:
  - `4W/0D/6L`, avg diff `-573.70`, placement `1.60`
- sample113 strong one-ply+coord:
  - `3W/0D/7L`, avg diff `-899.00`, placement `1.70`
- sample114 one-ply only:
  - `2W/0D/8L`, avg diff `-1447.50`, placement `1.80`
- sample115 light one-ply:
  - `5W/0D/5L`, avg diff `+324.20`, placement `1.50`
- sample116 comeback-gated one-ply:
  - `4W/0D/6L`, avg diff `-410.50`, placement `1.60`

Set2 vs `sample8`:
- Seeds: `1129430366,1359411720,1121763242,1608872725,120542422,412771289,1891750302,2048874186,1405906997,666650506`
- sample110 baseline:
  - `5W/0D/5L`, avg diff `-158.90`, placement `1.50`
- sample115 light one-ply:
  - `2W/0D/8L`, avg diff `-1286.70`, placement `1.80`
- sample116 comeback-gated one-ply:
  - `3W/0D/7L`, avg diff `-989.60`, placement `1.70`

Read:
- Directly transplanting 4P one-ply into 2P is not currently good.
- Strong one-ply/coord changes behavior but breaks too many existing sample8-style wins.
- Coord followup is especially risky in 2P because it overcommits into a single opponent's clean response.
- Light one-ply can rescue some seeds, but the effect is not general: it improved Set1 and badly regressed Set2.
- Current best 2P candidate remains the baseline `sample110`/sample8-style behavior; future 2P work should use narrower board-pattern gating or a true local tactical check that only filters launches without rewriting candidate ranking.
## 2026-06-23 sample127: single-package 4P=sample110 with partial 2P sample124 transfer

Goal:
- User wanted the submitted combined bot to use `sample124`-style 2P and `sample110` 4P.
- Dynamic folder wrappers were tested first, but they changed behavior through `orbit_lite` import/cache interference.

Important failed approaches:
- `sample125_2p124_4p110_submit_selector`
  - root wrapper selecting `sample124` for 2P and `sample110` for 4P.
  - After fixing missing `orbit_lite` folders, 2P still collapsed on a 3-seed check:
    - vs `sample8`, seeds `506034149,1084008127,1456205561`: `0W/3L`.
  - Cause: wrapper/import behavior changed runtime behavior despite no crash.
- `sample126_2p124_4p110_submit_integrated`
  - root `sample124`, 4P delegated to `sample110`.
  - 2P matched sample124 on the 3-seed check, but 4P collapsed:
    - seeds `1301457313,2012339891,1796441431,1426214042`: `0W/4L`.
  - Even after caching 4P/2P mode at step 0, 4P remained broken.
- `sample128_2p124_4p110_native4p`
  - root `sample110`, 2P delegated to `sample124`.
  - 4P matched sample110, but 2P collapsed on the same 3-seed check:
    - `0W/3L`.
  - First action matched direct sample124, so the likely issue is multi-agent import/cache interference across turns/opponents.

Adopted safer experiment:
- `sample127_2p124_4p110_singlefile`
  - parent: `sample110_4p_oneply_coord_from109`
  - 4P remains native sample110, no dynamic delegation.
  - 2P changes ported directly into the same file:
    - `enable_2p_mid_hold`
    - `_apply_2p_midgame_hold_bias`
    - `CONFIG_2P_HOLD`
    - `CONFIG_2P_AGGRESSIVE`
    - `CONFIG_2P_REINFORCE_SAFE`
    - `CONFIG_2P_S8_MULTI`
    - `_choose_2p_mode`
    - runtime `mode_2p`
    - 2P skips sample110 phase/dynamic 4P config.

Sanity:
- `py_compile`: passed.
- 4P mini vs `sample7`, `sample8`, `hairate5`, seeds `1301457313,2012339891,1796441431,1426214042`:
  - sample127: `1W/3L`, avg diff `-1403.25`, placement `1.75`
  - This matched direct sample110 on the same set.
- 2P mini vs `sample8`, seeds `506034149,1084008127,1456205561`:
  - sample127: `2W/1L`
  - direct sample124: `3W/0L`
- 2P 8-seed vs `sample8`, seeds `506034149,1456205561,83554485,1084008127,260721277,419543473,2103738413,2124103539`:
  - sample127: `5W/3L`, avg diff `+748.38`, placement `1.38`

Submit artifact:
- `sample127_2p124_4p110_singlefile_submit_flat.zip`
- Flat contents:
  - root `main.py`
  - root `params.json`
  - root `orbit_lite/`
- This is safer than folder wrappers, but note: it is not exact `sample124` behavior in 2P because the underlying planner remains sample110-lineage with selected sample124 2P logic ported in.

Current practical read:
- 2P:
  - `sample8` itself is public-code based, and it is strong in the current real/public environment.
  - Because `sample8` is public, real matches can include same-family opponents that share the baseline but have small improvements, especially behavior changes around turn 50/100.
  - The visible 2P losses are often against those improved same-family/public-code variants, plus a smaller number of genuinely high-level opponents.
  - Therefore, do not overreact to small public-code seed sets when they conflict with the broader submission feel. `sample124` is useful research, but a combined submission must not break the strong `sample8`/`sample110` 2P baseline.
- 4P:
  - `sample110_4p_oneply_coord_from109` is currently the strongest known 4P line among our code.
  - It improved somewhat over earlier versions, but it still has a structural weakness against top players: some games end around turn 40 with our position erased before a stable base forms.
  - Future 4P work should focus on avoiding early annihilation against top-tier pressure, not merely improving average diff or public-code matchups.

## 2026-06-23 sample130: threat-aware source garrison reservation (early-defense)

Date:
- 2026-06-23

Artifact:
- `sample130_4p_threat_reserve_from110`

Parent:
- `sample110_4p_oneply_coord_from109`

Motivation (diagnosis first):
- selector_probe on the fixed 30-seed pool: `s7_stable=19, s8_burst=6, lane_anchor=4, winner_outer_domain=1`.
  - The elaborate selector + specialized modes are mostly irrelevant on this benchmark; 63% of games are decided by the shared core engine (CONFIG_4P / s7_stable). So improvement must target the core producer, not the selector.
- Failed precursor `sample129_4p_early_response_ramp_from110`:
  - Only changed `orbit_response_ramp_end` 80->45 and `orbit_response_ramp_min` 0.35->0.55.
  - 30-seed result was byte-identical to sample110 (`13W/17L`, diff `-651.37`).
  - Conclusion: the orbit response gate does not engage in the early window (its `bonus - penalty` is ~0 there), so re-scheduling a score-bonus gate does nothing. Early defense must be a hard constraint, not a score term. sample129 not adopted.

Implementation (single change, core engine):
- New `_threat_reserve_source_budget()` in `main.py`.
- Called in `plan_lite_waves` immediately after `source_budget = obs.ships.clone()`, before offensive/reserve greedy selection.
- For each owned planet, compute inbound enemy ships over `threat_reserve_window` steps from `garrison_status.arrivals_by_owner` (enemy fleets already in flight), minus my own inbound reinforcement and a partial production buffer, plus a small margin. That is the garrison that must stay home.
- HARD source-budget cut (offense physically cannot strip that planet), not a score bonus.
- Threat-triggered: zero reserve when no enemy arrivals inbound (no passivity on quiet boards).
- Savable-gated: zero reserve when even full garrison cannot hold (`reserve_need > ships`), so it never wastes ships on a doomed planet.
- Enabled only in `CONFIG_4P` -> propagates to all 4P modes. 2P unaffected (`player_count < 4` early return; `CONFIG_2P` leaves it disabled by default).
- Config: `enable_threat_reserve_4p=True, threat_reserve_window=6, threat_reserve_margin=3.0, threat_reserve_prod_frac=0.5, threat_reserve_start=0, threat_reserve_limit=200`.

Sanity:
- `py_compile`: pass.
- Crash rate in the 30-seed run below: `0.0%`.
- Behavior is genuinely changed (unlike sample129): 2-seed smoke diffs moved from sample110's `+1541 / -1221` to `+1215 / -1676`.

Fixed 30-seed pool vs `sample7 + sample8 + hairate5`, seat0:
- Seeds: `1657384936,1351844563,1351611807,53351919,457793527,1190946047,1890565501,1767874900,1625436819,1798067820,1182650381,786534061,560842328,1319582963,1438880892,1851779903,1209263198,1271979858,1458181989,200181723,638633100,1414997908,1771745689,1397683760,1297486259,157338764,1670733138,165725081,1714320719,651533336`
- sample110 baseline:
  - `13W/17L`, placement `1.60`, avg diff `-651.37`, crash `0%`.
- sample130:
  - `16W/14L`, placement `1.47`, avg diff `+46.00`, crash `0%`.
  - Improves on every agreed metric (win count, placement, early survival, crash, diff).
  - In the sample130 run no seed has `survival < 60` (minimum survival `83`), consistent with the early-annihilation target.

Out-of-sample validation (random20, fresh seeds not in the 30-seed pool):
- Seeds: `244160128,2009432271,942375866,2048305392,1561304223,1218863651,903821700,1798151692,694569277,1983603324,1317309966,3732104,1390562965,663003121,352320887,1399615834,479904572,1269357001,676735579,2018857515`
- sample110 baseline:
  - `6W/14L`, placement `1.70`, avg diff `-738.55`, crash `0%`.
- sample130:
  - `7W/13L`, placement `1.65`, avg diff `-540.65`, crash `0%`.
- Per-seed: sample130 keeps ALL of sample110's 6 wins and additionally rescues seed `942375866` (`L -1049` -> `W +2043`). No win was lost.
- Two independent seed sets now agree in direction (30-seed `+3W`, random20 `+1W`, both better placement and diff), so the gain generalizes and is not 30-seed overfit.

Decision:
- PROMOTED. `sample130_4p_threat_reserve_from110` is the new 4P development base, replacing `sample110_4p_oneply_coord_from109`.
- It is a general mechanism-level improvement (a hard early-defense source constraint shared by all 4P modes), not a per-seed score bonus.
- 2P is unchanged by this branch, so the strong sample8/sample110-style 2P baseline is preserved.

Read / next:
- The reserve is passive (keep ships home). The complementary active mechanism is still open: when `garrison_status.owner[p, 1:W]` predicts an owned high-prod/home planet will flip, inject a high-priority defensive reinforcement wave from the nearest safe neighbor before offense. Natural next branch on top of sample130.
- Also try the same threat-reserve mechanism on the 2P path as a separate experiment (must not break the sample8/sample110 2P baseline; verify with the 2P gates).
- Param note: margin=3.0 / window=6 were not tuned; any tuning should use a train/validate split, not the 30-seed pool alone.

## 2026-06-23 sample131: active defensive reinforcement -- REJECTED

Artifact: `sample131_4p_defense_reinforce_from130` (parent sample130).
Idea: pre-offense pass that reinforces owned planets the no-action projection says will flip (garrison_status.owner flips within window, prod>=2), selecting existing defensive candidates before offense drains neighbours.
Result (worse than sample130 on BOTH sets):
- Fixed 30-seed: `15W/15L`, plc `1.50`, diff `-237.53` (sample130 `16W/14L`, plc `1.47`, diff `+46.00`).
- random20: `5W/15L`, plc `1.75`, diff `-986.20` (sample130 `7W/13L`, plc `1.65`, diff `-540.65`).
Read / lesson:
- A 2-seed smoke looked great (rescued 53351919) but the full sets regressed: forcing defensive reinforcement before offense over-defends and goes passive, losing previously-won seeds (e.g. random20 942375866 W->L).
- Confirms a general rule for this engine: **hard constraints that only PREVENT bad commitments help (sample130 reserve); mechanisms that FORCE actions or ADD score hurt (sample131 forced defense, plus historical broad bonuses / response-search).** Future defensive work should subtract options, not inject forced actions.
- sample131 not adopted. sample130 remains the 4P base.

## 2026-06-23 sample132/133/134 + early-death trace -- all NEUTRAL/NO-OP, root cause found

Base remains `sample130`. Three further defensive variants on top of it, all on the fixed 30-seed + random20 (or in-session trace):
- `sample132_4p_combat_aware_reserve_from130`: reserve threat = (top1_enemy - top2_enemy) instead of sum (enemies fight each other first). WASH: 30-seed `16W` plc `1.47` diff `+174` (better diff), random20 `7W` plc `1.65` diff `-595` (slightly worse). Same win count both sets. More-correct threat model; shelved, not adopted.
- `sample133_4p_brawl_avoid_from130`: hard-forbid attacking an enemy planet a third-party enemy is already attacking. NEAR NO-OP: identical to sample130 on 3 of 4 diagnosed early-death seeds. Hypothesis (we feed contested enemies) was wrong. Rejected.
- `sample134_4p_latent_reserve_from130`: add cheap_enemy_pressure (reachable enemy garrison mass) to the reserve threat to cover not-yet-launched pressure. NO-OP on the trace even at latent reach 14. Rejected.

Root-cause trace (tools/trace_4p.py, per-turn planets/ships, seed 1399615834, sample130):
- t<=32: all four players symmetric and even (3 planets each; we are actually AHEAD on ships, 112 vs 89).
- t32->t40: WE voluntarily launch ~37 ships (112->75sh) and drop 3->1 planets, while two opponents cleanly expand 3->5 planets. By t52 we are eliminated.
- Interpretation: the early death is NOT a defense/reserve failure (total ships stay flat while planets are lost). It is an OPENING EXPANSION / target-quality failure: our shared core (s7_stable, 63% of the 30-seed pool) over-commits launches that fail or get sniped, while the real sample7 expands cleanly and holds. Defensive reserves (sample130/132/134) and capture filters (sample133) cannot fix "we attack the wrong targets."
- Consequence for next work: the high-value lever is opening offensive target selection / launch sizing quality (build a holdable cluster, avoid failed over-commit), which is deeper and higher-regression-risk (cf. sample131). It partly lives in orbit_lite (build_target_shortlist / score_candidates) which must not be broken. Recommend a careful, well-gated experiment here, validated on both seed sets, rather than more defensive micro-constraints (diminishing returns confirmed).

Net status: `sample130_4p_threat_reserve_from110` is the confirmed improved 4P base (13->16W on 30-seed, 6->7W on random20, placement and diff up, crash 0%, 2P untouched). sample132 shelved (correct but neutral). sample131/133/134 rejected.

## 2026-06-23 sample135: opening neutral focus (blanket no-enemy-attack) -- REJECTED, but decisive signal

Artifact: `sample135_4p_opening_neutral_focus_from130`. Hard-forbid all enemy-target captures before turn 60 in 4P.
Trace (seed 1399615834): collapse GONE -- held 3 planets at t40 (vs sample130 1 planet) and was alive 4p/227sh at t60 (vs dead t52).
But full eval REGRESSED hard:
- 30-seed: `5W/25L`, plc `1.83`, diff `-1402` (sample130 `16W`).
- random20: `4W/16L`, plc `1.80`, diff `-1236` (sample130 `7W`).
Decisive signal:
- Removing enemy attacks costs ~11 wins. Enemy attacking is NET POSITIVE; only SOME enemy attacks (premature / un-holdable over-commit) are bad.
- The blanket ban also threw away the winning vulture captures (taking weakened/cheap enemy planets), so we survived but never won.
- Correct lever (confirmed by both the trace and this regression): distinguish GOOD enemy captures (holdable, weakened target = vulture) from BAD ones (un-holdable over-commit), and forbid only the bad ones -- i.e. a holdability / "can I hold it?" supply filter, not a time-based ban.
- sample135 rejected. sample130 remains the base. Next: sample136 = holdability filter on enemy captures (allow vulture, forbid un-holdable), trace-validated before any heavy eval.

## 2026-06-23 sample136: enemy-capture holdability filter -- REJECTED

Artifact: `sample136_4p_enemy_holdability_from130`. Forbid only enemy captures whose target cannot be held (combat-aware net enemy arrivals in the post-capture window > surviving + reinforce + prod + margin).
Targeted check (4 seeds): kept 1438880892 (W) but turned 165725081 from W->L (banned a capture that was part of sample130's winning line), and did NOT fix the 1399615834 death (survival still 48, same as sample130).
Decisive read:
- sample135 (ban ALL enemy attacks) fixed the 1399615834 death; sample136 (ban only un-holdable enemy captures) did not. So the t32-40 death is NOT target un-holdability -- the launches passed the holdability check. The real cause is SOURCE-STRIPPING: we empty our home to attack a (holdable) enemy, and a different enemy then takes the emptied home. The filter looks at the target, not the source.
- sample130's reserve protects the source only against fleets already in flight, not against the latent nearby enemy garrison that launches after we empty the planet.
- Conclusion confirmed across sample131-136 (six straight non-improvements): bolt-on tactical filters on top of an already-rich engine (true_one_ply / gates / reserve) are at/below a local optimum. The remaining gains require a STRUCTURAL change: a first-class supply analysis (Idea B) that scores the COST of weakening a source against all reachable enemy force, not just in-flight fleets.
- sample136 rejected. sample130 remains the 4P base. Proceeding to Idea B (integrated supply analysis).

## 2026-06-23 sample137: supply analysis / latent reachable-enemy reserve -- REJECTED; reserve direction EXHAUSTED

Artifact: `sample137_4p_supply_reserve_from130`. Extend the source reserve with a latent reachable-enemy term (strongest single enemy planet that could launch and reach an owned planet within horizon, from cross_dist + fleet speed), combined with the in-flight term via MAX.
Result (worse than sample130 on BOTH sets):
- 30-seed: `12W/18L`, plc `1.63`, diff `-836` (sample130 `16W`, plc `1.47`, diff `+46`).
- random20: `6W/14L`, plc `1.70`, diff `-733` (sample130 `7W`, plc `1.65`, diff `-541`).
Read: latent reserve over-reserves -> passivity -> fewer wins (same failure mode as sample131/135). Trace on the symmetric brawl seed showed it held ships at t36 but still lost all planets by t44; that seed is a 4-way coinflip not worth tuning to.

Summary of the whole 4P micro campaign (sample131-137, seven straight non-improvements over sample130):
- WHAT WORKS: a precise, threat-triggered, savable-gated DEFENSIVE CONSTRAINT that only prevents over-draining (sample130). +3W/30-seed, +1W/random20, generalised.
- WHAT FAILS: forcing actions (sample131 reinforce), banning options broadly (sample135 no-enemy-attack -> -11W), and any over-reservation (sample134/137 latent) -> passivity; and target-side holdability filters (sample133/136) miss the real cause and ban winning captures.
- ROOT CAUSE of the residual early deaths (trace-confirmed): SOURCE-STRIPPING in symmetric 4-way brawls -- we empty a home to attack and a different enemy takes it. The engine is already near a local optimum for bolt-on tactics (true_one_ply / gates / safe_drain / reserve); single-function additions cannot beat it.
- CONCLUSION: stop 4P bolt-on micro. The only remaining 4P upside is a STRUCTURAL rewrite (Idea A: replace the per-wave greedy `_greedy_select` with a global min-cost-flow / assignment allocation), which is a large, higher-risk, multi-step project (and must be hand-rolled in torch/numpy since scipy is not in the submission env). Alternatively, pivot to the untouched 2P axis (Ideas I/J) for fresher, lower-risk gains.

CONFIRMED DELIVERABLE: `sample130_4p_threat_reserve_from110` is the validated improved 4P base (13->16W 30-seed, 6->7W random20, placement+diff up, crash 0%, 2P untouched, flat-submittable). sample131/133/134/135/136/137 rejected; sample132 shelved (neutral, more-correct threat model).
