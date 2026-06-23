# Recent Orbit Wars Experiment Log

Updated: 2026-06-19

This file summarizes the recent `sample` / `hairate` experiments that were not
covered by the older `EXPERIMENT_LOG.md`.

## Current Best Submission Context

Public leaderboard references from recent submissions:

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
