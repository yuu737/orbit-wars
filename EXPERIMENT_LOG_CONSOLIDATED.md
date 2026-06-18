# Orbit Wars Consolidated Experiment Log

Updated: 2026-06-17

This file is the high-signal experiment history for handoff to another agent.
It compresses the older `main_v*`, `hairate*`, and recent `sample*` work into
decisions, evidence, and next actions.

## Contest Context

Competition: Kaggle **Orbit Wars**

Game:

- Real-time strategy game in continuous 2D space.
- 2-player and 4-player matches.
- Planets rotate around a central sun.
- Agents launch fleets from owned planets to capture neutral/enemy planets.
- Owned planets produce ships each turn.
- Good agents need future position prediction, launch sizing, target selection,
  defense/reinforcement, and 2P/4P-specific strategy.

Current target:

- Push public score toward `1300`.
- Recent best mixed public score from this workspace: `1204.1`.
- Win rate matters more than average score diff.
- Treat exact local `diff=0` as draw for analysis, because seat ordering can
  produce misleading local rewards.

## Current Best Known State

Safe public candidate:

- `sample7_4p_sample8_2p_submit_v3.zip`
- Public score: `1204.1`
- Behavior: 2P uses `sample8`, 4P uses `sample7`.

Clean local baseline:

- `sample11_s8_baseline_2p_s7_4p`
- 2P = `sample8`
- 4P = `sample7`

Important warning:

- Folder-based selector/wrapper can change behavior versus running the underlying
  sub-agent directly.
- Future serious submissions should prefer one loaded package / one `main.py` /
  one `orbit_lite`, not dynamic switching between separate `sample7` and
  `sample8` folders.

## Public Score References

Recent visible public scores:

- `sample7_4p_sample8_2p_submit_v3.zip`: `1204.1`
- `sample7.zip`: `1121.7`
- `sample8.zip`: `1117.2`
- `hairate5.zip`: about `1109.3`
- `hairate7.zip`: about `1119.0`
- `V10.2.zip`: about `1189.2`
- older `hairate2.zip`: high variance, including one run around `1296.1`

Interpretation:

- Local improvements have not reliably converted to public score.
- Stable submission structure and mode-specific behavior matter.
- `sample7/sample8` are stronger public bases than most recent `hairate` variants.

## Major Strategic Conclusions

### 1. `orbit_lite` should be kept as the engine

Observation:

- `sample7/orbit_lite` and `sample8/orbit_lite` are almost identical.
- Only meaningful file difference found: `orbit_lite/planner_core.py`.
- The difference is mainly safer player-count inference in `sample8`.

Decision:

- Do not rewrite physics, fleet flow, movement, or capture simulation.
- Use `orbit_lite` as the stable engine.
- Improvements should live in planning/candidate generation/stateful strategy
  above the engine.

### 2. Small score bonuses have not been enough

Repeated pattern:

- Adding target bonuses, penalties, anchor boosts, or broad response penalties
  often failed or disrupted an already strong base.

Examples:

- `sample9`: broad response search on `sample8`, bad.
- `sample10`: targeted response search on `sample8`, still bad.
- `sample12/13/14/15`: route/phase/counter-capture/anchor ideas, not enough.
- `sample18/19/20`: anchor/route attempts did not become promotion candidates.

Decision:

- Stop expecting small score tweaks to reach `1300`.
- Future work should change action generation and strategic state, not just
  tweak candidate scores.

### 3. 2P `sample8` is not always the best base

Key local discovery:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent sample7\main.py --opponent sample8\main.py --games 10 --both-seats --workers 10 --seed-start 5522554122
```

Result:

- `sample7` beat `sample8`: `11/20`, `55.0%`.

Related:

- Reverse was `sample8` vs `sample7`: `9/20`, `45.0%`.
- `sample7` vs `sample7` is symmetric as expected.

Decision:

- Do not assume `sample8` must be the 2P base.
- The next high-upside 2P path should use `sample7`/`orbit_lite` as an engine
  and add an independent stateful opening policy.

### 4. 4P winners build thick anchors/clusters

Replay observation from public games:

- Winning 4P agents do not just scatter to many small planets.
- They create a thick production cluster or anchor.
- After the anchor is strong, they expand to outer/central large planets.
- They avoid getting trapped in small local fights while another player grows a
  safe outer cluster.

Decision:

- 4P should aim to create first-place lines, not only survive.
- Good future 4P work should be an anchor/route planner with state, not a one-turn
  target bonus.

### 5. Broad anti-runaway did not work

Branches:

- `hairate39_4p_anti_runaway.py`

Result:

- It often behaved exactly like baseline or failed to meaningfully affect runaway
  games.
- Abstract "leader" suppression was too broad and too late.

Decision:

- If revisiting anti-runaway, detect enemy completed safe clusters, not just the
  current score leader.
- Prefer blocking/contesting neutral boundary planets over blindly attacking the
  leader.

## Branch Status Summary

### Safe / Useful

- `sample7`
  - Strong 4P public component.
  - Surprisingly competitive 2P component on some seed blocks.

- `sample8`
  - Strong 2P public component.
  - Not universally stronger than `sample7`.

- `sample11_s8_baseline_2p_s7_4p`
  - Clean local baseline.
  - 2P `sample8`, 4P `sample7`.

- `sample7_4p_sample8_2p_submit_v3.zip`
  - Best recent public mixed submission: `1204.1`.

### Research Only

- `sample12_2p_opening_route_commit_s7_4p`
  - Opening route candidate additions.
  - No clear win-rate lift.

- `sample13_2p_phase_controller_s7_4p`
  - Phase controller idea.
  - No clear win-rate lift.

- `sample14_2p_counter_capture_candidates_s7_4p`
  - Counter-capture candidate generation.
  - No clear promotion evidence.

- `sample15_s8_2p_s7_4p_anchor`
  - 4P anchor attempt.
  - Not clearly better.

- `sample17_s8_dynamic_roi_late_2p_s7_4p`
  - Imported `sample6_17`-style dynamic ROI/late ideas.
  - Short checks did not justify promotion.

- `sample18_planned_anchor_route`
  - 4P planned anchor route.
  - Similar to baseline on short checks; not promoted.

- `sample20_2p_s7_anchor_route_s7_4p`
  - Revealed wrapper/direct behavior issue.
  - Direct sub-agent beat `sample8`, wrapper did not.

- `sample21_2p_s7_counter_capture_s7_4p`
  - `sample7` base with 2P contested-neutral / thin-anchor ideas.
  - Some effect vs `sample8`, but hurt `sample7` mirror.
  - Research only.

### Frozen / Do Not Promote

- `sample9_s8_response_2p_s7_4p`
  - Broad response search hurt `sample8`.

- `sample10_s8_targeted_response_2p_s7_4p`
  - Targeted response search still hurt or failed to improve.

- `sample19_2p_anchor_route_s7_4p`
  - 2P anchor route on `sample8` hurt vs `sample7`.

- `hairate31_4p_home_sector_adaptive.py`
  - Adaptive home-sector idea got worse in tested block.

- `hairate33/34/35` cluster gates/shortlist variants
  - Attempts to gate or soften `hairate32` cluster behavior failed to preserve
    the good blocks.

## Hairate Line Summary

Early useful findings:

- `hairate5` was the strongest local `hairate` reference for a while.
- `hairate7` and `hairate8_w20` explored stronger local behavior.
- `hairate14_response_search.py` introduced enemy response search; useful as a
  concept, but later sample-based versions did not transfer cleanly.
- `hairate30_2p_h14_4p_h29.py` combined 2P/4P behavior.
- `hairate38_safe_strategy_selector.py` became a safer 4P selector-style research
  baseline.
- `hairate41_4p_opening_route_planner.py` explored route planning.
- `hairate42_4p_anchor_planner.py` was planned as the next anchor branch, but work
  paused when stronger `sample7/sample8` public evidence arrived.

Decision:

- `hairate` branches are useful for ideas, but current public direction should
  prioritize `sample7/sample8` and `orbit_lite`.

## Detailed Recent Experiments

### `sample9_s8_response_2p_s7_4p`

Goal:

- Add `hairate14`-style enemy response search to `sample8` 2P.

Result:

- Lost to improved baselines.
- Broad down-scoring disrupted `sample8`'s good timing.

Decision:

- Freeze.

### `sample10_s8_targeted_response_2p_s7_4p`

Goal:

- Make response search narrower and safer.

Result:

- Still did not beat baseline.

Decision:

- Freeze response-search-on-sample8 direction.

### `sample12_2p_opening_route_commit_s7_4p`

Goal:

- Add opening route targets to `sample8` 2P.

Result:

- Mostly baseline-like; no decisive lift.

Decision:

- Keep as reference only.

### `sample13_2p_phase_controller_s7_4p`

Goal:

- Add phase controller for early/mid/late behavior.

Result:

- No clear lift in gates.

Decision:

- Research only.

### `sample14_2p_counter_capture_candidates_s7_4p`

Goal:

- Add concrete counter-capture candidates.

Result:

- Did not clearly beat baseline.

Decision:

- Research only.

### `sample18_planned_anchor_route`

Goal:

- Add 4P anchor-route behavior on top of sample stack.

Results:

- 4P smoke vs `hairate5`, seed `12000000`, 3 games: `2W/1L`, crash `0`.
- 2P smoke vs `sample8`, 3 games both seats: crash `0`.
- 4P seed `56000000`, 5 games vs `hairate5`: `1W/4L`; baseline showed the same
  short-block result.

Decision:

- Not promoted.

### `sample19_2p_anchor_route_s7_4p`

Goal:

- Add 2P anchor/route planner to `sample8`.

Results:

- vs `sample8`: `10/20`, `50%`.
- vs `sample7`: `8/20`, `40%`.

Decision:

- Freeze.

### `sample20_2p_s7_anchor_route_s7_4p`

Goal:

- Test `sample7`-style 2P under a wrapper.

Important results:

- Direct `sample20_...\sample8\main.py` vs `sample8`: `11/20`, `55%`.
- Wrapper `sample20_...\main.py` vs `sample8`: `10/20`, `50%`.

Decision:

- Main value was discovering selector/wrapper risk.
- Do not rely on dynamic folder import for final high-score candidates.

### `sample21_2p_s7_counter_capture_s7_4p`

Goal:

- Use `sample7` as 2P base.
- Add contested-neutral and thin-anchor punish ideas.
- Exploit the fact that many `orbit_lite` agents have similar first ~20 turns.

Implementation:

- Copied `sample7`.
- Added 2P-only contested-neutral target augmentation.
- Added small early bonus against thin productive enemy anchors.
- 4P remains `sample7`.

Short results:

- Syntax: passed.
- 4P smoke vs `hairate5`, seed `12000000`, 1 game: win `1/1`, crash `0`.
- 2P vs `sample8`, seeds `5522554122-5522554125`, both seats: `2/8`.
- Baseline `sample7` vs `sample8` on same seeds: `1/8`.
- 2P vs `sample7`, same seeds, both seats: `3/8`.
- Baseline `sample7` vs `sample7` on same seeds: `4/8`.

Decision:

- The mirror weakness idea is real enough to study.
- This implementation hurts `sample7` mirror and is not promoted.
- Next attempt should be stateful opening policy, not another small bonus.

### `sample22_2p_stateful_opening_policy_s7_4p`

Goal:

- Move beyond small bonuses by giving 2P a persistent opening policy.
- Use `sample7` and `orbit_lite` as the engine.
- Choose one opening mode early:
  - `race`
  - `delayed_retake`
  - `alternate_route`
- Add the planned route targets to the shortlist and lightly boost candidates
  that follow the chosen plan.
- Keep 4P as `sample7`.

Implementation:

- Copied `sample7`.
- Added `opening_policy` to `ProducerLiteMemory`.
- Added 2P-only route classification and policy targets.
- Final launch sizing, reachability, exact scoring, and greedy selection remain
  handled by `orbit_lite`.

Short results:

- Syntax: passed.
- 4P smoke vs `hairate5`, seed `12000000`, 1 game: win `1/1`, crash `0`.
- 2P smoke vs `sample8`, seeds `5522554122-5522554123`, both seats: `1/4`.
- This matches the known `sample7` baseline behavior on the same mini-block.
- 2P smoke vs `sample7`, same seeds, both seats: `2/4`, symmetric.

Conclusion:

- It is safe and does not obviously break `sample7`.
- It does not yet change the 2P opening enough to create lift.
- Next version should make the policy more binding: if the chosen mode is
  `alternate_route`, suppress the bad contested race; if `delayed_retake`, time
  the follow-up more explicitly instead of merely adding the target to shortlist.

### `sample23_4p_anchor_line_planner_s8_2p_s7_4p`

Goal:

- Shift focus back to 4P because current public gap appears mostly 4P-related.
- Use `sample7 + orbit_lite` as the engine.
- Add a 4P-only stateful anchor line planner:
  - choose one early anchor,
  - add anchor neighborhood targets to shortlist,
  - softly boost the route,
  - avoid draining the anchor before it is thick,
  - after maturity, allow outer large targets.

Implementation:

- Copied `sample7`.
- Added `anchor_plan` to memory.
- Added 4P-only `anchor_targets`, route/outer targets, and soft score adjustment.
- 2P is still effectively `sample7` in this branch despite the branch name.

Short results:

- Syntax: passed.
- 2P smoke vs `sample8`, seed `5522554122`, both seats: `1/2`, crash `0`.
- 4P smoke vs `hairate5`, seed `12000000`, 2 games: `2/2`, crash `0`.

Paired short checks vs `sample7`:

`56000000-56000004`, vs `hairate5`, 5 games:

- `sample23`: `2W/3L`, avg diff `-410.20`
- `sample7`: `1W/4L`, avg diff `-1024.80`
- Improvement: `+1 win / 5`, better average diff.

`5421622-5421626`, vs `hairate5`, 5 games:

- `sample23`: `1W/2D/2L`, avg diff `-3332.60`
- `sample7`: `1W/2D/2L`, avg diff `-3333.20`
- Essentially unchanged.

Conclusion:

- This is the first recent 4P branch with a small positive local signal without
  obvious breakage.
- It is not yet a promotion candidate; sample size is too small and 4P remains
  weak.
- Next improvement should make the anchor plan more observable/diagnosable:
  record or print anchor id/route for selected test seeds, then tune whether the
  chosen anchor is actually the one a strong replay would pick.

### 2026-06-17: 4P base candidate pool check

Command:

```powershell
.\run_4p_candidate_eval.ps1 -Games 3 -Workers 6 -SeedStart 56000000 -Mode pool
```

Pool opponents:

- `sample7`
- `sample8`
- `hairate5`

Candidates:

- `sample7`
- `sample8`
- `hairate5`
- `sample23_4p_anchor_line_planner_s8_2p_s7_4p`

Result:

- `sample7`: `3W/0D/0L`, avg diff `+1648.67`
- `sample23`: `2W/0D/1L`, avg diff `+196.00`
- `sample8`: `1W/0D/2L`, avg diff `-378.67`
- `hairate5`: `1W/0D/2L`, avg diff `-2329.00`

Conclusion:

- On this short pool test, `sample7` is clearly the strongest 4P base.
- `sample23` is not better than `sample7` here; its anchor overlay changed one
  seed negatively.
- Next 4P work should use `sample7` as the baseline and only keep changes that
  beat `sample7` directly in pool-style evaluation.

Follow-up command:

```powershell
.\run_4p_candidate_eval.ps1 -Games 10 -Workers 10 -SeedStart 56000000 -Mode pool
```

Pool opponents:

- `sample7`
- `sample8`
- `hairate5`

Result:

- `sample7`: `6W/1D/3L`, avg diff `+226.50`, avg place `1.30`
- `sample23`: `4W/1D/5L`, avg diff `-407.80`, avg place `1.50`
- `sample8`: `4W/1D/5L`, avg diff `-608.60`, avg place `1.50`
- `hairate5`: `1W/4D/5L`, avg diff `-2901.30`, avg place `1.80`

Updated conclusion:

- `sample7` is the current best 4P base among the tested candidates.
- `sample23` anchor overlay is rejected for now because it underperforms plain
  `sample7` on the same pool block.
- `sample8` and `hairate5` are useful benchmark/opponent components, but not the
  current 4P base.

Second block:

```powershell
.\run_4p_candidate_eval.ps1 -Games 10 -Workers 10 -SeedStart 12000000 -Mode pool
```

Result:

- `sample7`: `4W/0D/6L`, avg diff `-2958.40`, avg place `1.70`
- `sample23`: `4W/0D/6L`, avg diff `-3179.30`, avg place `1.80`
- `hairate5`: `4W/0D/6L`, avg diff `-1923.80`, avg place `1.90`
- `sample8`: `1W/0D/9L`, avg diff `-4615.40`, avg place `2.10`

Interpretation:

- Win count is tied between `sample7`, `sample23`, and `hairate5`.
- `sample7` has the best average placement, which matters for 4P rank-style play.
- `hairate5` has better average diff and survival, but not better win count or
  placement in this pool.
- `sample23` again does not beat `sample7`.

Updated base decision:

- Keep `sample7` as the 4P base for now.
- Keep `hairate5` as a style reference / benchmark because it survives longer and
  loses less catastrophically in some blocks.
- Do not continue `sample23` as-is; if using anchor ideas, re-implement them more
  conservatively on top of `sample7` and require direct pool wins over `sample7`.

### `sample24_4p_s7_prize_targets` and `sample25_4p_s7_prize_targets_candidate_only`

Goal:

- Follow the "first place only matters" objective.
- Use `sample7` as the 4P base.
- Add high-value swing/prize targets in 4P so the agent sees reachable high-prod,
  large neutral, and thin enemy productive planets that proximity shortlist may
  miss.

Implementation:

- `sample24`: adds prize targets and a small score bonus.
- `sample25`: candidate-only version; prize targets are added, but no direct score
  bonus is applied.
- Both keep 2P effectively as `sample7`.

Short checks:

`56000000-56000004`, pool opponents `sample7/sample8/hairate5`:

- `sample24`: `4W/0D/1L`, avg diff `+795.80`, avg place `1.20`
- `sample25`: `4W/0D/1L`, avg diff `+795.80`, avg place `1.20`
- `sample7` reference from full block: first five were also `4W/0D/1L`, avg diff about `+700`

`12000000-12000004`, same pool:

- `sample24`: `2W/0D/3L`, avg diff `-350.80`, avg place `1.60`
- `sample25`: `2W/0D/3L`, avg diff `-350.80`, avg place `1.60`
- `sample7` reference first five: `2W/0D/3L`; sample7 had a much worse catastrophe
  on seed `12000004`, but still the same win count.

Remaining checked halves:

- `sample24` on `56000005-56000009`: `2W/1D/2L`, avg diff `-334.60`
- `sample24` on `12000005-12000009`: `1W/0D/4L`, avg diff `-2699.00`

Conclusion:

- Prize targets improve some score-diff/catastrophe behavior but do not improve
  first-place count over `sample7` in the checked blocks.
- Because the user explicitly prioritizes 1st place, neither `sample24` nor
  `sample25` should be promoted yet.
- The next 4P improvement should be based on loss replay/seed diagnosis for
  sample7's actual losing seeds, especially `12000004`, `12000005`, `12000006`,
  and `12000009`, rather than adding more generic targets.

## Evaluation Commands

2P same-family gate:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent <agent> --opponent sample8\main.py --games 10 --both-seats --workers 10 --seed-start 5522554122
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent <agent> --opponent sample7\main.py --games 10 --both-seats --workers 10 --seed-start 5522554122
```

2P stronger local benchmark:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent <agent> --opponent bots\hairate5.py --games 40 --both-seats --workers 10 --seed-start 65122554122
```

4P gates:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent <agent> --opponent bots\hairate5.py --games 10 --workers 10 --seed-start 12000000
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent <agent> --opponent bots\hairate5.py --games 10 --workers 10 --seed-start 56000000
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent <agent> --opponent bots\hairate5.py --games 10 --workers 10 --seed-start 5421622
```

## Recommended Next Work

### Highest-upside direction

Build a new branch:

- Proposed name: `sample22_2p_stateful_opening_policy_s7_4p`

Architecture:

- Use `orbit_lite` as the engine.
- Use `sample7` as the initial code base.
- Add independent 2P stateful opening policy.
- Keep 4P as `sample7` initially.

Core idea:

- Do not merely add target bonuses.
- At step `0-20`, classify the first contested neutral situation.
- Explicitly choose one opening mode:
  - `race`: we can arrive first or near-first, so commit.
  - `delayed_retake`: enemy arrives first, so plan a follow-up capture.
  - `alternate_route`: race is bad, so avoid wasting ships and go elsewhere.

Why this is different:

- It changes the action plan, not just the score.
- It directly targets `orbit_lite` mirror behavior.
- It may preserve `sample7`'s 2P strength better than broad candidate inflation.

### Submission-structure direction

Build an integrated submission:

- One `main.py`.
- One `orbit_lite`.
- No dynamic folder selector.

Reason:

- Wrapper/direct behavior mismatch has already been observed.
- A single package reduces Kaggle validation and runtime import risk.

### 4P direction after 2P

If returning to 4P:

- Do not start with broad anti-runaway.
- Build a stateful anchor/route planner:
  - choose an anchor,
  - thicken it,
  - expand from it to large planets,
  - avoid thin scattered overexpansion.

## Do Not Repeat Soon

- Broad `2-ply` response penalties on `sample8`.
- Small generic target bonuses without state.
- Folder-based dynamic `sample7/sample8` import for final submissions.
- Promoting branches based only on average score diff.
- Overreacting to `score=0`; win rate is the priority.

## 2026-06-18 4P Replay-Driven Experiments

### Replay generation

Added `make_replay_html.py` to generate Kaggle-style replay HTMLs for selected losing seeds.

Generated sample7 pool-loss replays:

- `12000004`
- `12000005`
- `56000004`
- `56000006`

Output folder:

- `research_runs/replays_sample7_pool_loss_20260618_011512`

### Visual diagnosis from seed 12000004

Observed failure pattern:

- The winner forms a thick outer/upper anchor by roughly `step 80-120`.
- Our bot does not create a useful connection point to contest that anchor.
- By `step 124`, the winner has a `500+` ship anchor and the game is effectively decided.
- This is not an endgame issue; the critical window is the opening-to-midgame transition.

### Branches tried

`sample26_4p_enemy_anchor_blockade_s8_2p_s7_4p`

- Idea: detect enemy thick anchor in `step 55-130`, add nearby blockade/border targets, and apply a small bonus.
- Result on seeds `12000004,12000005,56000004,56000006`: `0W/4L`.
- Problem: too aggressive; spending into enemy anchor/border accelerates collapse.
- Status: freeze, do not promote.

`sample27_4p_enemy_anchor_blockade_neutral_only`

- Idea: conservative version; only add neutral planets near enemy anchors, no score bonus.
- Result on same 4 seeds: `0W/4L`, almost identical to sample7.
- Problem: enemy anchor detection after formation is too late; candidate-only overlay is not enough.
- Status: freeze, do not promote.

`sample28_4p_opening_connector_s7`

- Idea: early route layer, add/boost neutral connector planets that move our anchor toward the board center.
- Initial candidate-only version: effectively identical to sample7.
- Stronger route bonus version: worsened `12000004` into early collapse.
- Problem: generic center progress is not a valid 4P strategy by itself; it breaks sample7's survival balance.
- Status: freeze, do not promote.

### Updated lesson

The visual diagnosis is still useful, but the implementation must not be a generic target bonus.

The needed next step is a **stateful 4P route policy**:

- Choose a concrete opening route at `step 0`.
- Commit only if the route has a local anchor and enough nearby support planets.
- Do not switch to generic center movement.
- Do not attack enemy anchors directly unless already backed by an owned connected anchor.

In short:

- Enemy anchor detection after the fact is too late.
- Generic central connector bonus is too crude.
- The next credible 4P improvement must choose a full route/anchor plan, not add broad per-target bonuses.

### `sample29_4p_stateful_domain_planner`

Goal:

- Build the first original 4P branch that chooses a stateful domain plan at game start.
- Keep `sample7` as the physics/scoring engine.
- Add a 4P-only `domain_plan` with anchor, support, connector, expansion, confidence, and phase.

Implementation:

- Copied `sample7` into `sample29_4p_stateful_domain_planner`.
- Added `_make_domain_plan_4p`, `_domain_plan_targets`, `_append_domain_targets`, and `_apply_domain_score_adjustment`.
- Added `domain_plan` to `ProducerLiteMemory`.
- Added `sample29_stateful_domain` to `run_4p_candidate_eval.ps1`.

Checks:

- Syntax: passed.
- 2P smoke vs `sample8`, 2 seeds both seats: no crash.

4P important losing seeds:

- Command seeds: `12000004,12000005,56000004,56000006`
- Result: `0W/0D/4L`, crash `0`.
- Compared with sample7, behavior is mostly unchanged, with a small improvement on `56000006` score diff.

4P pool results:

- `56000000-56000009`: `6W/1D/3L`, avg place `1.30`.
- This matches sample7's known win/placement on the same block, with slightly lower avg diff.
- `12000000-12000009`: `4W/0D/6L`, avg place `1.70`.
- This matches sample7's known win/placement on the same block, with slightly lower avg diff.
- `5421622-5421631`: `4W/2D/4L`, avg place `1.40`.
- Direct sample7 reference on same pool block: `3W/2D/5L`, avg place `1.50`.

Conclusion:

- `sample29` is the first stateful domain branch with a small positive win-count signal in one block while not breaking the two known pool blocks.
- It is not yet a submission candidate; it needs at least one more 30-50 game pool comparison.
- Next improvement should make the selected domain observable for replay/debugging, then tune plan selection rather than adding broad target bonuses.
