# Orbit Wars Handoff

Updated: 2026-06-19

Read this first when taking over the project. For the best compressed experiment
history, read `EXPERIMENT_LOG_CONSOLIDATED.md`. For rawer recent notes, also read
`EXPERIMENT_LOG_RECENT.md`.

## Contest Overview

We are working on Kaggle's **Orbit Wars** competition.

Orbit Wars is a real-time strategy game for **2 or 4 players**:

- The board is a continuous `100 x 100` space with a sun at the center.
- Players start from home planets and send fleets to capture neutral/enemy planets.
- Planets may orbit around the sun, so future positions matter.
- Fleets travel in straight lines and can be destroyed by the sun, bounds, or planet collisions.
- Owned planets produce ships each turn.
- A match lasts up to `500` turns.
- The winner is the player with the most total ships on planets plus fleets.

Key engineering challenge:

- Good agents need fast future simulation, good target generation, launch sizing, and mode-specific strategy.
- 2P and 4P behave differently enough that separate logic is useful.

Current goal:

- Push public score toward `1300`.
- Current strong public mixed submission is around `1204`.
- We are not trying to overfit one seed block; win rate matters more than average score diff.

## Current State

Safe baseline:

- Directory: `sample11_s8_baseline_2p_s7_4p`
- Zip: `sample11_s8_baseline_2p_s7_4p.zip`
- Behavior:
  - 2P uses `sample8`
  - 4P uses `sample7`

Best recent public submission:

- `sample7_4p_sample8_2p_submit_v3.zip`
- Public score: `1204.1`
- This is currently the safest proven submission from the recent batch.

Important caution:

- The current folder-based selector approach can change behavior compared with running the sub-agent directly.
- If making a serious final submission, prefer integrating logic into one loaded package/main instead of dynamically switching folders and clearing `orbit_lite`.

Current 4P research line:

- Main direction: initial-board classification + single-package selector.
- Best current research branch: `sample32_singlefile_s7_s8_selector`.
- Follow-up selector branch: `sample34_singlefile_s7_s8_selector_tuned`.
- Oracle-ready branch: `sample35_singlefile_oracle_selector`.
- Frozen branch: `sample33_singlefile_contested_selector`.

Key current idea:

- Use one shared `orbit_lite` package, then switch between `sample7`-style stable mode and `sample8`-style burst mode inside one `main.py`.
- Do not use dynamic folder imports for final candidates.
- Use initial board features, not seed IDs, to choose the mode.

## Important Files

- `README.md`
  - Contest/game rules.
- `EXPERIMENT_LOG_RECENT.md`
  - Recent experiment summary and decisions.
- `sample11_s8_baseline_2p_s7_4p/`
  - Clean current baseline.
- `sample7/`
  - Strong 4P component and surprisingly useful 2P component.
- `sample8/`
  - Strong 2P component, public `1117.2`.
- `evaluate.py`
  - Local evaluation script.
- `bots/hairate5.py`
  - Main strong local benchmark opponent.
- `analyze_4p_losses.py`
  - Reads evaluate logs or seed lists and writes loss-focused initial-board features plus HTML previews.
- `build_4p_oracle_dataset.py`
  - Runs/collects candidate evaluations on identical 4P seeds and builds oracle CSVs/rule suggestions.
- `dump_initial_boards.py`
  - Dumps raw initial boards, feature CSV, and browser preview HTML.
- `cluster_initial_boards.py`
  - Fast initial-board clustering for large seed ranges.

## Recent Decisions

Current 4P selector decisions:

- `sample32_singlefile_s7_s8_selector`
  - Promising research branch.
  - Uses sample8's `orbit_lite` as the single shared engine.
  - Reproduced sample8's strong seeds `9874600005/9874600006`.
  - Local pool results observed:
    - `56000000`: `6W / 1D / 3L`
    - `12000000`: `4W / 6L`
    - `9874600000`: `5W / 5L`
    - `5421622`: `2W / 2D / 6L`
    - `2000000`: `6W / 4L`

- `sample33_singlefile_contested_selector`
  - Frozen.
  - Reserve-based contested mode did not improve the `5421622` block.

- `sample34_singlefile_s7_s8_selector_tuned`
  - Research candidate to tighten `sample8` selection and return more seeds to stable mode.
  - Needs full evaluation.

- `sample35_singlefile_oracle_selector`
  - Same single-package direction as `sample34`.
  - Can read optional `oracle_rules.json`.
  - Without rules, it falls back to the safe tuned selector behavior.

Do not promote:

- `sample9_s8_response_2p_s7_4p`
- `sample10_s8_targeted_response_2p_s7_4p`
- `sample19_2p_anchor_route_s7_4p`

Reason:

- Broad response search hurt `sample8`.
- Anchor route added directly to `sample8` hurt sample7 matchup.

Research only:

- `sample12_2p_opening_route_commit_s7_4p`
- `sample13_2p_phase_controller_s7_4p`
- `sample14_2p_counter_capture_candidates_s7_4p`
- `sample15_s8_2p_s7_4p_anchor`
- `sample17_s8_dynamic_roi_late_2p_s7_4p`
- `sample18_planned_anchor_route`
- `sample20_2p_s7_anchor_route_s7_4p`

Useful discovery:

- On seed block `5522554122`, direct `sample7/main.py` beat `sample8/main.py`:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent sample7\main.py --opponent sample8\main.py --games 10 --both-seats --workers 10 --seed-start 5522554122
```

Result:

- `sample7` won `11/20`, or `55.0%`.

Interpretation:

- Do not assume `sample8` is always the best 2P base.
- A `sample7`-style 2P branch is worth studying.

## Evaluation Conventions

Use same seed blocks when comparing branches.

2P quick gate:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent <agent> --opponent sample8\main.py --games 10 --both-seats --workers 10 --seed-start 5522554122
```

2P additional gate:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent <agent> --opponent sample7\main.py --games 10 --both-seats --workers 10 --seed-start 5522554122
```

2P strong local benchmark:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent <agent> --opponent bots\hairate5.py --games 40 --both-seats --workers 10 --seed-start 65122554122
```

4P quick gate:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent <agent> --opponent sample7\main.py --opponent sample8\main.py --opponent bots\hairate5.py --games 10 --workers 10 --seed-start 12000000
```

4P additional blocks:

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent <agent> --opponent sample7\main.py --opponent sample8\main.py --opponent bots\hairate5.py --games 10 --workers 10 --seed-start 56000000
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent <agent> --opponent sample7\main.py --opponent sample8\main.py --opponent bots\hairate5.py --games 10 --workers 10 --seed-start 5421622
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent <agent> --opponent sample7\main.py --opponent sample8\main.py --opponent bots\hairate5.py --games 10 --workers 10 --seed-start 9874600000
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent <agent> --opponent sample7\main.py --opponent sample8\main.py --opponent bots\hairate5.py --games 10 --workers 10 --seed-start 2000000
```

Important:

- Treat `score_diff == 0` as draw for analysis, because local environment can award first seat on exact ties.
- Win rate is more important than average score diff.
- Do not promote a branch based only on average diff.

4P oracle dataset workflow:

```powershell
C:\tmp\ow\Scripts\python.exe build_4p_oracle_dataset.py --blocks 2000000,5421622,56000000,9874600000,12000000 --games 10 --workers 5
```

If interrupted, resume with the same `--out-dir`, or collect existing logs only:

```powershell
C:\tmp\ow\Scripts\python.exe build_4p_oracle_dataset.py --out-dir <existing_research_run_dir> --blocks 2000000,5421622,56000000,9874600000,12000000 --games 10 --workers 5 --collect-only
```

Outputs:

- `oracle_results.csv`
- `oracle_features.csv`
- `oracle_summary.md`
- `oracle_rules.json`

Loss-analysis workflow:

```powershell
C:\tmp\ow\Scripts\python.exe analyze_4p_losses.py --eval-log <evaluate_output.txt> --agent sample32_singlefile_s7_s8_selector\main.py
```

## What To Do Next

Recommended next 4P work:

- Build/finish the oracle dataset using `build_4p_oracle_dataset.py`.
- Inspect `oracle_summary.md` and `oracle_features.csv`.
- Convert only high-confidence `sample8`-winning board buckets into `sample35_singlefile_oracle_selector/oracle_rules.json`.
- Re-evaluate `sample35` across `5421622`, `56000000`, `9874600000`, `12000000`, and `2000000`.

Do not start with:

- More broad response search.
- More generic anchor score bonuses.
- More folder-based selector complexity.
- Another reserve-only contested mode.

Preferred next algorithm after oracle selector:

- If both `sample7` and `sample8` lose the same board bucket, create a true third mode.
- Candidate third mode should be based on:
  - contested-rich route blocking, or
  - enemy safe-cluster denial.
- Do not retry the `sample33` reserve-only idea as-is.

## Implementation Warning

Folder selector issue:

- Running a sub-agent directly can produce different results from running through a wrapper `main.py`.
- Example:
  - Direct `sample20_2p_s7_anchor_route_s7_4p\sample8\main.py` vs `sample8`: `11/20`
  - Wrapper `sample20_2p_s7_anchor_route_s7_4p\main.py` vs `sample8`: `10/20`

Likely cause:

- Dynamic module loading and repeated `orbit_lite` clearing/reloading.

Recommendation:

- For any serious combined submission, integrate into one package/module.
- Avoid loading separate `sample7/orbit_lite` and `sample8/orbit_lite` dynamically if possible.

## Submission Guidance

If submitting immediately:

- Use `sample7_4p_sample8_2p_submit_v3.zip` or another already-proven clean mixed submission.
- Do not submit `sample18/19/20` as-is.

If building a new submission:

- First validate locally.
- Then package with root `main.py` plus required package files.
- Verify zip structure before upload.

Zip should look like:

```text
main.py
sample7/...
sample8/...
```

or, preferably for future high-confidence submission:

```text
main.py
orbit_lite/...
```

with one integrated code path.

## Quick Mental Model

2P:

- `sample8` is strong but not dominant.
- `sample7` can beat `sample8` on some seed blocks.
- Next improvement should target contested neutral / counter-capture timing.

4P:

- Winning bots build thick clusters/anchors.
- Thin scattered expansion loses.
- Route/anchor ideas are directionally right but current implementation is not yet good enough.

Overall:

- Avoid small weight tuning unless it supports a clear algorithmic change.
- Prefer changes to candidate generation, route choice, phase behavior, and tactical counter-capture.
