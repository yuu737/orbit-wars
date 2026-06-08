# Orbit Wars Handoff for Next AI

Date: 2026-06-08
Workspace: `C:\Users\yuu98\Desktop\kaggle\orbit-wars`
Python env: `C:\tmp\ow\Scripts\python.exe`

## User Goal

The final goal is not just beating local bots. The goal is a strong Kaggle competition result.

Important user preferences:
- Keep improving aggressively, but avoid fake progress.
- `main.py` should be beaten around 90% or better by serious candidates.
- `hairate.py`, `hairate2.py`, and `hairate3.py` are reference/strong opponent bots.
- `hairate.py` is currently the most reliable strong local benchmark.
- `hairate3.py` was updated by the user and should be included.
- `hairate2.py` likely requires dependencies; current local env previously lacked `torch`, so do not rely on hairate2 results until dependency status is verified.

## Current Best Candidate

Best current generalist branch:

`bots/main_v6_19_soft_opening_pressure_hybrid.py`

Why:
- It preserves strong win rate vs current `main.py`.
- It slightly improves `hairate` average diff compared with V6.12.
- It keeps some of the opening/pressure ideas without the large regressions seen in harder experiments.

Key result:

`research_runs/v619_vs_v612_standard/summary.md`

```text
V6.19 standard vs V6.12 baseline
hairate: 20 games, 2 wins, avg diff -1618.75
main.py: 20 games, 19 wins, avg diff +2568.55
```

Gate result:

```text
V6.19 gate vs V6.12 baseline
hairate: 8 games, 2 wins, avg diff -368.38
main.py: 8 games, 8 wins, avg diff +3023.75
hairate3_ladder_2p: 60 games, 2 wins, avg diff -1829.02
```

## Important Branches and Decisions

### V6.9

`bots/main_v6_9_attack_ambiguity_penalty.py`

Strong-opponent-focused baseline.

Known result:
- `hairate` standard: 2/20, avg diff about `-1383.55`
- `main.py` standard: 11/20, avg diff about `+1596.65`
- `hairate3_ladder_2p`: 2/60, avg diff about `-1906.92`

Interpretation:
- Better than V6.19 on pure `hairate` average diff.
- Worse than V6.19 on `main.py`.
- Useful as a strong-opponent reference, not a generalist promotion candidate.

### V6.12

`bots/main_v6_12_attack_confidence_bonus.py`

Generalist baseline before V6.19.

Known result:
- `hairate` standard: 2/20, avg diff `-1694.65`
- `main.py` standard: 19/20, avg diff `+2671.25`

Interpretation:
- Very strong vs `main.py`.
- Not enough vs `hairate`.
- V6.19 is a small but real improvement over it for `hairate` while preserving 19/20 vs `main.py`.

### V6.15

`bots/main_v6_15_opening_params.py`

Tried a `hairate`-style opening layer.

Result:
- Improved `hairate3_ladder_2p` average diff relative to V6.9.
- Badly regressed vs `main.py`.

Interpretation:
- Opening weakness is real.
- Hard/strong opening filters are dangerous.

### V6.16

`bots/main_v6_16_aggressive_opening_confidence.py`

Aggressive opening shaping based on V6.12.

Result:
- `hairate` gate avg diff improved.
- `main.py` collapsed to 4/8.

Decision:
- Reject as too disruptive.

### V6.17

`bots/main_v6_17_soft_opening_aggression.py`

Softened opening shaping.

Result:
- `main.py` gate stayed 8/8.
- `hairate3_ladder_2p` avg diff `-1829.02`.

Decision:
- Safe opening-shape branch.

### V6.18

`bots/main_v6_18_early_enemy_pressure.py`

Added early pressure on high-production enemy planets.

Result:
- `main.py` gate stayed 8/8.
- `hairate` gate average diff improved vs V6.12.
- `hairate3_ladder_2p` worse than V6.17.

Decision:
- Idea useful, but not best alone.

### V6.19

`bots/main_v6_19_soft_opening_pressure_hybrid.py`

Hybrid of V6.17 soft opening and V6.18 enemy high-production pressure.

Decision:
- Best current generalist candidate.
- Use as the main base for future work unless targeting pure `hairate` only.

### V6.20

`bots/main_v6_20_forward_eval_selector.py`

First attempt at a forward-eval selector. It added a lightweight short-horizon board evaluation to rescore top candidates.

Result:

`research_runs/v620_vs_v619_gate/summary.md`

```text
hairate: 8 games, 2 wins, avg diff -103.88
main.py: 8 games, 4 wins, avg diff +179.88
hairate3_ladder_2p: 60 games, 0 wins, avg diff -2103.25
```

Interpretation:
- Very interesting: `hairate` diff improved by `+264.50` vs V6.19.
- But it badly broke `main.py` and `hairate3`.
- Forward evaluation has signal, but current evaluation function is too blunt and overrides good normal scoring.

Decision:
- Do not promote.
- Mine it for ideas.

### V6.20b

`bots/main_v6_20b_guarded_forward_eval.py`

Tried reducing the V6.20 forward-eval weight/candidate count.

Result:

`research_runs/v620b_vs_v619_gate/summary.md`

```text
hairate: 8 games, 2 wins, avg diff -574.62
main.py: 8 games, 6 wins, avg diff +1663.75
hairate3_ladder_2p: 60 games, 0 wins, avg diff -2060.83
```

Decision:
- Reject.
- The weaker forward signal did not preserve the good `hairate` improvement and still damaged stability.

### V6.21

`bots/main_v6_21_opening_forward_planner.py`

Status:
- File was created as a copy of V6.19.
- Implementation was started conceptually, but user interrupted before patch completed.
- Treat it as not implemented unless verified by diff.

Intended next step:
- Add forward evaluation only for opening neutral attack candidates, not all candidates.
- Keep V6.19 as base.
- Apply the eval only before turn 14 in 2P.
- This is a safer version of V6.20 because the damage scope is limited to opening neutral selection.

## Evaluation Infrastructure

Main command runner:

`run_research_loop.py`

Use examples:

```powershell
C:\tmp\ow\Scripts\python.exe run_research_loop.py --agent bots/main_v6_19_soft_opening_pressure_hybrid.py --baseline bots/main_v6_12_attack_confidence_bonus.py --name v619_vs_v612_standard --suite standard --no-audit
```

```powershell
C:\tmp\ow\Scripts\python.exe run_research_loop.py --agent bots/main_v6_20_forward_eval_selector.py --baseline bots/main_v6_19_soft_opening_pressure_hybrid.py --name v620_vs_v619_gate --suite gate --benchmark benchmarks/hairate3_ladder_2p.json --no-audit
```

Recommended fast gate:

```powershell
C:\tmp\ow\Scripts\python.exe run_research_loop.py --agent <candidate.py> --baseline bots/main_v6_19_soft_opening_pressure_hybrid.py --name <run_name> --suite gate --benchmark benchmarks/hairate3_ladder_2p.json --no-audit
```

Recommended broader check if gate is promising:

```powershell
C:\tmp\ow\Scripts\python.exe run_research_loop.py --agent <candidate.py> --baseline bots/main_v6_19_soft_opening_pressure_hybrid.py --name <run_name> --suite standard --no-audit
```

Compile check:

```powershell
C:\tmp\ow\Scripts\python.exe -m py_compile bots/<candidate.py>
```

## Benchmarks

Important benchmark files:

- `benchmarks/hairate_fixed_2p.json`
- `benchmarks/hairate_focus_2p.json`
- `benchmarks/hairate3_fixed_2p.json`
- `benchmarks/hairate3_focus_2p.json`
- `benchmarks/hairate3_ladder_2p.json`
- `benchmarks/hairate2_fixed_2p.json`
- `benchmarks/hairate2_focus_2p.json`

`hairate3_ladder_2p`:
- Seeds `65400000` through `65400029`, both seats.
- This should be included for hard regression checks.

`hairate2`:
- Verify dependencies before trusting results.
- Previous state: local env did not have `torch`.

## Strategic Understanding

The main weakness against `hairate`-style bots is probably not one weight. It is structural:

- Our bot still relies heavily on hand-scored greedy candidate selection.
- `hairate`-style bots appear to evaluate action outcomes more directly.
- Opening neutral order matters a lot.
- Enemy intent/counter-snipe timing matters a lot.
- Hard filters copied from hairate are dangerous because they can kill easy wins against weaker bots.

Useful lessons from experiments:

- Opening shaping helps but must be soft.
- Early enemy high-production pressure is useful if it does not suppress normal expansion.
- Full forward-eval selector showed strong signal vs `hairate`, but current evaluation function is not safe enough.
- Any future forward eval should be gated narrowly first.

## Recommended Next Work

Proceed in this order:

1. Finish `V6.21 opening forward planner`.
   - Base: `bots/main_v6_19_soft_opening_pressure_hybrid.py`
   - Scope: only 2P, only `current_step < 14`, only neutral `attack` candidates.
   - Use forward eval as a small bonus/penalty, not hard replacement.
   - Start with small clamp: about `[-20, +30]`.
   - Run gate with `hairate3_ladder_2p`.

2. Build `V6.22 enemy-intent/counter-snipe`.
   - Predict enemy likely neutral captures from enemy planets/fleets.
   - Add candidates for:
     - preemptive capture,
     - counter-snipe,
     - immediate recapture,
     - ignoring low-value captures.
   - Keep this in the unified candidate pool.

3. Add automated sweep after V6.21/V6.22 are stable.
   - Sweep only a small set of constants at first.
   - Candidate constants:
     - opening forward weight
     - opening forward clamp
     - early high-production enemy bonus
     - low-production neutral penalty
     - counter-snipe score bonus
     - hold margin

4. Only after stability, consider replacing `main.py`.
   - Do not replace `main.py` just because a branch improves one benchmark.
   - Promotion criteria should include:
     - `main.py` local win rate around 90%+,
     - no crash,
     - no severe `hairate3_ladder` regression,
     - improvement vs either V6.19 or V6.9 depending on target.

## Likely V6.21 Implementation Sketch

Add functions near `greedy_select` in a copy of V6.19:

- `opening_forward_score(planets, fleets, player, horizon, orders)`
- `apply_opening_forward_to_candidates(candidates, planets, fleets, player, current_step, is_2p)`

Then call:

```python
planner_candidates = apply_opening_forward_to_candidates(
    planner_candidates, planets, fleets, player, current_step, is_2p
)
```

right before:

```python
selected = greedy_select(planner_candidates, dict(unified_budgets), config)
```

Important guardrails:

- Only apply before `opening_turn_limit(is_2p)`.
- Only apply to candidates where `cand.kind == "attack"` and target owner is neutral.
- Only top 8 to 10 candidates by original score.
- Use a small score influence.
- Do not use the global V6.20 version directly; it broke `main.py`.

## Dirty Worktree Warning

There are many modified/untracked files. Do not reset or clean the tree.

Known important untracked files include:

- `run_research_loop.py`
- `audit_shots.py`
- `benchmarks/`
- many `bots/main_v*.py` branches
- `bots/hairate3.py`

The user likely expects these to remain available.

## Final Recommendation

Use `bots/main_v6_19_soft_opening_pressure_hybrid.py` as the current stable base.

Treat `bots/main_v6_20_forward_eval_selector.py` as an experimental proof that forward eval can move the `hairate` matchup, but do not promote it.

Next best action: implement and test `V6.21 opening-forward-only planner`.
