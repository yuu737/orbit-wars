# 4P Current Work Bundle

Created: 2026-06-19

## Purpose

This folder is a snapshot of the current 4P research state.
It is meant for quick comparison, handoff, and rollback while we work toward a first-place-oriented outer-domain/factory bot.

## Included Candidates

- `sample7/`
  - Current strongest simple 4P baseline in many local pools.
  - Use as the main benchmark for opening reliability.
- `sample8/`
  - Strong 2P-oriented sample family, weaker in 4P but useful as a contrasting 4P opponent.
- `bots/hairate5.py`
  - Older strong local 4P/hairate-family benchmark.
- `sample36_4p_third_mode_lane_anchor/`
  - Stable selector-era reference.
- `sample44_4p_script_portfolio_planner/`
  - Script portfolio planner reference.
- `sample45_4p_domain_control_planner/`
  - Latest stable outer-domain/factory candidate.
  - Important reference: did not clearly beat sample44, but kept the 15-seed score.
- `sample48_4p_sample_opening_domain_factory/`
  - Current experimental branch.
  - Idea: keep sample-style opening, then add domain/factory pressure after opening.
  - Not stable yet. Use for continued research, not immediate submission.

## Included Tools

- `evaluate.py`
  - Main local evaluator.
- `evaluate_4p_all_seats.py`
  - All-seat 4P comparison helper.
- `build_4p_oracle_dataset.py`
  - Builds seed/candidate oracle datasets from local evaluations.
- `analyze_4p_losses.py`
  - Generates initial-board/loss analysis snapshots.

## Current Direction

The current working hypothesis is:

1. Do not replace the opening completely.
2. Preserve sample-style early captures.
3. Add a stronger midgame outer-domain/factory layer.
4. For a real jump, the next branch likely needs an opening planner that still captures reliably but chooses one connected outer lane more intentionally.

## Useful Smoke Commands

```powershell
C:\tmp\ow\Scripts\python.exe -m py_compile sample48_4p_sample_opening_domain_factory\main.py
```

```powershell
C:\tmp\ow\Scripts\python.exe evaluate.py --players 4 --agent sample48_4p_sample_opening_domain_factory\main.py --opponent sample7\main.py --opponent sample8\main.py --opponent bots\hairate5.py --seed-list 2000001,5421622,9874600002,12000002,125693095,1661282750 --workers 6
```

