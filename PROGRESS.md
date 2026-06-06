# Orbit Wars Progress

Last updated: 2026-06-06

## Current Status

- [x] Competition joined
- [x] Starter kit downloaded and reviewed
- [x] Local Git repository initialized
- [x] GitHub repository connected and initial push completed
- [x] Local Python environment prepared
- [x] `kaggle-environments` available in the local runner environment
- [x] Starter `main.py` verified against `random`
- [x] Multi-seed local evaluation script created
- [x] Parallel local evaluation added
- [x] V1 bot implemented
- [x] Rotation-aware targeting implemented
- [x] Sun-safe routing implemented
- [ ] Defense and reinforcement logic implemented
- [ ] Endgame mode implemented
- [x] Kaggle submission loop started
- [ ] Final two submissions selected

## Current Environment

- Workspace: `C:\Users\yuu98\Desktop\kaggle\orbit-wars`
- Local runner Python: `C:\tmp\ow\Scripts\python.exe`
- Remote repository: `https://github.com/yuu737/orbit-wars`

## Next Step

Evaluate new bot candidates in both 2-player and 4-player setups, then build the next bot iteration with:

- Better enemy attack and reinforcement decisions
- Early defense of newly captured planets
- Follow-up analysis from fresh replay losses in both formats

Keep using `evaluate.py` to measure each change.

## Working Notes

- High-level status lives in this file.
- Change-by-change intent and evidence live in [EXPERIMENT_LOG.md](C:/Users/yuu98/Desktop/kaggle/orbit-wars/EXPERIMENT_LOG.md:1).
- `random` is now a sanity check, not a promotion gate.
- Head-to-head against earlier snapshots is required before treating a change as stronger.

## Latest Results

- Starter bot vs `random` over 20 games:
- Win rate: `70%`
- Average score diff: `+1621.75`
- Crash rate: `0%`
- Updated V1 bot vs `random` over 20 games:
- Win rate: `95%`
- Average score diff: `+13069.95`
- Crash rate: `0%`
- Updated V2 bot vs `random` over 20 games:
- Win rate: `100%`
- Average score diff: `+18869.60`
- Crash rate: `0%`
- V2.1 early-expansion bot vs `random` over 20 games:
- Win rate: `100%`
- Average score diff: `+18692.35`
- Crash rate: `0%`
- V2.2 opening-mode bot vs `random` over 20 games:
- Win rate: `100%`
- Average score diff: `+19040.00`
- Crash rate: `0%`
- V2.3 reinforcement bot vs `random` over 20 games:
- Win rate: `100%`
- Average score diff: `+21767.65`
- Crash rate: `0%`
- V2.3 vs `v2.2` over 20 games and both seats:
- Win rate: `30%`
- Average score diff: `-4432.25`
- Decision: `rejected`
- V2.4 threat-aware defense vs `random` over 20 games and both seats:
- Win rate: `100%`
- Average score diff: `+20796.00`
- Crash rate: `0%`
- V2.4 vs `v2.2` over 20 games and both seats:
- Win rate: `20%`
- Average score diff: `-7883.70`
- Decision: `rejected`

## Latest Replay Insight

- A real 1v1 loss showed that the bot was too passive early.
- By turn `50`, the opponent had already expanded to `7` planets while we were stuck on `4`.
- The next adjustment reduced early reserves and pushed harder for neutral expansion before turn `80`.
- A second real 1v1 loss showed an even slower opening, with us still on `1` planet at turn `25`.
- The next adjustment added an explicit opening mode that prioritizes cheap nearby neutral planets before turn `90`.

## Evaluation Script

Use `evaluate.py` to run `main.py` across multiple seeds and report:

- Win rate
- Average score difference
- Crash rate
- Average game length

Current 2-player sanity-check command:

`C:\tmp\ow\Scripts\python.exe evaluate.py --agent main.py --opponent random --games 20`

Fast 2-player self-play command:

`C:\tmp\ow\Scripts\python.exe evaluate.py --agent main.py --opponent bots\v2_2_opening_mode.py --games 20 --both-seats --workers 8`

Fast 4-player command:

`C:\tmp\ow\Scripts\python.exe evaluate.py --agent main.py --players 4 --opponent random --games 10 --both-seats --workers 8`

## Roadmap

### Today: environment and evaluation base

- Install `kaggle-environments>=1.28.0`
- Create a script to auto-run multi-seed matches for `main.py`
- Report win rate, average score difference, crash rate, and average survival/game length

### Today to tomorrow: build the first real bot

- Replace nearest-only expansion with target scoring based on:
- `production`
- `capture_cost`
- `travel_time`
- Add ship reservation so planets do not empty themselves recklessly

### Day 2 to day 4: prediction and safe firing

- Predict future positions of rotating planets using `initial_planets` and `angular_velocity`
- Aim at arrival-time positions instead of current positions
- Avoid shots that cross the sun

### Day 4 to day 7: stronger combat decisions

- Add enemy-planet attack logic
- Add friendly reinforcement logic
- Estimate incoming enemy fleets
- Check whether captured planets can actually be held

### Day 7 to day 10: endgame strategy

- Use remaining turns and reachability to stop low-value long-distance expansion
- Shift toward defense and ship-count preservation when the game is close to ending

### Day 10 to day 14: Kaggle submission loop

- Use the 5 daily submissions with clear roles
- Example variants:
- `v_aggressive`
- `v_balanced`
- `v_defensive`
- Review replay and log patterns to classify losses

### Final stretch: stabilization and final selection

- Compare a high-upside version with a stable low-error version
- Avoid major rewrites near the deadline
- Use the final days for targeted fixes only

## Strategy Priorities

1. Neutral planet value scoring
2. Ship reservation and launch sizing
3. Rotation-aware arrival prediction
4. Sun collision avoidance
5. Simple enemy fleet prediction
6. Endgame mode
7. 4-player specific heuristics

## Notes

- All future bot changes should be judged against repeatable seed-based results, not intuition alone.
- Snapshot comparisons should be run before new Kaggle submissions when possible.
