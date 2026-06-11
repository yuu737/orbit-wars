"""
my_orbit/simulator.py  —  Orbit Wars V10 Simulation Engine

Core concept (from hairate2/4 analysis):
  Top bots score actions by measuring the change in board flow:

    action_value = competitive_flow(after_action) - competitive_flow(baseline)

  where competitive_flow = player_flow - opponent_flow at the TARGET planet.

Design decisions:
  - We do NOT subtract source garrison loss.
    The `ships` in each candidate come from safe_drain (budget already
    committed to leaving). Charging source flow would double-count the cost.
    What matters is only: "which destination maximises target improvement?"

  - We simulate ONLY the target planet (single-planet re-simulation).
    This is fast (~0.1ms per candidate) and captures the essential signal.

  - Competitive diff: we subtract opponent's flow loss at target.
    Capturing an enemy planet scores higher than capturing a neutral
    of equal production, because we also deny the opponent their flow.

Future improvements (V10.x):
  - Multi-planet joint simulation (capture chains)
  - Beam search over action combinations
"""

import math
from my_orbit.projection import resolve_planet_combat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _player_flow(owners, ships_traj, player, horizon):
    """Total ships owned by player at this planet over turns 1..horizon."""
    total = 0.0
    lim = min(horizon + 1, len(owners))
    for t in range(1, lim):
        if owners[t] == player:
            total += float(ships_traj[t])
    return total


def _opp_flow(owners, ships_traj, opponents, horizon):
    """Total ships owned by any opponent at this planet over turns 1..horizon."""
    total = 0.0
    lim = min(horizon + 1, len(owners))
    for t in range(1, lim):
        if owners[t] in opponents:
            total += float(ships_traj[t])
    return total


def _resim_planet(base_owners, base_ships, incoming, production,
                  inject_turn, inject_owner, inject_ships, horizon):
    """
    Re-simulate a single planet from `inject_turn` onward with injected fleet.

    Returns (sim_owners, sim_ships) lists of same length as base_owners.
    """
    sim_owners = list(base_owners)
    sim_ships  = list(base_ships)

    t0 = max(1, min(horizon, inject_turn))
    cur_owner = sim_owners[t0 - 1]
    cur_ships = float(sim_ships[t0 - 1])
    prod = int(production)

    lim = min(horizon + 1, len(base_ships))
    for t in range(t0, lim):
        if cur_owner >= 0:
            cur_ships += prod

        turn_inc = dict(incoming[t]) if t < len(incoming) else {}
        if t == t0:
            turn_inc[inject_owner] = turn_inc.get(inject_owner, 0) + inject_ships

        if turn_inc:
            cur_owner, cur_ships = resolve_planet_combat(
                cur_owner, int(cur_ships), turn_inc
            )

        sim_owners[t] = cur_owner
        sim_ships[t]  = int(cur_ships)

    return sim_owners, sim_ships


# ---------------------------------------------------------------------------
# Core action evaluation
# ---------------------------------------------------------------------------

def action_value(projection, target_id, target_production,
                 ships, eta, player, horizon, opponents):
    """
    Player flow gain from sending `ships` to `target_id`.

    Formula:
        player_flow(with_action) - player_flow(baseline)

    Why no opp_denial:
        Adding opponent flow loss creates a huge bias toward attacking
        well-established enemy planets, causing the bot to ignore neutral
        expansion and opening economy. Player-only gain naturally balances
        attack vs. neutral expansion, while still rewarding enemy captures
        (we gain ships that were previously enemy-owned).

    Returns:
        float >= 0 typically. Negative = attack clearly fails (enemy retakes).
    """
    t_owners = projection.owner_by_id.get(target_id)
    t_ships  = projection.ships_by_id.get(target_id)
    t_inc    = projection.incoming_by_id.get(target_id) or [{}] * (horizon + 1)

    if t_owners is None:
        return 0.0

    arrival = max(1, min(horizon, int(math.ceil(eta))))

    sim_owners, sim_ships = _resim_planet(
        t_owners, t_ships, t_inc, target_production,
        inject_turn=arrival,
        inject_owner=player,
        inject_ships=int(ships),
        horizon=horizon,
    )

    base_player = _player_flow(t_owners,   t_ships,  player, horizon)
    sim_player  = _player_flow(sim_owners, sim_ships, player, horizon)

    return sim_player - base_player



# ---------------------------------------------------------------------------
# Board-level diagnostics
# ---------------------------------------------------------------------------

def board_flow(projection, player, opponents, horizon):
    """Total competitive flow (player_flow - opp_flow) across all planets."""
    p_total = 0.0
    o_total = 0.0
    for pid, owners in projection.owner_by_id.items():
        ships_t = projection.ships_by_id[pid]
        lim = min(horizon + 1, len(owners))
        for t in range(1, lim):
            if owners[t] == player:
                p_total += float(ships_t[t])
            elif owners[t] in opponents:
                o_total += float(ships_t[t])
    return p_total - o_total


# ---------------------------------------------------------------------------
# Main entry point: replace attack candidate scores
# ---------------------------------------------------------------------------

def score_candidates_sim(candidates, planet_by_id, source_ships_by_id,
                          projection, player, opponents, horizon,
                          attack_only=True):
    """
    Replace heuristic scores with simulation-based competitive flow delta.

    Only 'attack' candidates are re-scored (others keep heuristic scores).
    The simulation score is in ship-turn units; positive = good attack.
    """
    for cand in candidates:
        if attack_only and cand.kind != "attack":
            continue

        planet = planet_by_id.get(cand.target_id)
        if planet is None:
            continue

        cand.score = action_value(
            projection=projection,
            target_id=cand.target_id,
            target_production=int(planet.production),
            ships=cand.ships,
            eta=cand.eta,
            player=player,
            horizon=horizon,
            opponents=opponents,
        )

    return candidates
