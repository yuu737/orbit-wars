from __future__ import annotations

import os
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import hairate14_response_search as _h14


_ORIGINAL_TIER_CANDIDATES = _h14._tier_candidates

POWER_PROD_HORIZON = 30.0
LEADER_SAFE_ATTACK_PENALTY = 0.08
LEADER_DEFENSE_BONUS = 0.06
MIDDLE_ATTACK_BONUS = 0.03
TRAILING_ATTACK_BONUS = 0.07
WEAK_ENEMY_BONUS = 0.04
HIGH_PROD_SCALE = 0.020
LONG_ETA_PENALTY = 0.016


def _owner_power(obs, *, player_count: int, dtype, device):
    owners = torch.arange(int(player_count), dtype=obs.owner_abs.dtype, device=device)
    planet_owner = obs.owner_abs.view(-1, 1)
    owned_planets = (planet_owner == owners.view(1, -1)) & obs.alive.view(-1, 1)
    planet_ships = (obs.ships.to(dtype).view(-1, 1) * owned_planets.to(dtype)).sum(dim=0)
    planet_prod = (obs.prod.to(dtype).view(-1, 1) * owned_planets.to(dtype)).sum(dim=0)

    if int(obs.F) > 0:
        fleet_owner = obs.f_owner.view(-1, 1)
        owned_fleets = (fleet_owner == owners.view(1, -1)) & obs.f_alive.view(-1, 1)
        fleet_ships = (obs.f_ships.to(dtype).view(-1, 1) * owned_fleets.to(dtype)).sum(dim=0)
    else:
        fleet_ships = torch.zeros(int(player_count), dtype=dtype, device=device)

    return planet_ships + fleet_ships + planet_prod * float(POWER_PROD_HORIZON)


def _rank_mode(obs, *, player_count: int, dtype, device):
    power = _owner_power(obs, player_count=int(player_count), dtype=dtype, device=device)
    pid = int(obs.player_id)
    my_power = power[pid]
    better = (power > my_power + 1e-6).sum()
    rank = int(better.item()) + 1
    leader_power = power.max()
    leader_gap = (leader_power - my_power) / leader_power.clamp(min=1.0)
    return rank, leader_gap


def _rank_aware_adjustment(
    *,
    obs,
    player_count: int,
    cand_tgt_slot,
    cand_eta,
    cand_is_def,
    score,
    dtype,
    device,
):
    if int(player_count) < 4 or int(score.numel()) == 0:
        return score

    rank, leader_gap = _rank_mode(obs, player_count=int(player_count), dtype=dtype, device=device)
    tgt = cand_tgt_slot.clamp(0, int(obs.P) - 1)
    owner = obs.owner_abs[tgt]
    prod = obs.prod[tgt].to(dtype)
    ships = obs.ships[tgt].to(dtype)
    eta = cand_eta[:, 0].to(dtype)

    is_attack = ~cand_is_def
    is_enemy = (owner >= 0) & (owner != int(obs.player_id))
    is_neutral = owner < 0
    weak_enemy = is_enemy & (ships <= 28.0)
    high_prod = (prod * float(HIGH_PROD_SCALE)).clamp(max=0.18)
    close_bonus = (1.0 - eta / 14.0).clamp(min=0.0, max=1.0) * 0.05
    long_penalty = (eta - 8.0).clamp(min=0.0) * float(LONG_ETA_PENALTY)

    adjustment = torch.zeros_like(score)
    if rank == 1:
        safe_value = high_prod + close_bonus + weak_enemy.to(dtype) * 0.06
        attack_penalty = float(LEADER_SAFE_ATTACK_PENALTY) + long_penalty
        adjustment = adjustment + torch.where(is_attack, safe_value - attack_penalty, torch.zeros_like(score))
        adjustment = adjustment + cand_is_def.to(dtype) * float(LEADER_DEFENSE_BONUS)
    elif rank in (2, 3):
        opportunity = high_prod + close_bonus + weak_enemy.to(dtype) * float(WEAK_ENEMY_BONUS)
        adjustment = adjustment + torch.where(is_attack & (is_enemy | is_neutral), opportunity + float(MIDDLE_ATTACK_BONUS), torch.zeros_like(score))
    else:
        comeback = high_prod * 1.4 + weak_enemy.to(dtype) * (float(WEAK_ENEMY_BONUS) + 0.08)
        comeback = comeback + leader_gap.to(dtype) * 0.12
        adjustment = adjustment + torch.where(is_attack & (is_enemy | is_neutral), comeback + float(TRAILING_ATTACK_BONUS), torch.zeros_like(score))

    adjusted = score + torch.where(torch.isfinite(score), adjustment, torch.zeros_like(score))
    return adjusted


def _rank_aware_tier_candidates(**kwargs):
    result = _ORIGINAL_TIER_CANDIDATES(**kwargs)
    player_count = int(kwargs.get("player_count", 2))
    if player_count < 4:
        return result

    (
        cand_src,
        cand_send,
        cand_angle,
        cand_eta,
        cand_active,
        cand_tgt_slot,
        cand_tgt_short,
        cand_is_def,
        score,
    ) = result
    obs = kwargs["obs"]
    adjusted = _rank_aware_adjustment(
        obs=obs,
        player_count=player_count,
        cand_tgt_slot=cand_tgt_slot,
        cand_eta=cand_eta,
        cand_is_def=cand_is_def,
        score=score,
        dtype=score.dtype,
        device=score.device,
    )
    return (
        cand_src,
        cand_send,
        cand_angle,
        cand_eta,
        cand_active,
        cand_tgt_slot,
        cand_tgt_short,
        cand_is_def,
        adjusted,
    )


def agent(obs):
    _h14._tier_candidates = _rank_aware_tier_candidates
    return _h14.agent(obs)
