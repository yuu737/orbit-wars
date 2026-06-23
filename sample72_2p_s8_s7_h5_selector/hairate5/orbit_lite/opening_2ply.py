from __future__ import annotations

import torch
from torch import Tensor

from .geometry import fleet_speed


def apply_opening_2ply_score(
    *,
    score: Tensor,
    obs,
    cache,
    source_idx: Tensor,
    target_idx: Tensor,
    target_exists: Tensor,
    target_is_mine: Tensor,
    sizes: Tensor,
    eta: Tensor,
    floor_at_arr: Tensor,
    config,
    player_count: int,
    step: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Lightweight opening 2-ply: score my launch after likely enemy replies."""
    if not bool(getattr(config, "opening_2ply_enabled", False)):
        return score
    if int(player_count) > 2 or int(step) >= int(getattr(config, "opening_2ply_turns", 45)):
        return score

    P = int(obs.P)
    if P <= 0:
        return score

    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    if S == 0 or T == 0:
        return score

    enemy_mask = obs.is_enemy & obs.alive
    if not bool(enemy_mask.any()):
        return score

    src_safe = source_idx.clamp(0, P - 1)
    tgt_safe = target_idx.clamp(0, P - 1)

    enemy_ships = torch.where(
        enemy_mask,
        obs.ships.to(dtype),
        torch.zeros(P, dtype=dtype, device=device),
    )
    enemy_speed = fleet_speed(enemy_ships.clamp(min=1e-6)).clamp(min=1e-6)
    enemy_active = enemy_mask.to(dtype)

    dist_to_tgt = cache.cross_dist[0, :, tgt_safe].to(dtype)
    eta_to_tgt = dist_to_tgt / enemy_speed.view(P, 1)
    eta_to_tgt = torch.where(
        enemy_mask.view(P, 1),
        eta_to_tgt,
        torch.full_like(eta_to_tgt, float("inf")),
    )
    enemy_eta_to_tgt = eta_to_tgt.min(dim=0).values

    dist_to_src = cache.cross_dist[0, :, src_safe].to(dtype)
    eta_to_src = dist_to_src / enemy_speed.view(P, 1)
    eta_to_src = torch.where(
        enemy_mask.view(P, 1),
        eta_to_src,
        torch.full_like(eta_to_src, float("inf")),
    )
    enemy_eta_to_src = eta_to_src.min(dim=0).values

    response_turns = float(getattr(config, "opening_2ply_response_turns", 9.0))
    response_window = (eta + response_turns).clamp(min=1.0)
    tgt_decay = (1.0 - eta_to_tgt.unsqueeze(0) / response_window.unsqueeze(1)).clamp(min=0.0)
    retake_pressure = (enemy_ships.view(1, P, 1) * enemy_active.view(1, P, 1) * tgt_decay).sum(dim=1)
    target_margin = (sizes - floor_at_arr).clamp(min=0.0)
    retake_deficit = (retake_pressure - target_margin).clamp(min=0.0)

    source_after = obs.ships[src_safe].to(dtype).view(S, 1) - sizes
    source_window = float(getattr(config, "opening_2ply_source_turns", 8.0))
    source_pressure = torch.where(
        eta_to_src <= source_window,
        enemy_ships.view(P, 1) * enemy_active.view(P, 1),
        torch.zeros_like(eta_to_src),
    ).sum(dim=0)
    source_floor = torch.maximum(
        torch.full_like(source_pressure, float(getattr(config, "opening_2ply_source_floor", 5.0))),
        source_pressure * float(getattr(config, "opening_2ply_source_pressure_fraction", 0.22)),
    )
    source_deficit = (source_floor.view(S, 1) - source_after).clamp(min=0.0)

    neutral_t = obs.is_neutral[tgt_safe] & target_exists
    target_is_attack = ~target_is_mine.view(1, T)
    arrival_advantage = enemy_eta_to_tgt.view(1, T) - eta
    target_prod = obs.production[tgt_safe].to(dtype) if hasattr(obs, "production") else obs.prod[tgt_safe].to(dtype)
    race_bonus = (
        arrival_advantage.clamp(min=0.0, max=8.0) / 8.0
    ) * target_prod.view(1, T) * float(getattr(config, "opening_2ply_race_weight", 0.10))
    race_allowed = (
        neutral_t.view(1, T)
        & (arrival_advantage >= float(getattr(config, "opening_2ply_min_arrival_advantage", 1.5)))
        & (source_after >= float(getattr(config, "opening_2ply_min_source_after", 5.0)))
        & (score.reshape(S, T) >= float(getattr(config, "roi_threshold", 1.5)) - float(getattr(config, "opening_2ply_roi_margin", 0.35)))
    )

    adjusted = score.reshape(S, T)
    adjusted = adjusted + torch.where(race_allowed, race_bonus, torch.zeros_like(race_bonus))
    adjusted = adjusted - torch.where(
        target_is_attack,
        retake_deficit * float(getattr(config, "opening_2ply_retake_weight", 0.14)),
        torch.zeros_like(retake_deficit),
    )
    adjusted = adjusted - torch.where(
        target_is_attack,
        source_deficit * float(getattr(config, "opening_2ply_source_weight", 0.08)),
        torch.zeros_like(source_deficit),
    )

    if bool(getattr(config, "opening_2ply_hard_source_veto", False)):
        source_risk = (
            target_is_attack
            & (enemy_eta_to_src.view(S, 1) <= source_window)
            & (source_after < source_floor.view(S, 1))
        )
        adjusted = torch.where(source_risk, torch.full_like(adjusted, float("-inf")), adjusted)

    return adjusted.reshape_as(score)
