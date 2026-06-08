import math

def selected_ids(selected):
    used_sources = {cand.source_id for cand in selected}
    taken_targets = {cand.target_id for cand in selected if cand.kind != "regroup"}
    defended_targets = {cand.target_id for cand in selected if cand.kind == "defense"}
    return used_sources, taken_targets, defended_targets


def greedy_select(candidates, budgets, config):
    selected = []
    target_taken = set()
    defended_targets = set()
    used_sources = set()

    for cand in sorted(candidates, key=lambda item: item.score, reverse=True):
        if len(selected) >= config.max_actions:
            break
        if cand.score <= config.roi_threshold:
            break
        if cand.ships > budgets.get(cand.source_id, 0):
            continue
        if cand.kind != "regroup" and cand.target_id in target_taken:
            continue
        if cand.source_id in defended_targets:
            continue
        if cand.kind in ("defense", "urgent_hold") and cand.target_id in used_sources:
            continue

        selected.append(cand)
        budgets[cand.source_id] = max(0, budgets.get(cand.source_id, 0) - cand.ships)
        used_sources.add(cand.source_id)
        if cand.kind != "regroup":
            target_taken.add(cand.target_id)
        if cand.kind in ("defense", "urgent_hold"):
            defended_targets.add(cand.target_id)

    return selected


def greedy_select_limited(candidates, budgets, max_actions, roi_threshold):
    selected = []
    target_taken = set()
    defended_targets = set()
    used_sources = set()

    for cand in sorted(candidates, key=lambda item: item.score, reverse=True):
        if len(selected) >= max_actions:
            break
        if cand.score <= roi_threshold:
            break
        if cand.ships > budgets.get(cand.source_id, 0):
            continue
        if cand.kind != "regroup" and cand.target_id in target_taken:
            continue
        if cand.source_id in defended_targets:
            continue
        if cand.kind in ("defense", "urgent_hold") and cand.target_id in used_sources:
            continue

        selected.append(cand)
        budgets[cand.source_id] = max(0, budgets.get(cand.source_id, 0) - cand.ships)
        used_sources.add(cand.source_id)
        if cand.kind != "regroup":
            target_taken.add(cand.target_id)
        if cand.kind in ("defense", "urgent_hold"):
            defended_targets.add(cand.target_id)

    return selected


def build_multisource_capture_candidates(
    sources,
    targets,
    projection,
    budgets,
    config,
    player,
    current_step,
    angular_velocity,
    initial_planets,
    comet_ids,
    is_2p,
    selected,
    distance,
    predict_intercept_position,
    crosses_sun,
    capture_floor,
    Candidate,
    MultiCandidate,
):
    if not is_2p or current_step < 95 or current_step > 285:
        return []

    used_sources, taken_targets, _ = selected_ids(selected)
    if sum(1 for cand in selected if cand.kind in ("attack", "counter_snipe")) >= 2:
        return []
    candidates = []
    valuable_targets = [
        target for target in targets
        if target.id not in comet_ids
        and target.id not in taken_targets
        and target.production >= 5
    ]

    for target in valuable_targets:
        parts = []
        total_send = 0
        arrival_eta = 0.0
        for source in sorted(sources, key=lambda item: distance(item, target)):
            if source.id in used_sources:
                continue
            budget = budgets.get(source.id, 0)
            if budget < config.min_ships_to_launch:
                continue
            probe = min(budget, max(config.min_ships_to_launch, int(budget * 0.58)))
            tx, ty, eta = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, probe
            )
            if eta > 18.0 or crosses_sun(source, tx, ty):
                continue
            parts.append((eta, source, tx, ty, budget))

        if len(parts) < 2:
            continue
        parts.sort(key=lambda item: item[0])
        planned = []
        for eta, source, tx, ty, budget in parts[:3]:
            needed_at_eta = capture_floor(target, projection, eta, player, overhead=3)
            remaining_need = max(0, needed_at_eta + max(4, int(target.production * 2)) - total_send)
            if remaining_need <= 0:
                break
            send = min(budget, max(config.min_ships_to_launch, min(remaining_need, int(budget * 0.55))))
            if send < config.min_ships_to_launch:
                continue
            angle = math.atan2(ty - source.y, tx - source.x)
            planned.append(Candidate("multi_capture", source.id, target.id, angle, int(send), eta, 0.0))
            total_send += int(send)
            arrival_eta = max(arrival_eta, eta)

        if len(planned) < 2:
            continue
        needed_final = capture_floor(target, projection, arrival_eta, player, overhead=4)
        if total_send < needed_final + max(3, int(target.production)):
            continue
        total_budget = sum(budgets.get(order.source_id, 0) for order in planned)
        if total_send > max(18, int(total_budget * 0.62)):
            continue
        score = (
            target.production * 78.0
            + (36.0 if target.owner != -1 else 18.0)
            - total_send * 1.05
            - arrival_eta * 3.5
        )
        if current_step > 190 and target.owner == -1:
            score -= 35.0
        candidates.append(MultiCandidate("multi_capture", target.id, planned, arrival_eta, score))
    return candidates


def append_multisource_capture(selected, multi_candidates, budgets):
    used_sources, taken_targets, _ = selected_ids(selected)
    for multi in sorted(multi_candidates, key=lambda item: item.score, reverse=True):
        if multi.score <= 35.0:
            break
        if multi.target_id in taken_targets:
            continue
        if any(order.source_id in used_sources for order in multi.orders):
            continue
        if any(order.ships > budgets.get(order.source_id, 0) for order in multi.orders):
            continue
        for order in multi.orders:
            selected.append(order)
            budgets[order.source_id] = max(0, budgets.get(order.source_id, 0) - order.ships)
        break
    return selected
