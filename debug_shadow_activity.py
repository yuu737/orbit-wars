import argparse
import importlib.util
from pathlib import Path

from kaggle_environments import make


def load_agent(path):
    spec = importlib.util.spec_from_file_location("debug_agent", Path(path).resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrap_shadow_builders(module, counters):
    original_expand = module.build_shadow_opening_expansion
    original_defense = module.build_shadow_emergency_defense
    original_micro_regroup = getattr(module, "build_shadow_micro_regroup", None)

    def inspect_expand_filters(args):
        sources = args[0]
        planets = args[2]
        fleets = args[3]
        projection = args[4]
        remaining_budgets = args[5]
        config = args[6]
        player = args[7]
        current_step = args[8]
        angular_velocity = args[9]
        initial_planets = args[10]
        comet_ids = args[11]
        is_2p = args[12]
        selected = args[13]
        if current_step > (85 if is_2p else 60):
            counters["filter_step_late"] += 1
            return
        _, taken_targets, _ = module.selected_ids(selected)
        if len([cand for cand in selected if cand.kind in ("attack", "counter_snipe")]) >= 2:
            counters["filter_many_attacks"] += 1
            return
        neutral_targets = [
            p for p in planets
            if p.owner == -1 and p.id not in comet_ids and p.id not in taken_targets and p.production >= (3 if is_2p else 4)
        ]
        for source in sources:
            budget = remaining_budgets.get(source.id, 0)
            if budget < config.min_ships_to_launch:
                counters["filter_low_budget"] += max(1, len(neutral_targets))
                continue
            for target in neutral_targets:
                counters["filter_pairs"] += 1
                tx, ty, eta_probe = module.predict_intercept_position(
                    source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, budget
                )
                if eta_probe > (14.0 if is_2p else 10.0) or module.crosses_sun(source, tx, ty):
                    counters["filter_eta_or_sun_probe"] += 1
                    continue
                needed = module.capture_floor(target, projection, eta_probe, player, overhead=2)
                if target.production >= 5:
                    needed += 1
                if needed > budget or needed < config.min_ships_to_launch:
                    counters["filter_needed_budget"] += 1
                    continue
                if needed > min(18, int(budget * (0.72 if is_2p else 0.55))):
                    counters["filter_needed_ratio"] += 1
                    continue
                tx, ty, eta = module.predict_intercept_position(
                    source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, needed
                )
                if eta > (14.0 if is_2p else 10.0) or module.crosses_sun(source, tx, ty):
                    counters["filter_eta_or_sun_final"] += 1
                    continue
                turn = min(len(projection.ships_by_id[target.id]) - 1, int(module.math.ceil(eta)))
                post_capture = max(1, int(needed) - int(projection.ships_by_id[target.id][turn]))
                if target.production >= 4 and module.retake_risk_after_capture(target, planets, player, eta, post_capture, is_2p):
                    counters["filter_retake_risk"] += 1
                    continue
                pressure = module.enemy_pressure(target, planets, fleets, player, 16 if is_2p else 10)
                if target.production >= 4 and pressure > needed + (6 if is_2p else 4):
                    counters["filter_pressure"] += 1
                    continue
                counters["filter_would_candidate"] += 1

    def counted_expand(*args, **kwargs):
        inspect_expand_filters(args)
        planets = args[2]
        remaining_budgets = args[5]
        current_step = args[8]
        comet_ids = args[11]
        result = original_expand(*args, **kwargs)
        counters["expand_calls"] += 1
        counters["expand_candidates"] += len(result)
        if result:
            counters["expand_turns"].append(current_step)
        if current_step <= 90:
            counters["early_expand_calls"] += 1
            counters["early_sources_with_budget"] += sum(1 for value in remaining_budgets.values() if value >= 4)
            counters["early_neutral_targets"] += sum(
                1 for planet in planets
                if planet.owner == -1 and planet.id not in comet_ids and planet.production >= 3
            )
        return result

    def counted_defense(*args, **kwargs):
        result = original_defense(*args, **kwargs)
        counters["defense_calls"] += 1
        counters["defense_candidates"] += len(result)
        if result:
            counters["defense_turns"].append(args[6])
        return result

    module.build_shadow_opening_expansion = counted_expand
    module.build_shadow_emergency_defense = counted_defense
    if original_micro_regroup is not None:
        def counted_micro_regroup(*args, **kwargs):
            result = original_micro_regroup(*args, **kwargs)
            counters["micro_regroup_calls"] += 1
            counters["micro_regroup_candidates"] += len(result)
            if result:
                counters["micro_regroup_turns"].append(args[7])
            return result

        module.build_shadow_micro_regroup = counted_micro_regroup


def main():
    parser = argparse.ArgumentParser(description="Count V4 shadow candidate activity for one game.")
    parser.add_argument("--agent", default="bots/main_v4_30_surplus_shadow.py")
    parser.add_argument("--opponent", default="main.py")
    parser.add_argument("--seed", type=int, default=54661125)
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--players", type=int, default=2)
    args = parser.parse_args()

    module = load_agent(args.agent)
    counters = {
        "expand_calls": 0,
        "expand_candidates": 0,
        "expand_turns": [],
        "early_expand_calls": 0,
        "early_sources_with_budget": 0,
        "early_neutral_targets": 0,
        "filter_step_late": 0,
        "filter_many_attacks": 0,
        "filter_low_budget": 0,
        "filter_pairs": 0,
        "filter_eta_or_sun_probe": 0,
        "filter_needed_budget": 0,
        "filter_needed_ratio": 0,
        "filter_eta_or_sun_final": 0,
        "filter_retake_risk": 0,
        "filter_pressure": 0,
        "filter_would_candidate": 0,
        "defense_calls": 0,
        "defense_candidates": 0,
        "defense_turns": [],
        "micro_regroup_calls": 0,
        "micro_regroup_candidates": 0,
        "micro_regroup_turns": [],
    }
    wrap_shadow_builders(module, counters)

    agents = [args.opponent] * args.players
    agents[args.seat] = module.agent
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run(agents)
    final = env.steps[-1][args.seat]
    print(f"status={final.status} reward={final.reward} steps={len(env.steps)-1}")
    for key, value in counters.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
