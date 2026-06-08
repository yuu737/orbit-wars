import argparse
import contextlib
import io
import math
from collections import Counter

from kaggle_environments import make


CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.0
MAX_SPEED = 6.0


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def fleet_speed(ships):
    ships = max(1, int(ships))
    if ships == 1:
        return 1.0
    scaled = math.log(ships) / math.log(1000)
    return 1.0 + (MAX_SPEED - 1.0) * (scaled ** 1.5)


def distance_xy(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return distance_xy(px, py, ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return distance_xy(px, py, ax + t * dx, ay + t * dy)


def crosses_sun(ax, ay, bx, by):
    return point_to_segment_distance(CENTER_X, CENTER_Y, ax, ay, bx, by) <= SUN_RADIUS + SUN_MARGIN


def out_of_bounds(x, y):
    return x < 0.0 or x > 100.0 or y < 0.0 or y > 100.0


def angle_diff(a, b):
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def is_rotating(planet):
    orbital_radius = distance_xy(float(planet[2]), float(planet[3]), CENTER_X, CENTER_Y)
    return orbital_radius + float(planet[4]) < 50.0


def ship_bucket(ships):
    ships = int(ships)
    if ships <= 7:
        return "s_1_7"
    if ships <= 15:
        return "s_8_15"
    if ships <= 31:
        return "s_16_31"
    if ships <= 63:
        return "s_32_63"
    return "s_64_plus"


def infer_target(action, planets, player):
    source_id, angle, _ships = action
    source = next((planet for planet in planets if int(planet[0]) == int(source_id)), None)
    if source is None:
        return None

    best = None
    best_key = None
    for target in planets:
        if int(target[0]) == int(source_id):
            continue
        heading = math.atan2(float(target[3]) - float(source[3]), float(target[2]) - float(source[2]))
        diff = angle_diff(float(angle), heading)
        dist = distance_xy(float(source[2]), float(source[3]), float(target[2]), float(target[3]))
        key = (diff, dist)
        if best_key is None or key < best_key:
            best_key = key
            best = target
    return None if best is None else int(best[0])


def classify_disappearance(fleet, planets_next):
    _fleet_id, _owner, x, y, angle, _from_planet, ships = fleet
    speed = fleet_speed(ships)
    nx = float(x) + math.cos(float(angle)) * speed
    ny = float(y) + math.sin(float(angle)) * speed

    if crosses_sun(float(x), float(y), nx, ny):
        return "sun_loss", None
    if out_of_bounds(nx, ny):
        return "out_of_bounds", None

    best_target = None
    best_dist = None
    for planet in planets_next:
        px = float(planet[2])
        py = float(planet[3])
        radius = float(planet[4])
        dist = point_to_segment_distance(px, py, float(x), float(y), nx, ny)
        if dist <= radius + 0.25:
            center_dist = distance_xy(nx, ny, px, py)
            if best_dist is None or center_dist < best_dist:
                best_dist = center_dist
                best_target = int(planet[0])
    if best_target is not None:
        return "planet_hit", best_target
    return "unknown_loss", None


def audit_one(agent, opponent, seed, seat):
    agents = [opponent, opponent]
    agents[seat] = agent
    with quiet():
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)

    launched = {}
    launch_count = 0
    counters = Counter()

    for turn in range(1, len(env.steps)):
        prev_obs = env.steps[turn - 1][seat].observation
        state = env.steps[turn][seat]
        obs = state.observation
        actions = list(state.action or [])

        prev_fleets = {int(fleet[0]): fleet for fleet in prev_obs["fleets"] if int(fleet[1]) == seat}
        curr_fleets = {int(fleet[0]): fleet for fleet in obs["fleets"] if int(fleet[1]) == seat}

        # Match new fleet ids to actions from this turn.
        new_fleet_ids = [fleet_id for fleet_id in curr_fleets if fleet_id not in prev_fleets]
        unmatched = list(new_fleet_ids)
        for action in actions:
            source_id, angle, ships = action
            best_id = None
            best_key = None
            for fleet_id in unmatched:
                fleet = curr_fleets[fleet_id]
                key = (
                    abs(int(fleet[5]) - int(source_id)),
                    abs(int(fleet[6]) - int(ships)),
                    angle_diff(float(fleet[4]), float(angle)),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_id = fleet_id
            if best_id is None:
                counters["unmatched_launch_action"] += 1
                continue
            unmatched.remove(best_id)
            launched[best_id] = {
                "turn": turn,
                "source_id": int(source_id),
                "ships": int(ships),
                "target_guess": infer_target(action, prev_obs["planets"], seat),
            }
            target_guess = launched[best_id]["target_guess"]
            guessed_target = next((planet for planet in prev_obs["planets"] if int(planet[0]) == int(target_guess)), None)
            if guessed_target is not None:
                owner = int(guessed_target[1])
                if owner == -1:
                    launched[best_id]["target_kind"] = "neutral"
                elif owner == seat:
                    launched[best_id]["target_kind"] = "friendly"
                else:
                    launched[best_id]["target_kind"] = "enemy"
                launched[best_id]["target_rotating"] = "rotating" if is_rotating(guessed_target) else "static"
                launched[best_id]["ship_bucket"] = ship_bucket(ships)
                launched[best_id]["target_prod"] = int(guessed_target[6])
            else:
                launched[best_id]["target_kind"] = "unknown"
                launched[best_id]["target_rotating"] = "unknown"
                launched[best_id]["ship_bucket"] = ship_bucket(ships)
                launched[best_id]["target_prod"] = -1
            launch_count += 1

        disappeared = [fleet_id for fleet_id in prev_fleets if fleet_id not in curr_fleets]
        for fleet_id in disappeared:
            launch = launched.pop(fleet_id, None)
            if launch is None:
                continue
            fate, hit_target = classify_disappearance(prev_fleets[fleet_id], obs["planets"])
            counters[fate] += 1
            if fate == "planet_hit":
                if hit_target == launch["target_guess"]:
                    counters["target_hit_guess"] += 1
                else:
                    counters["wrong_planet_hit_guess"] += 1
                    counters[f"wrong_kind_{launch['target_kind']}"] += 1
                    counters[f"wrong_rot_{launch['target_rotating']}"] += 1
                    counters[f"wrong_ship_{launch['ship_bucket']}"] += 1
                    counters[f"wrong_prod_{launch['target_prod']}"] += 1
            if fate == "out_of_bounds":
                counters[f"out_kind_{launch['target_kind']}"] += 1
                counters[f"out_rot_{launch['target_rotating']}"] += 1
                counters[f"out_ship_{launch['ship_bucket']}"] += 1
                counters[f"out_prod_{launch['target_prod']}"] += 1

    # Surviving fleets are unresolved, but still count launches.
    counters["launches"] = launch_count
    counters["still_in_flight"] = len(launched)
    return counters


def main():
    parser = argparse.ArgumentParser(description="Audit Orbit Wars shot outcomes from replay data.")
    parser.add_argument("--agent", default="main.py")
    parser.add_argument("--opponent", default="bots/hairate.py")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--both-seats", action="store_true")
    args = parser.parse_args()

    if args.seed_list:
        seeds = [int(seed.strip()) for seed in args.seed_list.split(",") if seed.strip()]
    else:
        seeds = [args.seed_start + offset for offset in range(args.games)]

    seats = [0, 1] if args.both_seats else [0]
    totals = Counter()
    for seat in seats:
        for seed in seeds:
            totals.update(audit_one(args.agent, args.opponent, seed, seat))

    launches = max(1, totals["launches"])
    print(f"launches={totals['launches']}")
    for key in [
        "sun_loss",
        "out_of_bounds",
        "unknown_loss",
        "planet_hit",
        "target_hit_guess",
        "wrong_planet_hit_guess",
        "still_in_flight",
        "unmatched_launch_action",
    ]:
        value = totals.get(key, 0)
        ratio = value / launches
        print(f"{key}={value} rate={ratio:.3f}")

    for prefix in [
        "out_kind_", "out_rot_", "out_ship_", "out_prod_",
        "wrong_kind_", "wrong_rot_", "wrong_ship_", "wrong_prod_",
    ]:
        matching = sorted((key, value) for key, value in totals.items() if key.startswith(prefix))
        for key, value in matching:
            denom = totals["out_of_bounds"] if prefix.startswith("out_") else totals["wrong_planet_hit_guess"]
            ratio = value / max(1, denom)
            print(f"{key}={value} share={ratio:.3f}")


if __name__ == "__main__":
    main()
