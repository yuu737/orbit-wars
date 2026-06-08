import math


def resolve_planet_combat(owner, garrison, arrivals):
    arrivals = {int(k): int(v) for k, v in arrivals.items() if int(v) > 0}
    if not arrivals:
        return int(owner), max(0, int(garrison))

    ranked = sorted(arrivals.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = ranked[0]
    second_ships = ranked[1][1] if len(ranked) > 1 else 0
    survivor = top_ships - second_ships
    if survivor <= 0:
        return int(owner), max(0, int(garrison))

    if top_owner == owner:
        return int(owner), max(0, int(garrison) + survivor)

    if survivor > garrison:
        return int(top_owner), int(survivor - garrison)
    return int(owner), int(garrison - survivor)


def project_planet_states(
    planets,
    fleets,
    player,
    horizon,
    Projection,
    fleet_points_toward_planet,
    fleet_eta_to_planet,
):
    owner_by_id = {}
    ships_by_id = {}
    incoming_by_id = {}
    first_loss_turn_by_id = {}

    for planet in planets:
        owner_by_id[planet.id] = [int(planet.owner)] + [int(planet.owner)] * horizon
        ships_by_id[planet.id] = [int(planet.ships)] + [int(planet.ships)] * horizon
        incoming_by_id[planet.id] = [dict() for _ in range(horizon + 1)]
        first_loss_turn_by_id[planet.id] = None

    planet_by_id = {planet.id: planet for planet in planets}

    for fleet in fleets:
        best_planet = None
        best_eta = None
        for planet in planets:
            if not fleet_points_toward_planet(fleet, planet):
                continue
            eta = fleet_eta_to_planet(fleet, planet)
            if eta <= horizon and (best_eta is None or eta < best_eta):
                best_planet = planet
                best_eta = eta

        if best_planet is None or best_eta is None:
            continue

        turn = max(1, min(horizon, int(math.ceil(best_eta))))
        arrivals = incoming_by_id[best_planet.id][turn]
        arrivals[int(fleet.owner)] = arrivals.get(int(fleet.owner), 0) + int(fleet.ships)

    for planet_id, planet in planet_by_id.items():
        owner = int(planet.owner)
        ships = int(planet.ships)
        for turn in range(1, horizon + 1):
            if owner >= 0:
                ships += int(planet.production)

            owner, ships = resolve_planet_combat(
                owner,
                ships,
                incoming_by_id[planet_id][turn],
            )
            owner_by_id[planet_id][turn] = owner
            ships_by_id[planet_id][turn] = ships

            if first_loss_turn_by_id[planet_id] is None and planet.owner == player and owner != player:
                first_loss_turn_by_id[planet_id] = turn

    return Projection(
        owner_by_id,
        ships_by_id,
        incoming_by_id,
        first_loss_turn_by_id,
    )