from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet as OrbitFleet
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet as OrbitPlanet
except Exception:
    OrbitFleet = None
    OrbitPlanet = None


CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
MAX_TURNS = 500
NUMBER_OF_PREDICTED_TURNS = 20


@dataclass
class Fleet:
    _owner: int
    _num_ships: int
    _source_planet: int
    _destination_planet: int
    _total_trip_length: int
    _turns_remaining: int

    def Owner(self):
        return self._owner

    def NumShips(self):
        return self._num_ships

    def SourcePlanet(self):
        return self._source_planet

    def DestinationPlanet(self):
        return self._destination_planet

    def TotalTripLength(self):
        return self._total_trip_length

    def TurnsRemaining(self):
        return self._turns_remaining


class Planet:
    def __init__(self, planet_id, owner, num_ships, growth_rate, x, y, radius=1.0):
        self._planet_id = int(planet_id)
        self._owner = int(owner)
        self._num_ships = int(num_ships)
        self._growth_rate = int(growth_rate)
        self._x = float(x)
        self._y = float(y)
        self._radius = float(radius)

    def PlanetID(self):
        return self._planet_id

    def Owner(self, new_owner=None):
        if new_owner is None:
            return self._owner
        self._owner = int(new_owner)

    def NumShips(self, new_num_ships=None):
        if new_num_ships is None:
            return self._num_ships
        self._num_ships = int(new_num_ships)

    def GrowthRate(self):
        return self._growth_rate

    def X(self):
        return self._x

    def Y(self):
        return self._y

    def Radius(self):
        return self._radius

    def AddShips(self, amount):
        self._num_ships += int(amount)

    def RemoveShips(self, amount):
        self._num_ships -= int(amount)


class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, obj):
        if obj not in self.parents:
            self.parents[obj] = obj
            self.weights[obj] = 1
            return obj
        path = [obj]
        root = self.parents[obj]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def MinimumSpanningTree(G):
    subtrees = UnionFind()
    tree = []
    edges = [(G[u][v], u, v) for u in G for v in G[u]]
    edges.sort()
    for _weight, u, v in edges:
        if subtrees[u] != subtrees[v]:
            tree.append((u, v))
            subtrees.union(u, v)
    return tree


def numberOfNeighborsInMST(G, vertex):
    number_of_neighbors = 0
    for u in G:
        if len(u) != 2:
            continue
        if u[0] == vertex or u[1] == vertex:
            number_of_neighbors += 1
    return number_of_neighbors


def fleet_speed(ships):
    ships = max(1, int(ships))
    if ships == 1:
        return 1.0
    scaled = math.log(ships) / math.log(1000)
    return 1.0 + (MAX_SPEED - 1.0) * (scaled ** 1.5)


def distance_xy(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def angle_diff(a, b):
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return distance_xy(px, py, ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return distance_xy(px, py, ax + t * dx, ay + t * dy)


def crosses_sun(source, target):
    return point_to_segment_distance(
        CENTER_X, CENTER_Y, source.X(), source.Y(), target.X(), target.Y()
    ) <= SUN_RADIUS + 1.0


class PlanetWarsAdapter:
    """Planet Wars shaped facade over an Orbit Wars observation.

    The original MyBot.py issues source/destination planet orders. Orbit Wars
    wants source/angle orders, so IssueOrder records a converted angle.
    """

    def __init__(self, obs):
        self.obs = obs
        self.player = int(obs.get("player", 0)) if isinstance(obs, dict) else int(obs.player)
        self.step = int(obs.get("step", 0)) if isinstance(obs, dict) else int(getattr(obs, "step", 0))
        self.angular_velocity = (
            float(obs.get("angular_velocity", 0.0))
            if isinstance(obs, dict)
            else float(getattr(obs, "angular_velocity", 0.0))
        )
        self.comet_ids = (
            set(int(x) for x in obs.get("comet_planet_ids", []))
            if isinstance(obs, dict)
            else set(int(x) for x in getattr(obs, "comet_planet_ids", []))
        )
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        raw_initial_planets = (
            obs.get("initial_planets", raw_planets) if isinstance(obs, dict) else getattr(obs, "initial_planets", raw_planets)
        )
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets

        self._initial_planets = {}
        for raw in raw_initial_planets:
            self._initial_planets[int(raw[0])] = Planet(
                int(raw[0]),
                self._map_owner(int(raw[1])),
                int(raw[5]),
                int(raw[6]),
                float(raw[2]),
                float(raw[3]),
                float(raw[4]),
            )

        self._planets = []
        for raw in raw_planets:
            pid = int(raw[0])
            owner = self._map_owner(int(raw[1]))
            x = float(raw[2])
            y = float(raw[3])
            radius = float(raw[4])
            ships = int(raw[5])
            production = int(raw[6])
            self._planets.append(Planet(pid, owner, ships, production, x, y, radius))
        self._planets.sort(key=lambda p: p.PlanetID())
        self._planet_by_id = {p.PlanetID(): p for p in self._planets}

        self._fleets = []
        for raw in raw_fleets:
            owner = self._map_owner(int(raw[1]))
            x = float(raw[2])
            y = float(raw[3])
            angle = float(raw[4])
            source = int(raw[5])
            ships = int(raw[6])
            dest, trip, remaining = self._infer_fleet_destination(x, y, angle, ships, source)
            if dest is not None:
                self._fleets.append(Fleet(owner, ships, source, dest, trip, remaining))

        self.orders = []

    def _map_owner(self, owner):
        if owner < 0:
            return 0
        if owner == self.player:
            return 1
        return 2

    def _infer_fleet_destination(self, x, y, angle, ships, source):
        best = None
        best_eta = None
        for planet in self._planets:
            heading = math.atan2(planet.Y() - y, planet.X() - x)
            dist = distance_xy(x, y, planet.X(), planet.Y())
            tolerance = 0.24 + min(0.35, planet.Radius() / max(8.0, dist))
            if angle_diff(angle, heading) > tolerance:
                continue
            eta = max(1, int(math.ceil(dist / fleet_speed(ships))))
            if best_eta is None or eta < best_eta:
                best = planet.PlanetID()
                best_eta = eta
        if best is None:
            return None, 0, 0
        source_planet = self._planet_by_id.get(source)
        trip = best_eta
        if source_planet is not None:
            trip = max(1, int(math.ceil(self.Distance(source, best) / fleet_speed(ships))))
        return best, trip, best_eta

    def _is_rotating(self, planet):
        initial = self._initial_planets.get(planet.PlanetID())
        if planet.PlanetID() in self.comet_ids or initial is None:
            return False
        orbital_radius = distance_xy(initial.X(), initial.Y(), CENTER_X, CENTER_Y)
        return orbital_radius + initial.Radius() < 50.0

    def predicted_position(self, planet, future_turns):
        if not self._is_rotating(planet):
            return planet.X(), planet.Y()
        initial = self._initial_planets[planet.PlanetID()]
        dx = initial.X() - CENTER_X
        dy = initial.Y() - CENTER_Y
        radius = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) + self.angular_velocity * (self.step + float(future_turns))
        return CENTER_X + radius * math.cos(angle), CENTER_Y + radius * math.sin(angle)

    def InterceptPoint(self, source_planet, destination_planet, num_ships):
        source = self.GetPlanet(source_planet)
        dest = self.GetPlanet(destination_planet)
        speed = fleet_speed(num_ships)
        tx, ty = dest.X(), dest.Y()
        eta = max(1.0, distance_xy(source.X(), source.Y(), tx, ty) / speed)
        for _ in range(4):
            tx, ty = self.predicted_position(dest, eta)
            eta = max(1.0, distance_xy(source.X(), source.Y(), tx, ty) / speed)
        return tx, ty, eta

    def TravelTime(self, source_planet, destination_planet, num_ships=None):
        source = self.GetPlanet(source_planet)
        ships = source.NumShips() if num_ships is None else num_ships
        _, _, eta = self.InterceptPoint(source_planet, destination_planet, max(1, int(ships)))
        return int(math.ceil(eta))

    def NumPlanets(self):
        return len(self._planets)

    def GetPlanet(self, planet_id):
        return self._planet_by_id[int(planet_id)]

    def Planets(self):
        return self._planets

    def MyPlanets(self):
        return [p for p in self._planets if p.Owner() == 1]

    def NeutralPlanets(self):
        return [p for p in self._planets if p.Owner() == 0]

    def EnemyPlanets(self):
        return [p for p in self._planets if p.Owner() > 1]

    def NotMyPlanets(self):
        return [p for p in self._planets if p.Owner() != 1]

    def Fleets(self):
        return self._fleets

    def MyFleets(self):
        return [f for f in self._fleets if f.Owner() == 1]

    def EnemyFleets(self):
        return [f for f in self._fleets if f.Owner() > 1]

    def Distance(self, source_planet, destination_planet):
        source = self.GetPlanet(source_planet)
        destination = self.GetPlanet(destination_planet)
        return int(math.ceil(distance_xy(source.X(), source.Y(), destination.X(), destination.Y())))

    def IssueOrder(self, source_planet, destination_planet, num_ships):
        num_ships = int(num_ships)
        if num_ships <= 0:
            return
        source = self.GetPlanet(source_planet)
        dest = self.GetPlanet(destination_planet)
        if source.Owner() != 1:
            return
        num_ships = min(num_ships, max(0, source.NumShips()))
        if num_ships <= 0:
            return
        target_x, target_y, _eta = self.InterceptPoint(source.PlanetID(), dest.PlanetID(), num_ships)
        angle = math.atan2(target_y - source.Y(), target_x - source.X())
        self.orders.append([source.PlanetID(), angle, num_ships])


def getHowManyShipsComeToPlanet(pw, idPlanet, beforeTurn, idPlayer):
    total_incoming_fleets = 0
    for fleet in pw.Fleets():
        if fleet.DestinationPlanet() != idPlanet or fleet.TurnsRemaining() > beforeTurn:
            continue
        if fleet.Owner() == idPlayer:
            total_incoming_fleets += fleet.NumShips()
        elif idPlayer != 0:
            total_incoming_fleets -= fleet.NumShips()
        else:
            total_incoming_fleets += fleet.NumShips()
    return total_incoming_fleets


def getHowManyShipsComeToPlanetAtTurn(pw, idPlanet, turnNumber, idPlayer):
    total_incoming_fleets = 0
    for fleet in pw.Fleets():
        if fleet.DestinationPlanet() != idPlanet or fleet.TurnsRemaining() != turnNumber:
            continue
        if fleet.Owner() == idPlayer:
            total_incoming_fleets += fleet.NumShips()
        elif idPlayer != 0:
            total_incoming_fleets -= fleet.NumShips()
        else:
            total_incoming_fleets += fleet.NumShips()
    return total_incoming_fleets


def getGrowth(pw, planet, beforeTurn):
    if planet in pw.NeutralPlanets():
        return 0
    if planet in pw.EnemyPlanets():
        return -1 * planet.GrowthRate() * beforeTurn
    if planet in pw.MyPlanets():
        return planet.GrowthRate() * beforeTurn
    return 0


def getClosestPlanetDistance(pw, idPlanet, idPlayer):
    if idPlayer == 0:
        planets = pw.NeutralPlanets()
    elif idPlayer == 1:
        planets = pw.MyPlanets()
    else:
        planets = pw.EnemyPlanets()

    best_distance = 9999
    best_id = 0
    for planet in planets:
        if planet.PlanetID() == idPlanet:
            continue
        distance = pw.Distance(idPlanet, planet.PlanetID())
        if distance < best_distance:
            best_distance = distance
            best_id = planet.PlanetID()
    return best_distance, best_id


def closestPlanetWithSmallerFrontScore(pw, idPlanet, myPlanetsFrontScore, idPlayer):
    planets = pw.MyPlanets() if idPlayer == 1 else pw.EnemyPlanets()
    distance = 9999
    id_neighbor_planet = 0
    for planet in planets:
        pid = planet.PlanetID()
        if pid == idPlanet:
            continue
        if pid not in myPlanetsFrontScore or idPlanet not in myPlanetsFrontScore:
            continue
        d = pw.Distance(idPlanet, pid)
        if d < distance and myPlanetsFrontScore[pid] < myPlanetsFrontScore[idPlanet]:
            distance = d
            id_neighbor_planet = pid
    return distance, id_neighbor_planet


def getClosestPlanets(pw, idPlanet, idPlayer, minDist, maxDist):
    if idPlayer == 0:
        planets = pw.NeutralPlanets()
    elif idPlayer == 1:
        planets = pw.MyPlanets()
    else:
        planets = pw.EnemyPlanets()
    return {
        planet: pw.Distance(idPlanet, planet.PlanetID())
        for planet in planets
        if minDist <= pw.Distance(idPlanet, planet.PlanetID()) <= maxDist
        and planet.PlanetID() != idPlanet
    }


def getFirstOccurenceOfACode(_pw, predictions, code):
    for count, prediction in enumerate(predictions):
        if prediction[0] == code:
            return count
    return len(predictions)


def isItFirstTurn(pw):
    return (
        len(pw.MyPlanets()) == 1
        and len(pw.EnemyPlanets()) >= 1
        and pw.MyPlanets()[0].NumShips() <= 12
    )


def isEnemyPlanetVulnerable(pw, enemyPlanet, myPlanetsSpareShips):
    perimeter = 9
    total_ships = enemyPlanet.NumShips()
    total_ships += getHowManyShipsComeToPlanet(pw, enemyPlanet.PlanetID(), perimeter, 2)
    total_ships += getGrowth(pw, enemyPlanet, perimeter)

    my_attack_planets = {}
    for non_neutral_planet in pw.EnemyPlanets():
        distance = pw.Distance(non_neutral_planet.PlanetID(), enemyPlanet.PlanetID())
        if distance <= perimeter:
            total_ships += non_neutral_planet.NumShips() + getGrowth(
                pw, non_neutral_planet, max(0, perimeter - distance)
            )

    for non_neutral_planet in pw.MyPlanets():
        distance = pw.Distance(non_neutral_planet.PlanetID(), enemyPlanet.PlanetID())
        if distance <= perimeter:
            sendable = myPlanetsSpareShips.get(non_neutral_planet.PlanetID(), 0)
            sendable += getGrowth(pw, non_neutral_planet, max(0, perimeter - distance))
            total_ships -= sendable
            my_attack_planets[non_neutral_planet] = myPlanetsSpareShips.get(
                non_neutral_planet.PlanetID(), 0
            )
            if total_ships < 0:
                break
    return total_ships < 0, my_attack_planets


def enemy_pressure_near(pw, planet, horizon):
    pressure = 0.0
    for enemy in pw.EnemyPlanets():
        if enemy.PlanetID() == planet.PlanetID():
            continue
        eta = max(1, pw.TravelTime(enemy.PlanetID(), planet.PlanetID(), max(1, enemy.NumShips())))
        if eta > horizon:
            continue
        pressure += enemy.NumShips() * (1.0 - float(eta) / float(horizon + 1))
    for fleet in pw.EnemyFleets():
        if fleet.DestinationPlanet() != planet.PlanetID() or fleet.TurnsRemaining() > horizon:
            continue
        pressure += fleet.NumShips() * (1.0 - float(fleet.TurnsRemaining()) / float(horizon + 1))
    return pressure


def sample8_safe_spare(pw, planet, predicted_spare):
    """Lightweight safe_drain analogue from sample8.

    It keeps more reserve on high-production/front planets and under nearby
    enemy pressure, but still leaves the original Planet Wars prediction in
    charge of the baseline spare amount.
    """
    front_eta, _ = getClosestPlanetDistance(pw, planet.PlanetID(), 2)
    pressure = enemy_pressure_near(pw, planet, NUMBER_OF_PREDICTED_TURNS)
    reserve = 1
    if front_eta <= 12:
        reserve += int(max(0.0, 0.06 * pressure))
    if front_eta <= 6:
        reserve += 1 + max(0, planet.GrowthRate() // 2)
    return max(0, min(int(predicted_spare), planet.NumShips() - reserve))


def build_future_predictions(pw, planets):
    planets_in_future = {}
    threatened_planets = []
    won_planets = []
    for planet in planets:
        predictions = []
        code = 0 if planet.Owner() == 0 else (1 if planet.Owner() == 1 else -1)
        num_ships = planet.NumShips()
        owner = planet.Owner()
        for i in range(NUMBER_OF_PREDICTED_TURNS):
            if owner == 0:
                num_ships -= getHowManyShipsComeToPlanetAtTurn(pw, planet.PlanetID(), i, 0)
                if num_ships < 0:
                    if getHowManyShipsComeToPlanetAtTurn(pw, planet.PlanetID(), i, 1) > 0:
                        num_ships = abs(num_ships)
                        code = 2
                        owner = 1
                        if planet not in won_planets:
                            won_planets.append(planet)
                    else:
                        code = -2
                        owner = 2
                        if planet not in threatened_planets:
                            threatened_planets.append(planet)
            else:
                num_ships += getHowManyShipsComeToPlanetAtTurn(pw, planet.PlanetID(), i, 1)
                if owner == 1:
                    if num_ships >= 0:
                        code = 1
                    else:
                        code = -3
                        owner = 2
                        if planet not in threatened_planets:
                            threatened_planets.append(planet)
                else:
                    if num_ships > 0:
                        code = 3
                        owner = 1
                    else:
                        code = -1
            predictions.append((code, num_ships))
            if owner == 1:
                num_ships += planet.GrowthRate()
            elif owner == 2:
                num_ships -= planet.GrowthRate()
        planets_in_future[planet.PlanetID()] = predictions
    return planets_in_future, threatened_planets, won_planets


def DoTurn(pw):
    if len(pw.MyPlanets()) == 0:
        return

    first_turn = isItFirstTurn(pw)

    non_neutral_planets_graph = {}
    non_neutral_planets = pw.MyPlanets() + pw.EnemyPlanets()
    for u in non_neutral_planets:
        non_neutral_planets_graph[u.PlanetID()] = {}
        for v in non_neutral_planets:
            non_neutral_planets_graph[u.PlanetID()][v.PlanetID()] = pw.Distance(
                u.PlanetID(), v.PlanetID()
            )
    non_neutral_planets_mst = MinimumSpanningTree(non_neutral_planets_graph)

    my_planets_graph = {}
    for u in pw.MyPlanets():
        my_planets_graph[u.PlanetID()] = {}
        for v in pw.MyPlanets():
            my_planets_graph[u.PlanetID()][v.PlanetID()] = pw.Distance(
                u.PlanetID(), v.PlanetID()
            )
    my_planets_mst = MinimumSpanningTree(my_planets_graph)

    my_front_planets = []
    enemy_front_planets = []
    for u in non_neutral_planets_mst:
        if len(u) != 2:
            continue
        p0 = pw.GetPlanet(u[0])
        p1 = pw.GetPlanet(u[1])
        if p0 in pw.MyPlanets() and p1 in pw.EnemyPlanets():
            my_front_planets.append(p0)
            enemy_front_planets.append(p1)
        if p0 in pw.EnemyPlanets() and p1 in pw.MyPlanets():
            my_front_planets.append(p1)
            enemy_front_planets.append(p0)

    my_planets_front_score = {}
    for my_planet in pw.MyPlanets():
        my_planets_front_score[my_planet.PlanetID()], _ = getClosestPlanetDistance(
            pw, my_planet.PlanetID(), 2
        )

    my_planets_in_future, my_threatened_planets, _ = build_future_predictions(
        pw, pw.MyPlanets()
    )
    if len(my_threatened_planets) == 1 and len(pw.MyPlanets()) == 1:
        return
    if len(my_threatened_planets) == len(pw.MyPlanets()):
        return

    neutral_planets_in_future, neutral_threatened_planets, neutral_won_planets = (
        build_future_predictions(pw, pw.NeutralPlanets())
    )

    my_planets_spare_ships = {}
    for my_planet in pw.MyPlanets():
        future = my_planets_in_future.get(my_planet.PlanetID(), [(1, my_planet.NumShips())])
        spare_ships = min(step[1] for step in future)
        my_planets_spare_ships[my_planet.PlanetID()] = sample8_safe_spare(
            pw, my_planet, max(0, int(spare_ships))
        )

    for my_threatened_planet in my_threatened_planets:
        if len(pw.MyPlanets()) <= 1:
            continue
        time_remaining = getFirstOccurenceOfACode(
            pw, my_planets_in_future[my_threatened_planet.PlanetID()], -3
        )
        closest_planets = getClosestPlanets(
            pw, my_threatened_planet.PlanetID(), 1, 0, max(0, time_remaining + 1)
        )
        ships_needed = (
            getHowManyShipsComeToPlanet(
                pw, my_threatened_planet.PlanetID(), NUMBER_OF_PREDICTED_TURNS, 2
            )
            - my_threatened_planet.NumShips()
        )
        for helper_planet in closest_planets:
            if ships_needed < 0:
                break
            num_ships = max(
                0,
                min(
                    ships_needed + 1,
                    my_planets_spare_ships.get(helper_planet.PlanetID(), 0),
                    helper_planet.NumShips() - 1,
                ),
            )
            pw.IssueOrder(helper_planet.PlanetID(), my_threatened_planet.PlanetID(), num_ships)
            my_planets_spare_ships[helper_planet.PlanetID()] -= num_ships
            helper_planet.RemoveShips(num_ships)

    for neutral_threatened_planet in neutral_threatened_planets:
        distance1, id_planet1 = getClosestPlanetDistance(
            pw, neutral_threatened_planet.PlanetID(), 1
        )
        distance2, _ = getClosestPlanetDistance(pw, neutral_threatened_planet.PlanetID(), 2)
        if distance1 <= distance2 and id_planet1 in pw._planet_by_id:
            first_occurrence = getFirstOccurenceOfACode(
                pw, neutral_planets_in_future[neutral_threatened_planet.PlanetID()], -2
            )
            my_source = pw.GetPlanet(id_planet1)
            if my_source in my_threatened_planets:
                continue
            num_ships = (
                neutral_threatened_planet.GrowthRate()
                + 2
                + (
                    getHowManyShipsComeToPlanet(
                        pw, neutral_threatened_planet.PlanetID(), first_occurrence + 1, 2
                    )
                    - neutral_threatened_planet.NumShips()
                )
            )
            if (
                num_ships > 0
                and my_source.NumShips() > num_ships
                and distance1 - 2 < first_occurrence < distance1 + 2
            ):
                pw.IssueOrder(id_planet1, neutral_threatened_planet.PlanetID(), num_ships)
                my_planets_spare_ships[my_source.PlanetID()] -= num_ships
                my_source.RemoveShips(num_ships)

    for enemy_front_planet in enemy_front_planets:
        do_we_attack, my_attack_planets = isEnemyPlanetVulnerable(
            pw, enemy_front_planet, my_planets_spare_ships
        )
        if not do_we_attack:
            continue
        for my_attack_planet in my_attack_planets:
            if my_attack_planet in my_threatened_planets:
                continue
            num_ships = min(
                my_attack_planet.NumShips() - 1,
                my_attack_planets[my_attack_planet],
            )
            if num_ships <= 0:
                continue
            pw.IssueOrder(my_attack_planet.PlanetID(), enemy_front_planet.PlanetID(), num_ships)
            my_planets_spare_ships[my_attack_planet.PlanetID()] -= num_ships
            my_attack_planet.RemoveShips(num_ships)

    planets_already_targeted = []
    iteration = 5
    if first_turn and pw.MyPlanets() and pw.EnemyPlanets():
        my_planet = pw.MyPlanets()[0]
        enemy_planet = pw.EnemyPlanets()[0]
        if pw.Distance(my_planet.PlanetID(), enemy_planet.PlanetID()) < 18:
            iteration = 1

    for _ in range(iteration):
        source = None
        source_score = -999999.0
        for p in pw.MyPlanets():
            if p in my_threatened_planets:
                continue
            score = float(p.NumShips())
            if score > source_score:
                source_score = score
                source = p
        if source is None:
            break

        dest = None
        dest_score = -999999.0
        for p in pw.NeutralPlanets():
            if planets_already_targeted.count(p.PlanetID()) > 0 and first_turn:
                continue
            if p in neutral_won_planets or p in neutral_threatened_planets:
                continue
            if getHowManyShipsComeToPlanet(pw, p.PlanetID(), MAX_TURNS, 1) > 20:
                continue
            if crosses_sun(source, p):
                continue
            distance1 = pw.Distance(source.PlanetID(), p.PlanetID())
            distance2, _ = getClosestPlanetDistance(pw, p.PlanetID(), 2)
            if distance1 >= distance2:
                continue
            score = (1.0 + p.GrowthRate()) / (p.NumShips() + 3 * distance1)
            if score > dest_score:
                dest_score = score
                dest = p

        if (
            dest_score > 0.006
            and source.NumShips() > 1
            and dest is not None
            and source not in my_threatened_planets
        ):
            planets_already_targeted.append(dest.PlanetID())
            num_ships = dest.NumShips() + 1
            if (
                num_ships > 0
                and source.NumShips() > num_ships
                and my_planets_spare_ships.get(source.PlanetID(), 0) > num_ships
            ):
                pw.IssueOrder(source.PlanetID(), dest.PlanetID(), num_ships)
                my_planets_spare_ships[source.PlanetID()] -= num_ships
                source.RemoveShips(num_ships)
        else:
            break

    my_back_planets = list(pw.MyPlanets())
    front_planets = list(dict.fromkeys(my_front_planets))
    for p_enemy in pw.EnemyPlanets():
        if len(pw.MyPlanets()) <= 1:
            continue
        closest_planet = None
        distance = 99999
        for p in pw.MyPlanets():
            d = pw.Distance(p_enemy.PlanetID(), p.PlanetID())
            if d < distance:
                closest_planet = p
                distance = d
        if closest_planet is None:
            continue
        if closest_planet not in front_planets:
            front_planets.append(closest_planet)
        if closest_planet in my_back_planets:
            my_back_planets.remove(closest_planet)

    for my_threatened_planet in my_threatened_planets:
        if my_threatened_planet in my_back_planets:
            my_back_planets.remove(my_threatened_planet)

    for p_back in my_back_planets:
        if p_back in my_threatened_planets or not front_planets:
            continue
        closest_front_planet = min(
            front_planets,
            key=lambda p_front: pw.Distance(p_back.PlanetID(), p_front.PlanetID()),
        )
        distance = pw.Distance(p_back.PlanetID(), closest_front_planet.PlanetID())
        num_ships = min(int(p_back.NumShips() / 2), my_planets_spare_ships.get(p_back.PlanetID(), 0))
        if num_ships <= 0 or num_ships >= p_back.NumShips():
            continue
        if crosses_sun(p_back, closest_front_planet):
            continue
        if distance < 14:
            pw.IssueOrder(p_back.PlanetID(), closest_front_planet.PlanetID(), num_ships)
            my_planets_spare_ships[p_back.PlanetID()] -= num_ships
            p_back.RemoveShips(num_ships)
            continue

        smallest_front_score = 9999
        smallest_front_score_neighbor_planet = None
        neighbor_planet = None
        for u in my_planets_mst:
            if len(u) != 2:
                continue
            if u[0] == p_back.PlanetID():
                neighbor_planet = pw.GetPlanet(u[1])
            elif u[1] == p_back.PlanetID():
                neighbor_planet = pw.GetPlanet(u[0])
            else:
                continue
            score = my_planets_front_score.get(neighbor_planet.PlanetID(), 9999)
            if score < smallest_front_score:
                smallest_front_score = score
                smallest_front_score_neighbor_planet = neighbor_planet

        if (
            neighbor_planet is not None
            and numberOfNeighborsInMST(my_planets_mst, p_back.PlanetID()) <= 1
            and my_planets_front_score.get(neighbor_planet.PlanetID(), 9999)
            > my_planets_front_score.get(p_back.PlanetID(), 9999)
        ):
            dist, id_neighbor_planet = closestPlanetWithSmallerFrontScore(
                pw, p_back.PlanetID(), my_planets_front_score, 1
            )
            if dist != 9999:
                smallest_front_score_neighbor_planet = pw.GetPlanet(id_neighbor_planet)

        closest_neutral_planet_distance, _ = getClosestPlanetDistance(pw, p_back.PlanetID(), 0)
        if (
            smallest_front_score_neighbor_planet is not None
            and (p_back.NumShips() > 80 or closest_neutral_planet_distance > 10)
            and not crosses_sun(p_back, smallest_front_score_neighbor_planet)
        ):
            pw.IssueOrder(
                p_back.PlanetID(),
                smallest_front_score_neighbor_planet.PlanetID(),
                num_ships,
            )
            my_planets_spare_ships[p_back.PlanetID()] -= num_ships
            p_back.RemoveShips(num_ships)


_LOCAL_STEP = 0
_LAST_START_SIGNATURE = None
_SAMPLE8_AGENT = None


def _load_sample8_agent():
    global _SAMPLE8_AGENT
    if _SAMPLE8_AGENT is not None:
        return _SAMPLE8_AGENT
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    sample8_candidates = [
        os.path.abspath(os.path.join(here, "..", "sample8", "main.py")),
        os.path.abspath(os.path.join(here, "sample8", "main.py")),
        os.path.abspath(os.path.join(os.getcwd(), "sample8", "main.py")),
        os.path.abspath(os.path.join(os.getcwd(), "..", "sample8", "main.py")),
        r"C:\Users\yuu98\Desktop\kaggle\orbit-wars\sample8\main.py",
    ]
    sample8_path = next((path for path in sample8_candidates if os.path.exists(path)), None)
    if sample8_path is None:
        _SAMPLE8_AGENT = False
        return None
    try:
        spec = importlib.util.spec_from_file_location("_orbit_wars_sample8_for_pw_mix", sample8_path)
        if spec is None or spec.loader is None:
            _SAMPLE8_AGENT = False
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SAMPLE8_AGENT = getattr(module, "agent", None) or False
    except Exception:
        _SAMPLE8_AGENT = False
    return _SAMPLE8_AGENT if _SAMPLE8_AGENT is not False else None


def _with_local_step(obs):
    global _LOCAL_STEP, _LAST_START_SIGNATURE
    if not isinstance(obs, dict):
        return obs
    if "step" in obs:
        return obs

    player = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    my_planets = [p for p in planets if int(p[1]) == player]
    owned_ids = tuple(sorted(int(p[0]) for p in my_planets))
    start_like = len(my_planets) == 1 and not fleets and my_planets[0][5] <= 12
    signature = (player, owned_ids, len(planets))
    if start_like and signature != _LAST_START_SIGNATURE:
        _LOCAL_STEP = 0
        _LAST_START_SIGNATURE = signature

    copied = dict(obs)
    copied["step"] = _LOCAL_STEP
    _LOCAL_STEP += 1
    return copied


def agent(obs):
    try:
        original_obs = obs
        sample8_agent = _load_sample8_agent()
        sample8_orders = []
        if sample8_agent is not None:
            try:
                sample8_orders = sample8_agent(original_obs) or []
            except Exception:
                sample8_orders = []
        if sample8_orders:
            return sample8_orders

        obs = _with_local_step(original_obs)
        pw = PlanetWarsAdapter(obs)
        DoTurn(pw)
        return pw.orders
    except Exception:
        return []
