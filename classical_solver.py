"""
Classical CVRP solver — shared module for yale.ipynb and quantum_hybrid_solver.ipynb.
All classical components live here. The quantum notebook imports these and only
replaces build_route with a QAOA-based version.
"""

import math
import os
from itertools import permutations


def encode_locations(depot, customers):
    locations = {0: tuple(depot)}
    for i, c in enumerate(customers, start=1):
        locations[i] = tuple(c)
    return locations


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def build_distance_matrix(locations):
    dist_matrix = {}
    ids = list(locations.keys())
    for i in ids:
        for j in ids:
            if i != j:
                dist_matrix[(i, j)] = euclidean(locations[i], locations[j])
    return dist_matrix


def compute_num_groups(H, C, N):
    return min(math.ceil(H / C), N)


def _route_distance_estimate(group, dist_matrix):
    unvisited = list(group)
    current = 0
    total = 0.0
    while unvisited:
        nearest = min(unvisited, key=lambda c: dist_matrix[(current, c)])
        total += dist_matrix[(current, nearest)]
        current = nearest
        unvisited.remove(nearest)
    total += dist_matrix[(current, 0)]
    return total


def cluster_houses(locations, dist_matrix, G, C):
    customer_ids = [n for n in locations if n != 0]
    H = len(customer_ids)
    depot_x, depot_y = locations[0]
    sorted_customers = sorted(
        customer_ids,
        key=lambda n: math.atan2(locations[n][1] - depot_y, locations[n][0] - depot_x)
    )
    chunk_size = min(math.ceil(H / G), C)
    best_groups, best_dist = None, float('inf')
    for start in range(H):
        rotated = sorted_customers[start:] + sorted_customers[:start]
        groups = [rotated[i:i + chunk_size] for i in range(0, H, chunk_size)]
        while len(groups) > G:
            groups[-2] = groups[-2] + groups[-1]
            groups.pop()
        groups = [g for g in groups if g]
        total = sum(_route_distance_estimate(g, dist_matrix) for g in groups)
        if total < best_dist:
            best_dist = total
            best_groups = [list(g) for g in groups]
    return best_groups


def assign_vehicles(groups, N):
    assignment = {v: [] for v in range(1, N + 1)}
    for i, group in enumerate(groups):
        assignment[(i % N) + 1].append(group)
    return assignment


def build_route(group, dist_matrix):
    """Exact TSP via brute-force permutations (feasible for small groups capped at C)."""
    best_route, best_dist = None, float('inf')
    for perm in permutations(group):
        route = [0] + list(perm) + [0]
        dist = sum(dist_matrix[(route[i], route[i + 1])] for i in range(len(route) - 1))
        if dist < best_dist:
            best_dist, best_route = dist, route
    return best_route


def build_all_routes(assignment, dist_matrix):
    routes = []
    for v in sorted(assignment):
        for group in assignment[v]:
            routes.append((v, build_route(group, dist_matrix)))
    return routes


def total_distance(routes, dist_matrix):
    return sum(
        dist_matrix[(route[i], route[i + 1])]
        for _, route in routes
        for i in range(len(route) - 1)
    )


def solve_cvrp(depot, customers, N, C, verbose=True):
    locations   = encode_locations(depot, customers)
    dist_matrix = build_distance_matrix(locations)
    H = len(customers)
    G = compute_num_groups(H, C, N)

    if verbose:
        print(f"G = min(ceil({H}/{C}), {N}) = {G} vehicle(s)")

    groups     = cluster_houses(locations, dist_matrix, G, C)
    assignment = assign_vehicles(groups, N)

    if verbose:
        print("Groups:")
        for i, g in enumerate(groups):
            print(f"  G_{i+1}: customers {g}")
        print("Vehicle assignments:")
        for v, grps in assignment.items():
            print(f"  Vehicle {v}: {grps}")

    routes = build_all_routes(assignment, dist_matrix)
    dist   = total_distance(routes, dist_matrix)

    if verbose:
        print("Routes:")
        for i, (v, route) in enumerate(routes):
            print(f"  r{i+1} (vehicle {v}): {' -> '.join(str(n) for n in route)}")
        print(f"Total distance: {dist:.4f}")

    return routes, dist
