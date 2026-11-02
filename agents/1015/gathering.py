"""Exact local gathering supervision sampled from the learner's visible territory.

A query fixes a friendly induced subgraph, a destination and an action budget.
Its static transport problem holds ownership and armies fixed between commands.
The host solves these small problems and supplies sparse auxiliary labels.
"""

from functools import lru_cache
import time

import jax.numpy as jnp
from jax.nn import logsumexp
import numpy as np

NODES = 8
ACTIONS = NODES * 8 + 1
TEMPERATURE = 0.1
FIELDS = ("nodes", "neighbors", "edges", "valid", "budget")


@lru_cache(maxsize=4096)
def connected_sets(adjacency):
    """Enumerate connected vertex sets containing destination zero."""
    n = len(adjacency)
    subsets = []
    for mask in range(1, 1 << n, 2):
        reached = 1
        while True:
            expanded = reached
            for u in range(n):
                if reached & (1 << u):
                    expanded |= adjacency[u] & mask
            if expanded == reached:
                break
            reached = expanded
        if reached == mask:
            subsets.append([(mask >> u) & 1 for u in range(n)])
    return np.asarray(subsets, np.int64)


def solve(army, neighbors, edges, budget):
    """Exact target mass and first-command values, including full, half and pass.

    For a connected set S containing the target, a full-move postorder schedule
    leaves one army on each other selected cell. Maximizing this mass over sets
    of at most budget+1 vertices gives the static gathering optimum.
    """
    army = np.asarray(army, np.int64)
    n = len(army)
    adjacency = tuple(
        sum(1 << int(v) for v in neighbors[u, edges[u]]) for u in range(n)
    )
    subsets = connected_sets(adjacency)
    cost = subsets.sum(-1) - 1

    def optimum(states, remaining):
        selected = subsets[cost <= remaining].copy()
        selected[:, 0] = 0
        return states[:, 0] + ((states - 1) @ selected.T).max(-1)

    best = int(optimum(army[None], budget)[0])
    legal = np.zeros(ACTIONS, bool)
    successor = np.broadcast_to(army, (ACTIONS, n)).copy()
    legal[-1] = True
    for u in range(n):
        if army[u] < 2:
            continue
        for direction in np.flatnonzero(edges[u]):
            v = neighbors[u, direction]
            for half in range(2):
                action = u * 8 + direction * 2 + half
                moved = army[u] // 2 if half else army[u] - 1
                successor[action, u] -= moved
                successor[action, v] += moved
                legal[action] = True
    values = np.zeros(ACTIONS, np.int64)
    values[legal] = optimum(successor[legal], max(budget - 1, 0))
    optimal = legal & (values == best)
    if budget < 1 or not optimal.any():
        raise ValueError("Gathering queries require a positive budget and an optimum")
    scale = max(int(army.sum()), 1)
    return values.astype(np.float32) / scale, best / scale, legal, optimal


def empty(count):
    """Row zero is an inactive, finite query for ordinary PPO samples."""
    return {
        "nodes": np.zeros((count, NODES), np.int32),
        "neighbors": np.zeros((count, NODES, 4), np.int32),
        "edges": np.zeros((count, NODES, 4), bool),
        "valid": np.zeros((count, NODES), bool),
        "budget": np.ones(count, np.float32),
        "q": np.zeros((count, ACTIONS), np.float32),
        "value": np.zeros(count, np.float32),
        "legal": np.tile(np.arange(ACTIONS) == ACTIONS - 1, (count, 1)),
        "optimal": np.tile(np.arange(ACTIONS) == ACTIONS - 1, (count, 1)),
        "weight": np.zeros(count, np.float32),
    }


def prepare(data, graph, count, seed):
    """Build a bounded CPU task bank with its own RNG and no simulator access."""
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    indices = rng.choice(data.count, min(count, data.count), replace=False)
    bank = empty(len(indices) + 1)
    lookup = np.zeros(data.count, np.int32)
    neighbors = np.asarray(graph.neighbors)[:, :4]
    geometric = np.asarray(graph.valid)[:, :4]
    accepted = 0
    for index in indices:
        tick, lane = divmod(int(index), data.lanes)
        own = (data.flags[tick, lane] & 2) != 0
        owned = np.flatnonzero(own)
        if len(owned) < 2:
            continue
        nodes = [int(rng.choice(owned))]
        limit = int(rng.integers(2, NODES + 1))
        while len(nodes) < limit:
            frontier = sorted({
                int(v) for u in nodes for v in neighbors[u, geometric[u]]
                if own[v] and v not in nodes
            })
            if not frontier:
                break
            nodes.append(int(rng.choice(frontier)))
        if len(nodes) < 2:
            continue
        accepted += 1
        row = accepted
        lookup[index] = row
        local = {u: i for i, u in enumerate(nodes)}
        bank["nodes"][row, :len(nodes)] = nodes
        bank["valid"][row, :len(nodes)] = True
        for u, cell in enumerate(nodes):
            for direction in range(4):
                v = int(neighbors[cell, direction])
                if geometric[cell, direction] and v in local:
                    bank["neighbors"][row, u, direction] = local[v]
                    bank["edges"][row, u, direction] = True
        army = np.rint(np.expm1(
            data.army[tick + data.archive_prefix, lane, nodes].astype(np.float64) * np.log(256.0)
        )).astype(np.int64)
        budget = int(rng.integers(1, len(nodes)))
        bank["budget"][row] = budget
        q, value, legal, optimal = solve(
            army, bank["neighbors"][row, :len(nodes)],
            bank["edges"][row, :len(nodes)], budget,
        )
        bank["q"][row], bank["value"][row] = q, value
        bank["legal"][row], bank["optimal"][row] = legal, optimal
    # This factor makes minibatch means estimate the mean over sampled tasks.
    bank["weight"][1:accepted + 1] = data.count / max(accepted, 1)
    data.gathering = {k: v[:accepted + 1].copy() for k, v in bank.items()}
    data.gathering_index = lookup
    return {
        "gathering_tasks": accepted,
        "gathering_label_seconds": time.monotonic() - started,
        "gathering_storage_mib": (lookup.nbytes + sum(
            x.nbytes for x in data.gathering.values()
        )) / 2**20,
    }


def losses(prediction, targets):
    """Regress normalized mass and put probability on every tied optimal command."""
    q, value = prediction[..., :-1], prediction[..., -1]
    legal = targets["gathering_legal"]
    optimal = targets["gathering_optimal"]
    weight = targets["gathering_weight"]
    error = ((q - targets["gathering_q"]) ** 2 * legal).sum(-1)
    error /= jnp.maximum(legal.sum(-1), 1)
    error += (value - targets["gathering_value"]) ** 2
    logits = jnp.where(legal, q / TEMPERATURE, -1e9)
    ranking = logsumexp(logits, -1) - logsumexp(
        jnp.where(optimal, logits, -1e9), -1
    )
    choice = logits.argmax(-1)
    agreement = jnp.take_along_axis(optimal, choice[..., None], -1)[..., 0]
    regret = targets["gathering_value"] - jnp.take_along_axis(
        targets["gathering_q"], choice[..., None], -1
    )[..., 0]
    total = error + 0.1 * ranking
    return jnp.stack([
        (total * weight).mean(), (error * weight).mean(),
        (agreement * weight).mean(), (regret * weight).mean(),
    ])
