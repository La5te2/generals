"""Frozen observation-based potential and independent per-episode factual memory."""

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

SCHEMA = "observation-hce-research-v2"
RAW_FEATURES = (
    "land", "army", "production_lo", "production_hi", "production_exact",
    "separation", "transport_distance", "mobile_present", "terrain_known", "vision",
    "exposure", "enemy_general_known", "home_general_exposed", "tick", "land_wait",
    "production_wait", "short_side", "long_side", "army_total", "land_total",
    "production_total_lo", "production_total_hi",
)


def basis(raw, xp=jnp):
    r = dict(zip(RAW_FEATURES, raw))
    terms = {"intercept": xp.asarray(1.0), "land": r["land"], "army": r["army"],
             "production_center": (r["production_lo"] + r["production_hi"]) / 2,
             "production_width": (r["production_hi"] - r["production_lo"]) / 2,
             "production_exact": r["production_exact"], "separation": r["separation"],
             "transport_distance": r["transport_distance"], "mobile_present": r["mobile_present"]}
    information = ("terrain_known", "vision", "exposure", "enemy_general_known", "home_general_exposed")
    terms.update({name: r[name] for name in information})
    states = tuple(name for name in terms if name != "intercept")
    terms.update({"time": xp.log1p(r["tick"]), "land_wait": r["land_wait"],
                  "production_wait": r["production_wait"], "short_side": xp.log1p(r["short_side"]),
                  "long_side": xp.log1p(r["long_side"]), "army_total": xp.log1p(r["army_total"]),
                  "land_total": xp.log1p(r["land_total"]),
                  "production_total_center": (xp.log1p(r["production_total_lo"]) + xp.log1p(r["production_total_hi"])) / 2,
                  "production_total_width": (xp.log1p(r["production_total_hi"]) - xp.log1p(r["production_total_lo"])) / 2})
    economic = ("land", "army", "production_center")

    def product(a, b):
        terms[a + "*" + b] = terms[a] * terms[b]

    for i, a in enumerate(economic):
        for b in economic[i + 1:]:
            product(a, b)
    for spatial in ("separation", "transport_distance"):
        for e in economic:
            product(spatial, e)
    for info in information:
        product(info, "army")
    for state in states:
        product(state, "time")
    for e in economic:
        product(e, "army_total")
    product("transport_distance", "short_side")
    product("transport_distance", "long_side")
    product("land", "land_wait")
    product("production_center", "production_wait")
    product("separation", "transport_distance")
    return xp.stack(list(terms.values())), tuple(terms)


def load_parameters(path):
    payload = Path(path).read_bytes()
    data = json.loads(payload)
    _, names = basis(np.zeros(len(RAW_FEATURES)), np)
    if data["schema"] != SCHEMA or data["raw_features"] != list(RAW_FEATURES) or data["terms"] != list(names):
        raise ValueError("Potential feature schema differs from fitted parameters")
    weights = np.asarray(data["weights"], np.float32)
    if weights.shape != (len(names),) or not np.isfinite(weights).all():
        raise ValueError("Invalid potential coefficients")
    return weights, hashlib.sha256(payload).hexdigest()


class View(NamedTuple):
    terrain: jax.Array
    owner: jax.Array
    army: jax.Array
    totals: jax.Array
    land: jax.Array
    tick: jax.Array
    player: jax.Array


class Memory(NamedTuple):
    terrain: jax.Array
    enemy_known: jax.Array
    home_exposed: jax.Array
    lower: jax.Array
    upper: jax.Array
    general: jax.Array
    difference: jax.Array
    own_loss: jax.Array
    enemy_loss: jax.Array
    tick: jax.Array
    value: jax.Array


def empty(area):
    return Memory(jnp.zeros(area, jnp.int8), jnp.array(False), jnp.array(False),
                  jnp.array(1.0), jnp.array(1.0), jnp.array(-1, jnp.int32),
                  jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
                  jnp.array(-1, jnp.int32), jnp.array(0.0))


def observe(state, player, topology):
    """Construct public totals and a masked board with integer-valued armies."""
    ownership = state.ownership.reshape(2, -1)
    armies = state.armies.reshape(-1).astype(jnp.float32)
    own = ownership[player]
    visible = own | jnp.any(own[topology.neighbors] & topology.valid, -1)
    mountains, cities, generals = (x.reshape(-1) for x in (state.mountains, state.castles, state.generals))
    terrain = jnp.where(generals, 4, jnp.where(cities, 3, jnp.where(mountains, 2, 1)))
    terrain = jnp.where(visible, terrain, jnp.where(mountains | cities, 5, 0)).astype(jnp.int8)
    owner = jnp.where(ownership[0], 0, jnp.where(ownership[1], 1, -1))
    return View(terrain, jnp.where(visible, owner, -1).astype(jnp.int8),
                jnp.where(visible, armies, 0), (ownership * armies).sum(-1),
                ownership.sum(-1), state.time.astype(jnp.int32), player)


def neutral_upper(view, player, topology):
    visible = (view.terrain != 0) & (view.terrain != 5)
    source = visible & (view.owner == player)
    neighbors = topology.neighbors[:, :4]
    known = visible[neighbors]
    neutral = (view.owner[neighbors] < 0) & (view.terrain[neighbors] != 2)
    mobile = jnp.maximum(view.army - 1, 0)
    loss = jnp.where(known, jnp.where(neutral, jnp.minimum(mobile[:, None], view.army[neighbors]), 0), mobile[:, None])
    loss = jnp.where(source[:, None] & topology.valid[:, :4], loss, 0)
    hidden = view.totals[player] - jnp.sum(jnp.where(source, view.army, 0))
    return jnp.maximum(loss.max(), jnp.maximum(hidden - 1, 0))


def transport(own, mobile, topology):
    """Exact BFS in compact source blocks, with boolean workspace and weighted sums."""
    n = len(own)
    block = min(n, 32)
    count = jnp.sum(mobile > 0)
    sources = jnp.pad(jnp.nonzero(mobile > 0, size=n, fill_value=0)[0], (0, block))

    def source_block(index, totals):
        indices = jax.lax.dynamic_slice(sources, (index * block,), (block,))
        active = index * block + jnp.arange(block) < count
        mass_source = jnp.where(active, mobile[indices], 0)
        frontier = (jnp.arange(n)[None, :] == indices[:, None]) & active[:, None]

        def expand(carry):
            frontier, visited, distance, pairs, weighted = carry
            reached = jnp.zeros_like(frontier)
            for d in range(4):
                reached |= frontier[:, topology.neighbors[:, d]] & topology.valid[None, :, d]
            reached &= own[None, :] & ~visited
            mass = jnp.sum(mass_source * jnp.sum(jnp.where(reached, mobile[None, :], 0), -1))
            return reached, visited | reached, distance + 1, pairs + mass, weighted + (distance + 1) * mass

        _, _, _, pairs, weighted = jax.lax.while_loop(
            lambda c: jnp.any(c[0]), expand,
            (frontier, frontier, jnp.array(0, jnp.int32), jnp.sum(mass_source**2), jnp.array(0.0)))
        return totals + jnp.array([pairs, weighted])

    pairs, weighted = jax.lax.fori_loop(0, (count + block - 1) // block, source_block, jnp.zeros(2))
    total = mobile.sum()
    separation = jnp.where(total > 0, 1 - pairs / jnp.maximum(total * total, 1), 0)
    return jnp.clip(separation, 0, 1), weighted / jnp.maximum(pairs, 1)


def update(view, previous, topology, height, width):
    """Consume one consecutive observation and return raw features plus memory."""
    p, q = view.player, 1 - view.player
    visible = (view.terrain != 0) & (view.terrain != 5)
    own, enemy = view.owner == p, (view.owner == q) & visible
    producer = (view.terrain == 3) | (view.terrain == 4)
    known = jnp.where(visible, view.terrain, previous.terrain)
    enemy_known = previous.enemy_known | jnp.any(enemy & (view.terrain == 4))
    home = own & (view.terrain == 4)
    general = jnp.where(home.any(), jnp.argmax(home), -1).astype(jnp.int32)
    cover = jnp.any(enemy[topology.neighbors] & topology.valid, -1)
    home_exposed = previous.home_exposed | jnp.any(home & cover)
    own_production = jnp.sum(own & producer).astype(jnp.float32)
    seen_production = jnp.sum(enemy & producer)
    possible = (known == 3) | (known == 4) | ((known == 0) & ((view.terrain == 5) | ~enemy_known))
    lower = jnp.maximum(1, seen_production).astype(jnp.float32)
    upper = jnp.minimum(view.land[q], seen_production + jnp.sum(~visible & possible)).astype(jnp.float32)
    ordinary = (previous.tick >= 0) & (previous.general == general)
    lower = jnp.maximum(lower, jnp.where(ordinary, previous.lower - 2, lower))
    upper = jnp.minimum(upper, jnp.where(ordinary, previous.upper + 2, upper))
    difference = view.totals[p] - view.totals[q]
    center = own_production + jnp.where(view.tick % 50 == 0, view.land[p] - view.land[q], 0) - (difference - previous.difference)
    recruitment = ordinary & (view.tick % 2 == 0)
    lower = jnp.maximum(lower, jnp.where(recruitment, center - previous.own_loss, lower))
    upper = jnp.minimum(upper, jnp.where(recruitment, center + previous.enemy_loss, upper))
    mobile = jnp.where(own, jnp.maximum(view.army - 1, 0), 0)
    separation, distance = transport(own, mobile, topology)

    def ratio(a, b):
        return jnp.clip(jnp.log(jnp.maximum(a, 1) / jnp.maximum(b, 1)) / jnp.log(4.0), -1, 1)

    raw = jnp.array([
        ratio(view.land[p], view.land[q]), ratio(view.totals[p], view.totals[q]),
        ratio(own_production, upper), ratio(own_production, lower), lower == upper,
        separation, distance, jnp.any(mobile > 0), (known != 0).mean(), visible.mean(),
        jnp.sum(own & cover) / jnp.maximum(view.land[p], 1), enemy_known, home_exposed,
        view.tick, (50 - view.tick % 50) / 50, (2 - view.tick % 2) / 2,
        min(height, width), max(height, width), view.totals.sum(), view.land.sum(),
        own_production + lower, own_production + upper,
    ], jnp.float32)
    memory = Memory(known, enemy_known, home_exposed, lower, upper, general, difference,
                    neutral_upper(view, p, topology), neutral_upper(view, q, topology),
                    view.tick, previous.value)
    return raw, memory


def evaluate(raw, weights):
    terms, _ = basis(raw)
    return jnp.tanh(jnp.dot(terms, weights) / 2)
