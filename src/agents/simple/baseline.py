"""Batch the baseline algorithm in JAX, using only each player's masked observations and remembered routes."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Memory(NamedTuple):
    terrain: jax.Array
    owner: jax.Array
    army: jax.Array
    seen: jax.Array
    general: jax.Array
    route: jax.Array
    length: jax.Array


class Plan(NamedTuple):
    path: jax.Array
    length: jax.Array
    army: jax.Array
    gain: jax.Array


def empty(count, area):
    return Memory(jnp.zeros((count, area), jnp.int32), jnp.zeros((count, area), jnp.int32),
                  jnp.zeros((count, area), jnp.int32), jnp.zeros((count, area), jnp.bool_),
                  jnp.full(count, -1, jnp.int32), jnp.full((count, area + 1), -1, jnp.int32),
                  jnp.zeros(count, jnp.int32))


def choose(memory, terrain, owner, army, inside, tick, reset):
    side = terrain.shape[0]
    area = side * side
    terrain, owner, army, inside = (value.reshape(-1) for value in (terrain, owner, army, inside))
    memory = jax.tree.map(lambda old, fresh: jnp.where(reset, fresh[0], old), memory, empty(1, area))
    visible = inside & (terrain != 0) & (terrain != 5)
    remember = visible | ~memory.seen
    memory = memory._replace(terrain=jnp.where(remember, terrain, memory.terrain),
                             owner=jnp.where(remember, owner, memory.owner),
                             army=jnp.where(remember, army, memory.army), seen=memory.seen | visible)
    positions = jnp.arange(area)
    general = jnp.max(jnp.where(visible & (terrain == 4) & (owner == 1), positions, -1))
    general = jnp.where(general >= 0, general, memory.general)
    memory = memory._replace(general=general)
    neighbors = jnp.stack((positions - side, positions + side, positions - 1, positions + 1), axis=1)
    valid = jnp.stack((positions >= side, positions < area - side, positions % side > 0,
                       positions % side < side - 1), axis=1)
    neighbors = jnp.clip(neighbors, 0, area - 1)
    valid = valid & inside[:, None] & inside[neighbors]
    opened = inside & (memory.terrain != 2) & (memory.terrain != 5)
    producer = (memory.terrain == 3) | (memory.terrain == 4)
    sources = jnp.argsort(-jnp.where((owner == 1) & (army > 1) & inside, army, -1), stable=True)[:8]
    usable = (owner[sources] == 1) & (army[sources] > 1) & inside[sources]
    blank = Plan(jnp.full(area + 1, -1, jnp.int32), jnp.int32(0), jnp.int32(0), jnp.int32(0))
    passing = jnp.int32(8 * area)

    def distances(target):
        # a fixed-capacity BFS queue visits every reachable cell once, including winding corridors.
        distance = jnp.full(area, -1, jnp.int32).at[target].set(0)
        queue = jnp.zeros(area, jnp.int32).at[0].set(target)

        def visit(state):
            head, tail, queue, distance = state
            position = queue[head]

            def edge(direction, state):
                tail, queue, distance = state
                cell = neighbors[position, direction]
                take = valid[position, direction] & opened[cell] & (distance[cell] < 0)
                queue = jax.lax.cond(take, lambda data: data.at[tail].set(cell), lambda data: data, queue)
                distance = distance.at[cell].set(jnp.where(take, distance[position] + 1, distance[cell]))
                return tail + take.astype(jnp.int32), queue, distance

            tail, queue, distance = jax.lax.fori_loop(0, 4, edge, (tail, queue, distance))
            return head + 1, tail, queue, distance

        return jax.lax.while_loop(lambda state: state[0] < state[1], visit,
                                  (jnp.int32(0), jnp.int32(1), queue, distance))[3]

    def safe(source, target):
        cells = neighbors[source]
        threat = valid[source] & (owner[cells] == 2) & (army[cells] > 2)
        captured = (cells == target) & (army[source] - 1 > army[cells])
        return (source != general) | ~jnp.any(threat & ~captured)

    def extend(plan, cell):
        owned = owner[cell] == 1
        defense = jnp.where(visible[cell], army[cell], memory.army[cell])
        known_owner = jnp.where(visible[cell], owner[cell], memory.owner[cell])
        defense += jnp.where(~owned & (known_owner > 0) & producer[cell], plan.length // 2, 0)
        allowed = (plan.army > 1) & (owned | (plan.army - 1 > defense))
        troops = plan.army - 1 + jnp.where(owned, army[cell], -defense)
        # quarter-units represent the original 1, 2, 25 and 0.25 gains exactly.
        gain = plan.gain + jnp.where(owned, 0, 4 + 4 * (known_owner == 2) + 100 * producer[cell] + ~memory.seen[cell])
        return Plan(plan.path.at[plan.length].set(cell), plan.length + 1, troops, gain), allowed

    def better(candidate, best, ties=False):
        left = candidate.gain * jnp.maximum(best.length - 1, 1)
        right = best.gain * jnp.maximum(candidate.length - 1, 1)
        return (candidate.length >= 2) & ((left > right) | (ties & (left == right) & (candidate.gain > best.gain)))

    def select(condition, first, second):
        return jax.tree.map(lambda a, b: jnp.where(condition, a, b), first, second)

    def toward(target, winning):
        distance = distances(target)

        def follow(source, active):
            plan = Plan(blank.path.at[0].set(source), jnp.int32(1), army[source], jnp.int32(0))
            active = active & (distance[source] > 0)

            def advance(state):
                position, plan = state[:2]

                def edge(direction, best):
                    chosen, next_plan, largest = best
                    cell = neighbors[position, direction]
                    candidate, allowed = extend(plan, cell)
                    allowed = allowed & valid[position, direction] & (distance[cell] == distance[position] - 1)
                    allowed = allowed & ((position != source) | safe(source, cell)) & (candidate.army > largest)
                    return (jnp.where(allowed, cell, chosen), select(allowed, candidate, next_plan),
                            jnp.where(allowed, candidate.army, largest))

                chosen, next_plan = jax.lax.fori_loop(0, 4, edge, (jnp.int32(-1), plan, jnp.int32(-1)))[:2]
                return jnp.maximum(chosen, 0), next_plan, chosen >= 0

            position, plan, active = jax.lax.while_loop(
                lambda state: state[2] & (state[0] != target), advance, (source, plan, active))
            return select(active & (position == target) & (plan.length >= 2), plan, blank)

        plans = jax.vmap(follow)(sources, usable)

        def pick(index, best):
            plan = jax.tree.map(lambda value: value[index], plans)
            take = (plan.length >= 2) & ((best.length == 0) | jnp.where(winning, plan.length < best.length, better(plan, best)))
            return select(take, plan, best)

        return jax.lax.fori_loop(0, sources.size, pick, blank)

    def action(plan):
        source, target = plan.path[0], plan.path[1]
        # clipped indices are safe to read, but an off-board neighbor cannot define the move's direction.
        direction = jnp.argmax(valid[source] & (neighbors[source] == target))
        return jnp.where(plan.length >= 2, direction * area + source, passing).astype(jnp.int32)

    def expand(source, active):
        width, depth = 24, 12
        paths = jnp.full((width, depth + 1), -1, jnp.int32).at[0, 0].set(source)
        troops = jnp.zeros(width, jnp.int32).at[0].set(army[source])
        gains = jnp.zeros(width, jnp.int32)
        live = jnp.zeros(width, jnp.bool_).at[0].set(active)

        def layer(level, state):
            paths, troops, gains, live, best = state
            ends = jnp.maximum(paths[:, level], 0)
            cells = neighbors[ends].reshape(-1)
            candidates = jnp.repeat(paths, 4, axis=0)
            allowed = jnp.repeat(live, 4) & valid[ends].reshape(-1) & opened[cells]
            allowed = allowed & ~jnp.any(candidates == cells[:, None], axis=1)
            allowed = allowed & ((level > 0) | jax.vmap(lambda cell: safe(source, cell))(cells))
            old_troops, old_gains = jnp.repeat(troops, 4), jnp.repeat(gains, 4)
            owned = owner[cells] == 1
            known_owner = jnp.where(visible[cells], owner[cells], memory.owner[cells])
            defense = jnp.where(visible[cells], army[cells], memory.army[cells])
            defense += jnp.where(~owned & (known_owner > 0) & producer[cells], (level + 1) // 2, 0)
            allowed = allowed & (old_troops > 1) & (owned | (old_troops - 1 > defense))
            next_troops = old_troops - 1 + jnp.where(owned, army[cells], -defense)
            next_gains = old_gains + jnp.where(owned, 0, 4 + 4 * (known_owner == 2) + 100 * producer[cells] + ~memory.seen[cells])
            candidates = candidates.at[:, level + 1].set(cells)
            # all candidates here have equal length; argmax retains the first path on a tie.
            winner = jnp.argmax(jnp.where(allowed, next_gains, -1))
            candidate = Plan(blank.path.at[:depth + 1].set(candidates[winner]), level + 2,
                             next_troops[winner], next_gains[winner])
            best = select(allowed[winner] & better(candidate, best, True), candidate, best)
            # gain + 0.02 * army, scaled by 100, preserves the original stable beam ordering.
            rank = jnp.where(allowed, 25 * next_gains + 2 * next_troops, -2147483647)
            keep = jnp.argsort(-rank, stable=True)[:width]
            return candidates[keep], next_troops[keep], next_gains[keep], allowed[keep], best

        return jax.lax.fori_loop(0, depth, layer, (paths, troops, gains, live, blank))[4]

    def decide(memory):
        home = distances(general)
        dangers = (owner == 2) & (home > 0) & (home <= 8) & (army - home > army[general] + home // 2)
        threat = jnp.argmin(jnp.where(dangers, home, area + 1))
        threatened = jnp.any(dangers)

        def defend():
            target = toward(threat, True)
            intercept = (target.length >= 2) & (target.length - 1 <= home[threat])
            cells = neighbors[sources]
            allowed = usable[:, None] & (sources[:, None] != general) & valid[sources]
            allowed &= (home[sources, None] > 0) & (home[sources, None] <= home[threat])
            allowed &= (home[cells] == home[sources, None] - 1) & (owner[cells] == 1)
            rates = jnp.where(allowed, (army[sources, None] - 1) / jnp.maximum(home[sources, None], 1), 0)
            index = jnp.argmax(rates)
            reinforce = jnp.where(jnp.any(allowed), (index % 4) * area + sources[index // 4], passing)
            return jnp.where(intercept, action(target), reinforce).astype(jnp.int32)

        rescue = jax.lax.cond(threatened, defend, lambda: passing)

        def ordinary(memory):
            enemies = (memory.terrain == 4) & (memory.owner == 2) & inside

            def attack(state):
                remaining = state[0]
                target = jnp.argmax(remaining)
                return remaining.at[target].set(False), toward(target, True)

            attack_plan = jax.lax.while_loop(
                lambda state: jnp.any(state[0]) & (state[1].length == 0), attack, (enemies, blank))[1]
            attack_action = action(attack_plan)

            def continue_route(memory):
                source, target = jnp.maximum(memory.route[0], 0), jnp.maximum(memory.route[1], 0)
                follows = ((memory.length >= 2) & (owner[source] == 1) & (army[source] >= 2) & opened[target]
                           & visible[target] & ((owner[target] == 1) | (army[source] - 1 > army[target])) & safe(source, target))

                def plan_route():
                    cities = (memory.terrain == 3) & (owner != 1) & inside

                    def visit_city(state):
                        remaining, best = state
                        target = jnp.argmax(remaining)
                        candidate = toward(target, False)
                        return remaining.at[target].set(False), select(better(candidate, best), candidate, best)

                    best = jax.lax.while_loop(lambda state: jnp.any(state[0]), visit_city, (cities, blank))[1]
                    plans = jax.vmap(expand)(sources, usable)

                    def compare(index, best):
                        candidate = jax.tree.map(lambda value: value[index], plans)
                        return select(better(candidate, best), candidate, best)

                    best = jax.lax.fori_loop(0, sources.size, compare, best)
                    start = jnp.maximum(best.path[0], 0)
                    wait = (army[start] < 4) & producer[start] & ~threatened & (tick % 50 < 44)
                    return select(wait, blank, best)

                plan = jax.lax.cond(follows, lambda: Plan(memory.route, memory.length, jnp.int32(0), jnp.int32(0)), plan_route)
                route = jnp.concatenate((plan.path[1:], jnp.array([-1], jnp.int32)))
                return memory._replace(route=route, length=jnp.maximum(plan.length - 1, 0)), action(plan)

            return jax.lax.cond(attack_action != passing,
                                lambda: (memory._replace(length=jnp.int32(0)), attack_action), lambda: continue_route(memory))

        return jax.lax.cond(rescue != passing, lambda: (memory._replace(length=jnp.int32(0)), rescue),
                            lambda: ordinary(memory))

    return jax.lax.cond(general >= 0, decide, lambda data: (data, passing), memory)


decisions = jax.jit(jax.vmap(choose))


class Baseline:
    def __init__(self, count, side, backend):
        self.backend = backend
        with jax.default_device(self.backend):
            self.memory = empty(count, side * side)
        self.choice = None

    def act(self, view, reset):
        terrain, owner, army, stats, inside = view
        with jax.default_device(self.backend):
            self.memory, self.choice = decisions(self.memory, terrain.astype(jnp.int32), owner.astype(jnp.int32),
                                                 army.astype(jnp.int32), inside, stats[:, 0].astype(jnp.int32), reset)
        return self.choice

    def submitted(self, action):
        # a different executed action invalidates the route, but not the observed terrain.
        with jax.default_device(self.backend):
            self.memory = self.memory._replace(length=jnp.where(action == self.choice, self.memory.length, 0))
