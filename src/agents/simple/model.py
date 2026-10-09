"""GenFormer learns spatial and historical relations, then predicts primitive actions and game outcomes."""

import equinox as eqx
import jax
import jax.numpy as jnp
from features import adjacent, coverage


def number(value):
    return jnp.stack((value / 1024, jnp.sign(value) * jnp.log1p(jnp.abs(value)) / 8), axis=-1)


def categories(value, count):
    return jax.nn.one_hot(value.astype(jnp.int32), count)


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, incoming, outgoing, key, zero=False, orthogonal=False):
        first, second = jax.random.split(key)
        bound = incoming**-0.5
        self.weight = jax.random.uniform(first, (outgoing, incoming), minval=-bound, maxval=bound)
        self.bias = jax.random.uniform(second, (outgoing,), minval=-bound, maxval=bound)
        if zero:
            self.weight = jnp.zeros_like(self.weight)
            self.bias = jnp.zeros_like(self.bias)
        if orthogonal:
            self.weight = jax.nn.initializers.orthogonal(0.01)(first, (incoming, outgoing)).T
            self.bias = jnp.zeros_like(self.bias)

    def __call__(self, value, dtype=jnp.float32):
        return (value.astype(dtype) @ self.weight.astype(dtype).T + self.bias.astype(dtype)).astype(dtype)


class Norm(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, width):
        self.weight, self.bias = jnp.ones(width), jnp.zeros(width)

    def __call__(self, value):
        value = value.astype(jnp.float32)
        centered = value - value.mean(-1, keepdims=True)
        return centered * jax.lax.rsqrt(jnp.mean(centered**2, -1, keepdims=True) + 1e-5) * self.weight + self.bias


class Conv(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, width, key):
        first, second = jax.random.split(key)
        bound = (9 * width)**-0.5
        self.weight = jax.random.uniform(first, (width, width, 3, 3), minval=-bound, maxval=bound)
        self.bias = jax.random.uniform(second, (width,), minval=-bound, maxval=bound)

    def __call__(self, value, dtype):
        result = jax.lax.conv_general_dilated(value.astype(dtype), self.weight.astype(dtype),
                                              (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW"))
        return result + self.bias.astype(dtype)[None, :, None, None]


def observations(board, scalars, inside):
    batch, times, _, rows, cols = board.shape
    terrain, owner, army = (board[:, :, index] for index in range(3))
    visible = (terrain != 0) & (terrain != 5) & inside[:, None]
    cells = jnp.arange(rows * cols).reshape(1, 1, rows, cols)
    action = scalars[:, :, 5].astype(jnp.int32)
    source = cells == (action % (rows * cols))[:, :, None, None]
    kind = action // (rows * cols)
    move = (action >= 0) & (action < 8 * rows * cols)
    submitted = (kind[:, :, None] == jnp.arange(8)) & move[:, :, None]
    commands = submitted[:, :, None, None, :] * source[..., None]
    fields = jnp.concatenate((categories(terrain, 6), categories(owner, 3), number(army),
                              jnp.broadcast_to(inside[:, None, :, :, None], (batch, times, rows, cols, 1)),
                              visible[..., None], coverage((owner == 1) & visible)[..., None] / 9,
                              coverage((owner == 2) & visible)[..., None] / 9, commands), axis=-1)
    fields = fields * inside[:, None, :, :, None]
    age = jnp.maximum(scalars[:, :1, 0] - scalars[:, :, 0], 0)
    height = jnp.sum(jnp.any(inside, 2), 1) / 32
    width = jnp.sum(jnp.any(inside, 1), 1) / 32
    public = jnp.concatenate((number(scalars[:, :, :5]).reshape(batch, times, 10),
                              (scalars[:, :, :1] % 2) / 2, (scalars[:, :, :1] % 50) / 50,
                              number(age), (action >= 0)[..., None], (action == 8 * cells.size)[..., None],
                              jnp.broadcast_to(height[:, None, None], (batch, times, 1)),
                              jnp.broadcast_to(width[:, None, None], (batch, times, 1))), axis=-1)
    return fields.reshape(batch, times, -1, 23), public


def memories(memory, tick):
    terrain, owner, army, stamp, exposure, enemy_army, enemy_stamp = (memory[:, i] for i in range(7))
    seen, exposed, enemy_seen = stamp >= 0, exposure >= 0, enemy_stamp >= 0
    age = jnp.maximum(jnp.where(seen, tick[:, None, None] - stamp, 0), 0)
    exposure_age = jnp.maximum(jnp.where(exposed, tick[:, None, None] - exposure, 0), 0)
    enemy_age = jnp.maximum(jnp.where(enemy_seen, tick[:, None, None] - enemy_stamp, 0), 0)
    fields = jnp.concatenate((categories(terrain, 6), categories(owner, 3), number(army), number(age),
                              seen[..., None], number(exposure_age), exposed[..., None],
                              number(enemy_army), number(enemy_age), enemy_seen[..., None]), axis=-1)
    return fields.reshape(len(tick), -1, 22), (seen | exposed).reshape(len(tick), -1)


class History(eqx.Module):
    cell: Linear
    public: tuple
    memory: Linear
    norm: Norm
    qkv: Linear
    output: Linear
    heads: int = eqx.field(static=True)

    def __init__(self, width, heads, key):
        keys = jax.random.split(key, 6)
        self.cell = Linear(23, width, keys[0])
        self.public = (Linear(18, width, keys[1]), Linear(width, width, keys[2]))
        self.memory, self.norm = Linear(22, width, keys[3]), Norm(width)
        self.qkv = Linear(width, 3 * width, keys[4], zero=True)
        self.qkv = eqx.tree_at(lambda layer: layer.weight, self.qkv,
                               jax.nn.initializers.glorot_uniform()(keys[4], (width, 3 * width)).T)
        self.output = Linear(width, width, keys[5])
        self.output = eqx.tree_at(lambda layer: layer.bias, self.output, jnp.zeros(width))
        self.heads = heads

    def __call__(self, board, scalars, valid, memory, inside, dtype):
        fields, public = observations(board, scalars, inside)
        record, known = memories(memory, scalars[:, 0, 0])
        summary = self.public[1](jax.nn.silu(self.public[0](public, dtype)), dtype)
        encoded = self.cell(fields, dtype) + summary[:, :, None]
        current = encoded[:, 0].astype(jnp.float32)
        sequence = jnp.concatenate((encoded, self.memory(record, dtype)[:, None]), axis=1).transpose(0, 2, 1, 3)
        available = jnp.concatenate((jnp.broadcast_to(valid[:, :, None], encoded.shape[:3]), known[:, None]), axis=1)
        sequence = self.norm(sequence).reshape(-1, sequence.shape[2], sequence.shape[3])
        width, heads = current.shape[-1], self.heads
        weight, bias = self.qkv.weight.astype(dtype), self.qkv.bias.astype(dtype)
        query = (sequence[:, 0].astype(dtype) @ weight[:width].T + bias[:width]).reshape(-1, heads, width // heads)
        projected = sequence.astype(dtype) @ weight[width:].T + bias[width:]
        projected = projected.reshape(sequence.shape[0], sequence.shape[1], 2, heads, width // heads)
        key, value = projected[:, :, 0], projected[:, :, 1]
        scores = (query[:, None].astype(jnp.float32) * key.astype(jnp.float32)).sum(-1) * (width // heads)**-0.5
        available = available.transpose(0, 2, 1).reshape(sequence.shape[:2])
        weights = jax.nn.softmax(jnp.where(available[..., None], scores, -jnp.inf), axis=1)
        correction = (weights[..., None] * value.astype(jnp.float32)).sum(1).reshape(-1, width)
        correction = self.output(correction, dtype).reshape(current.shape)
        return (current + correction) * inside.reshape(len(board), -1, 1), summary[:, 0]


class Bias(eqx.Module):
    geometry: tuple
    coefficients: Linear
    source: Linear
    target: Linear
    heads: int = eqx.field(static=True)
    templates: int = eqx.field(static=True)

    def __init__(self, width, heads, templates, key):
        a, b, c, d, e = jax.random.split(key, 5)
        self.geometry = (Linear(2, 16, a), Linear(16, templates, b))
        self.coefficients = Linear(width, heads * (templates + 3), c, zero=True)
        self.source = Linear(width, heads * (templates + 3), d, zero=True)
        self.target = Linear(width, heads * (templates + 3), e, zero=True)
        self.heads, self.templates = heads, templates

    def __call__(self, cells, context, relations, offsets, pairs, sight, dtype):
        weights = self.coefficients(context, dtype).reshape(-1, self.heads, self.templates + 3)
        shape = (*cells.shape[:2], self.heads, self.templates + 3)
        source = self.source(cells, dtype).reshape(shape)
        target = self.target(cells, dtype).reshape(shape)
        geometry = self.geometry[1](jax.nn.silu(self.geometry[0](offsets, dtype)), dtype)
        positional = weights[:, :, :self.templates] @ geometry.T
        bias = positional[:, :, pairs].astype(jnp.float32)
        # contract templates directly; do not materialize a batch of per-pair coefficient vectors.
        pair_geometry = geometry[pairs]
        bias = bias + jnp.einsum("bihr,ijr->bhij", source[..., :self.templates], pair_geometry).astype(jnp.float32)
        bias = bias + jnp.einsum("bjhr,ijr->bhij", target[..., :self.templates], pair_geometry).astype(jnp.float32)
        for index, relation in enumerate((*relations, sight[None])):
            slot = self.templates + index
            coefficient = (weights[:, :, slot, None, None].astype(jnp.float32)
                           + source[..., slot].transpose(0, 2, 1)[..., None].astype(jnp.float32)
                           + target[..., slot].transpose(0, 2, 1)[:, :, None].astype(jnp.float32))
            bias = bias + coefficient * relation[:, None]
        return bias


class Block(eqx.Module):
    norm: Norm
    pool: Linear
    qkv: Linear
    output: Linear
    bias: Bias
    local: Conv
    project: Conv
    feed: tuple
    heads: int = eqx.field(static=True)

    def __init__(self, config, key):
        keys = jax.random.split(key, 8)
        width = config.width
        self.heads, self.norm = config.heads, Norm(width)
        self.qkv, self.output = Linear(width, 3 * width, keys[0]), Linear(width, width, keys[1])
        self.bias = Bias(width, config.heads, config.templates, keys[2])
        self.local, self.project = Conv(width, keys[3]), Conv(width, keys[4])
        self.feed = (Norm(width), Linear(width, 4 * width, keys[5]), Linear(4 * width, width, keys[6]))
        self.pool = Linear(width, 1, keys[7], orthogonal=True)

    def __call__(self, cells, inside, valid, relations, offsets, pairs, sight, dtype):
        batch, count, width = cells.shape
        area = inside.shape[-2] * inside.shape[-1]
        normalized = self.norm(cells)
        projected = self.qkv(normalized, dtype).reshape(batch, count, 3, self.heads, width // self.heads)
        query, key, value = (projected[:, :, index] for index in range(3))
        # g weights board cells and four public-history tokens by learned scores; padding receives zero weight.
        # this lossy summary conditions the bias; individual cells still enter both the bias and content attention.
        scores = self.pool(normalized, dtype)[..., 0].astype(jnp.float32)
        weights = jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), axis=1)
        context = (normalized * weights[..., None]).sum(1)
        bias = self.bias(normalized[:, :area], context, relations, offsets, pairs, sight, dtype)
        bias = jnp.pad(bias, ((0, 0), (0, 0), (0, count - area), (0, count - area))).astype(dtype)
        bias = jnp.where(valid[:, None, None, :], bias, -jnp.inf)
        attended = jax.nn.dot_product_attention(query, key, value, bias=bias)
        cells = (cells + self.output(attended.reshape(batch, count, width), dtype)) * valid[..., None]
        grid = cells[:, :area].transpose(0, 2, 1).reshape(batch, width, *inside.shape[-2:])
        local = self.project(jax.nn.silu(self.local(grid, dtype)) * inside[:, None], dtype) * inside[:, None]
        cells = cells + jnp.pad(local.reshape(batch, width, area).transpose(0, 2, 1), ((0, 0), (0, count - area), (0, 0)))
        feed = self.feed[2](jax.nn.silu(self.feed[1](self.feed[0](cells), dtype)), dtype)
        return (cells + feed) * valid[..., None]


class GenFormer(eqx.Module):
    history: History
    scores: tuple
    identity: jax.Array
    blocks: tuple
    norm: Norm
    context: Linear
    action: jax.Array
    moves: tuple
    source: Linear
    target: Linear
    wait: Linear
    critic: Linear
    side: int = eqx.field(static=True)
    bins: int = eqx.field(static=True)

    def __init__(self, config, key):
        keys = jax.random.split(key, 12 + config.depth)
        width = config.width
        self.side, self.bins = config.side, config.bins
        self.history = History(width, config.heads, keys[0])
        self.scores = (Linear(5 * config.temporal, width, keys[1]), Linear(width, width, keys[2]))
        self.identity = jax.random.normal(keys[3], (4, width))
        self.blocks = tuple(Block(config, key) for key in keys[12:])
        self.norm, self.context = Norm(width), Linear(3 * width, width, keys[4])
        self.action = jax.random.normal(keys[5], (8, width))
        self.moves = (Linear(3 * width, width, keys[6]), Linear(width, 1, keys[7], orthogonal=True))
        self.wait, self.critic = Linear(width, 1, keys[8], orthogonal=True), Linear(width, config.bins, keys[9])
        self.source = Linear(width, 2 * width, keys[10], orthogonal=True)
        self.target = Linear(width, 2 * width, keys[11])

    def interaction(self, cells, dtype):
        """Score adjacent source-target pairs with separate learned projections for full and half moves."""
        batch, area, width = cells.shape
        source = self.source(cells, dtype).reshape(batch, area, 2, width).transpose(0, 2, 1, 3)
        grid = self.target(cells, dtype).transpose(0, 2, 1).reshape(batch, 2 * width, self.side, self.side)
        target = adjacent(grid).transpose(0, 2, 3, 4, 1).reshape(batch, 4, area, 2, width).transpose(0, 3, 1, 2, 4)
        product = source[:, :, None].astype(jnp.float32) * target.astype(jnp.float32)
        return (product.sum(-1) * width**-0.5).reshape(batch, 8 * area)

    def __call__(self, board, scalars, valid, memory, inside, legal, timeline, *, bf16=False):
        dtype = jnp.bfloat16 if bf16 else jnp.float32
        board, scalars, memory = (value.astype(jnp.float32) for value in (board, scalars, memory))
        cells, public = self.history(board, scalars, valid, memory, inside, dtype)
        observed = timeline[:, :, 0] >= 0
        age = jnp.maximum(scalars[:, :1, 0] - timeline[:, :, 0], 0)
        numbers = number(timeline[:, :, 1:].transpose(0, 2, 1))
        dates = jnp.broadcast_to(number(age)[:, None], (*numbers.shape[:3], 2))
        flags = jnp.broadcast_to(observed[:, None, :, None], (*numbers.shape[:3], 1))
        series = jnp.concatenate((numbers, dates, flags), axis=-1) * flags
        temporal = self.scores[1](jax.nn.silu(self.scores[0](series.reshape(len(board), 4, -1), dtype)), dtype) + self.identity[None]
        cells = jnp.concatenate((cells, temporal), axis=1)
        valid = jnp.pad(inside.reshape(len(board), -1), ((0, 0), (0, 4)), constant_values=True)
        visible = (board[:, 0, 0] != 0) & (board[:, 0, 0] != 5)
        terrain = jnp.where(visible, board[:, 0, 0], memory[:, 0]).reshape(len(board), -1)
        known = (visible | (memory[:, 3] >= 0)).reshape(len(board), -1)
        possible = inside.reshape(len(board), -1) & ~(known & (terrain == 2))
        row, col = jnp.meshgrid(jnp.arange(self.side), jnp.arange(self.side), indexing="ij")
        positions = jnp.stack((row.ravel(), col.ravel()), axis=-1)
        relative = positions[None] - positions[:, None]
        pairs = (relative[..., 0] + self.side - 1) * (2 * self.side - 1) + relative[..., 1] + self.side - 1
        delta = jnp.arange(1 - self.side, self.side)
        offsets = jnp.stack(jnp.meshgrid(delta, delta, indexing="ij"), axis=-1).reshape(-1, 2) / 32
        neighbor, sight = jnp.abs(relative).sum(-1) == 1, jnp.abs(relative).max(-1) <= 1
        relations = tuple(neighbor[None] & region[:, :, None] & region[:, None, :] for region in (possible & known, possible))
        for block in self.blocks:
            cells = block(cells, inside, valid, relations, offsets, pairs, sight, dtype)
        cells = self.norm(cells) * valid[..., None]
        mean = cells.sum(1) / jnp.maximum(valid.sum(1, keepdims=True), 1)
        maximum = jnp.where(valid[..., None], cells, -jnp.inf).max(1)
        # the output heads use a separate learned summary of mean/max tokens and current public statistics.
        context = jax.nn.silu(self.context(jnp.concatenate((mean, maximum, public), axis=-1), dtype))
        cells = cells[:, :-4]
        source_weight, target_weight, command_weight = jnp.split(self.moves[0].weight.astype(dtype), 3, axis=1)
        source = (cells.astype(dtype) @ source_weight.T)[:, None]
        grid = (cells.astype(dtype) @ target_weight.T).transpose(0, 2, 1).reshape(len(board), -1, self.side, self.side)
        target = jnp.tile(adjacent(grid).transpose(0, 2, 3, 4, 1).reshape(len(board), 4, -1, cells.shape[-1]), (1, 2, 1, 1))
        command = ((self.action[None] + context[:, None]).astype(dtype) @ command_weight.T + self.moves[0].bias.astype(dtype))[:, :, None]
        combined = source.astype(jnp.float32) + target.astype(jnp.float32) + command.astype(jnp.float32)
        moves = self.moves[1](jax.nn.silu(combined.astype(dtype)), dtype).reshape(len(board), -1)
        moves = moves.astype(jnp.float32) + self.interaction(cells, dtype)
        logits = jnp.concatenate((moves, self.wait(context, dtype)), axis=1).astype(jnp.float32)
        logits = jnp.where(legal, logits, -1e9)
        value_logits = self.critic(context, dtype).astype(jnp.float32)
        centers = jnp.linspace(-1, 1, self.bins)
        value = (jax.nn.softmax(value_logits, axis=-1) * centers).sum(-1)
        return logits, value, value_logits
