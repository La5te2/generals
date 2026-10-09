"""Network capacity, self-play optimization and map curriculum for GenFormer."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    side: int = 24
    minimum: int = 18
    maximum: int = 23
    distance: tuple[int, int | None] = (17, None)
    width: int = 128
    heads: int = 4
    templates: int = 8
    depth: int = 6
    history: int = 8
    temporal: int = 512
    environments: int = 512
    pool: int = 4096
    horizon: int = 2048
    steps: int = 512
    minibatch: int = 1024
    microbatch: int = 32  # forward-pass limit; gradients accumulate to minibatch before each optimizer step.
    epochs: int = 1
    samples: int = 100_000 * 512 * 512  # run budget, not a stopping criterion for playing strength.
    # passing a win-rate gate advances new games to the next spawn-distance range.
    # ongoing games keep their maps; the final stage uses distance above.
    curriculum: tuple = ((3, 4), (4, 8), (6, 13), (11, 17))
    threshold: float = 0.6
    refresh: int = 50
    save: int = 10
    evaluate: int = 50
    games: int = 512
    evaluation_seed: int = 100044
    learning_rate: float = 1e-4
    final_rate: float = 5e-6
    gamma: float = 1.0
    gae: float = 0.9
    clip: float = 0.2
    entropy: float = 0.05
    final_entropy: float = 0.0
    value: float = 0.5
    gradient: float = 0.267
    fraction: float = 0.25
    bins: int = 128
    sigma: float = 0.04
    ema: float = 0.999
    kl: float | None = None
    augment: bool = True
    bf16: bool = True
    seed: int = 44

    def validate(self):
        if not 4 <= self.minimum <= self.maximum <= self.side <= 40:
            raise ValueError("Require 4 <= minimum <= maximum <= side <= 40")
        if (len(self.distance) != 2 or self.distance[0] < 3 or
                (self.distance[1] is not None and self.distance[1] < self.distance[0])):
            raise ValueError("General distance requires a minimum >= 3 and an optional upper bound")
        for name in ("width", "depth", "heads", "templates", "temporal", "environments", "pool", "horizon",
                     "steps", "minibatch", "microbatch", "epochs", "samples", "refresh", "save", "evaluate", "games"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.history < 0:
            raise ValueError("History length must be nonnegative")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.pool < (self.maximum - self.minimum + 1)**2:
            raise ValueError("pool must hold at least one map of each requested shape")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae <= 1 or not 0 < self.clip < 1:
            raise ValueError("Invalid PPO discount, GAE or clip")
        if self.learning_rate <= 0 or self.gradient <= 0 or self.entropy < 0 or self.value < 0:
            raise ValueError("Invalid optimizer setting")
        if self.final_rate <= 0 or self.final_entropy < 0 or self.bins < 2 or self.sigma <= 0:
            raise ValueError("Invalid learning schedule or value distribution")
        if not 0 <= self.ema < 1 or (self.kl is not None and self.kl <= 0):
            raise ValueError("Invalid EMA or KL limit")
        if not 0 < self.fraction <= 1:
            raise ValueError("Advantage selection fraction must be in (0, 1]")
        if self.games % 2:
            raise ValueError("Evaluation games must form red/blue pairs")
        if not 0 < self.threshold <= 1:
            raise ValueError("Curriculum threshold must be in (0, 1]")
        for minimum, maximum in self.curriculum:
            if minimum < 3 or (maximum is not None and maximum < minimum):
                raise ValueError("Invalid curriculum spawn distances")
        return self


def maps(config, stage):
    """Return the configuration for new games at one curriculum stage."""
    if not 0 <= stage <= len(config.curriculum):
        raise ValueError("Curriculum stage is outside the configured ranges")
    distance = config.curriculum[stage] if stage < len(config.curriculum) else config.distance
    return replace(config, distance=tuple(distance))
