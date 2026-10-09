"""Training settings separate the paper-sized network from the small execution check."""

from dataclasses import dataclass, replace


@dataclass
class Config:
    side: int = 24
    patch: int = 3
    width: int = 448
    depth: int = 7
    heads: int = 8
    expansion: int = 3
    temporal: int = 512
    bins: int = 128
    sigma: float = 0.04
    environments: int = 512
    # each map size contributes equally to a pool, which is refreshed between training iterations.
    minimum: int = 17
    maximum: int = 23
    pool: int = 4096
    refresh: int = 50
    steps: int = 512
    minibatch: int = 1024
    microbatch: int = 128  # bounds device memory without changing the optimizer's minibatch.
    epochs: int = 1
    updates: int = 100000
    horizon: int = 2048
    gamma: float = 1.0
    gae: float = 0.9
    clip: float = 0.2
    value: float = 0.5
    grad: float = 0.267
    fraction: float = 0.25
    ema: float = 0.999
    seed: int = 44
    bf16: bool = True
    save: int = 50
    evaluate: int = 50
    games: int = 512
    # the paper starts with maximum distance four. later ranges follow the released curriculum.
    curriculum: tuple = ((3, 4), (4, 8), (6, 13), (11, 17), (17, 28))
    threshold: float = 0.6

    def validate(self):
        if self.patch < 1 or not 18 <= self.side <= 40 or self.side % self.patch:
            raise ValueError("Input size must be 18..40 and divisible by patch size")
        if self.width < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("Embedding width must be divisible by the head count")
        for name in ("depth", "expansion", "temporal", "environments", "pool", "refresh", "steps", "minibatch", "microbatch", "epochs", "updates", "save", "games"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.temporal != 512 or self.bins < 2 or self.sigma <= 0:
            raise ValueError("Invalid temporal or value-head settings")
        if not 17 <= self.minimum <= self.maximum <= self.side:
            raise ValueError("Map sizes must satisfy 17 <= minimum <= maximum <= input size")
        if not 0 < self.fraction <= 1 or not 0 <= self.ema < 1 or not 1 <= self.horizon <= 50000:
            raise ValueError("Invalid sampling, EMA or episode limit")
        for lower, upper in self.curriculum:
            if lower < 3 or upper < lower:
                raise ValueError("Curriculum distances must satisfy 3 <= minimum <= maximum")
        if not self.curriculum:
            raise ValueError("A spawn-distance stage is required")
        return self


def profile(name):
    config = Config()
    if name == "check":
        config = replace(config, width=48, depth=2, heads=4, environments=2, steps=16,
                         minibatch=16, updates=2, horizon=64, bf16=False, save=1,
                         evaluate=0, games=4, minimum=18, maximum=18, pool=4)
    elif name != "paper":
        raise ValueError(f"Unknown profile: {name}")
    return config.validate()
