# Agent experiments

Each numbered folder is a self-contained training source snapshot. Run its
commands from that folder and install its own requirements. Initial weights and
third-party licenses are included. Datasets, generated runs and caches stay local.

| Version | Starting point for readers |
| --- | --- |
| [1012](1012/README.md) | Replay-supervised GNN/attention policy followed by optional self-play PPO |
| [1013](1013/README.md) | PPO, potential-based shaping and opponent modeling |
| [1014](1014/README.md) | Q-critic, auxiliary predictions and opponent-pool promotion |
| [1015](1015/README.md) | Spatial memory, fitted potential and gathering supervision |

The per-version README documents that version's implementation and commands.
These snapshots retain their historical evaluation settings and checkpoint
formats. Use weights with their matching version. Source preparation and training
remain independent of the interface and server builds.
