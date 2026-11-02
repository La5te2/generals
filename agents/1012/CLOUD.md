# Cloud Training

Upload this entire folder, not the repository. It includes the network, input
encoder, supervised/PPO entry points, historical simulator and license.
Download the two datasets using the sources and filenames in [README.md](README.md#sources-and-local-placement)
before preparation. Raw datasets and pilot samples are external to this Git snapshot. Python dependencies are
installed on the destination; Windows `.deps`, old checkpoints and caches are
intentionally absent. This is not an offline Python environment bundle.

## First Run

Use a Linux NVIDIA GPU machine with Python 3.12 or 3.13, Node.js, Git and network
access for pip. Run these commands inside the uploaded directory:

```sh
bash setup.sh
bash prepare.sh
python -B smoke.py --data data/processed/train
bash train.sh
```

`setup.sh` installs into the selected Python environment (`PYTHON=python3` can
override it). It defaults to the pip CUDA 12 runtime and verifies that JAX sees
a GPU. It neither installs drivers nor silently falls back to CPU. On a machine
with a suitable driver, `bash setup.sh cuda13` selects CUDA 13 instead. Driver
requirements and runtime troubleshooting are in the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
An existing `LD_LIBRARY_PATH` pointing at incompatible CUDA libraries can override
pip's libraries; resolve that environment conflict if initialization fails.

`smoke.py` checks that the full network can fit four recorded moves. It writes to
`runs/smoke/`, which is NOT used to initialize the subsequent training run.

`prepare.sh` converts up to 1,000 NEW accepted games from each source on CPU,
high-rank data first, into `data/processed/`. Both sources use the same deduplication
and whole-map train/validation/test split. Rerunning it adds another portion;
`bash prepare.sh 0` converts all remaining supported games. Conversion is resumable
through its manifest but may need to rescan earlier archive entries. Do not modify
this prepared dataset while a training run is using it. Full conversion requires
substantially more disk than the compressed archives; first inspect the size and
conversion rate of the bounded run. Neither full-dataset time nor size is measured.

`train.sh` starts one supervised epoch from new random weights, batch size 8,
learning rate 1e-4, and player rank >=70. Thus both sources contribute eligible
players, not every recorded player's actions. It does NOT start PPO automatically.
This is a conservative first-run configuration, not a GPU-benchmarked speed optimum.
The actual split counts and device are printed at startup. New board shapes incur
JAX compilation costs. The first update can take much longer than later ones.

Training writes metrics to `runs/supervised/metrics.jsonl`, logs progress every
25 updates and atomically saves `checkpoint.zip` every 100 updates and before
validation. `policy.zip` is selected after validation by the lowest held-out
action NLL. No win-rate claim follows from that metric. The held-out test split
is not used for model selection.

## Resume and Scaling

```sh
bash train.sh --resume runs/supervised/checkpoint.zip --epochs 3
```

`--epochs` is the total target, not the number of additional epochs. Resume
restores model weights, Adam state, epoch, game/chunk cursor and accumulated
metrics. The interrupted game's history is reconstructed without repeating its
completed updates. Up to 99 updates since the last periodic save can be lost
on an unplanned interruption. Validation restarts if interrupted. Use the same
output directory, data, batch size, rank filter, seed and loss settings. Altered
prepared-data inventory is rejected. Keep converted NPZ contents immutable;
the inventory guard is not a full checksum scan on each training launch.

For another run, use `--output runs/second`. `--init path/to/policy.zip` loads
only weights and deliberately starts a new optimizer and counters. This is also
how to start training on an expanded dataset. Do not use smoke-test weights.

Arguments after `train.sh` override defaults, for example `--batch 4` for lower
memory use or `--epochs 3`. No GPU time or peak-memory figure has been measured
locally; local JAX only sees a CPU. Begin with the bounded conversion above before
committing the full rental budget. For a bounded runtime check, use
`--max-updates 100` and then resume without that limit. The limit counts total
optimizer steps, including any restored steps; a mid-epoch pause produces a
checkpoint but no new validation-selected policy.

## Later PPO Experiments

```sh
python -u -B ppo.py --init runs/supervised/policy.zip --init-pass-scale 0.03 --output runs/ppo
```

This uses the same network with a new optimizer and on-policy value targets.
The default map is 24x24 with minimum general distance 17. Pass odds are adjusted
once at initialization and remain trainable. Choose a new output directory.
Frozen opponents are evaluated every 50 iterations and at the
end; each opponent gets 128 color-paired games with 16 concurrent games. This
evaluation can take longer than training iterations. Progress and results are
printed, and `evaluation.jsonl` contains summaries and changes from the first evaluation.
See README for evaluation flags and the versioned weights under `snapshots/`.
PPO collects 512 environments for 512 half-turns per iteration. The effective
minibatch is 1,024, accumulated in microbatches of 16; inference batches contain
32 player observations. The lossless host rollout buffer needs about 8.3 GiB of
system RAM, plus model/runtime overhead. Lower `--microbatch` and
`--inference-batch` to reduce GPU memory. The entry point has no automatic
curriculum or exact interrupted-rollout resumption. GPU throughput remains
unmeasured locally. Supervised fit alone does not establish strength.

## Local Checks

`python -B test_agent.py` runs bounded CPU-compatible checks including exact
supervised resume equivalence. `ALLOW_CPU=1 bash train.sh ...` allows explicit
CPU checks; normal cloud training refuses a missing GPU. No Ruff installation is
required. No generated files are written outside this folder by the agent code.
