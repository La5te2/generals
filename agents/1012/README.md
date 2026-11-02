# Agent 1012

An independent experiment in player-history representation and a medium-sized
GNN/attention policy. Human replay supervision and PPO use the same network.
No code is imported from another project folder. This is not a trained strong
agent and finite history is not claimed to be a sufficient information state.

## Representation and Network

### Training pipeline and model interface

Historical replays are reconstructed into player-visible observations and recorded
actions. `features.py` updates the player's memory, `model.py` maps that memory
to action probabilities, and `train.py` fits those probabilities to replay actions.
The separate `ppo.py` entry point then learns from freshly sampled self-play games.
Both stages share the same actor architecture and action indexing.

For a board with N cells and history length H=8, the model receives:

| Tensor | Shape | Contents |
| --- | --- | --- |
| Current spatial features | N x 25 | Ten current observation channels, seven last-seen channels and eight memory/coverage channels |
| Recent spatial history | H x N x 13 | Ten observation channels plus previous-action source, target and half-move source markers |
| Public history | H x 11 | Time, growth phases, both army totals, both land totals, seat and three rule flags |
| Masks and graph | Board-dependent | Valid history slots, real cells, movement edges and legal actions |

The ten observation channels encode visibility, own/enemy/neutral ownership,
mountain, city, general, plain fog, obscured structure and visible army.
Last-seen channels preserve own/enemy/neutral ownership, mountain, city, general
and army. The eight extra channels encode ever-seen status, observation age,
own coverage, visible-enemy coverage, ever-exposed status, current own ownership,
known mountains and unexplored obscured structures.
Army values use log(1+a)/log(256), with a clamped to zero from below.
Public time uses tick/512, tick modulo 50 divided by 50, and tick modulo 2.
Land totals are divided by the number of real cells. Rule flags identify city
regeneration, historical action priority and general-trade behavior.

Each cell's temporal attention compresses its H spatial frames to 128 values.
A separate temporal attention compresses public history to 128 values and
broadcasts it to cells. Their concatenation with the projected current spatial
features is mapped to the 128-wide graph representation.
The final output has 8N+1 action logits, one value scalar, N general-location
logits, N x 3 ownership logits and N army predictions. Move index 8u+2d+b
selects source cell u, direction d and full/half selector b. Index 8N is pass.

### Learning objectives

Let I be the encoded player history, a the recorded action, pi the masked policy
and w the example weight. Supervised learning minimizes the weighted mean of:

$$
L_{SL}(I,a)=-\log\pi(a\mid I)+c_{aux}L_{aux}(I).
$$

The auxiliary loss adds enemy-general cross-entropy to the hidden-cell mean of
ownership cross-entropy and Huber loss on encoded army counts. Simulator truth
provides targets for these heads. The actor forward pass consumes the inputs
listed above. Supervised training selects weights by validation action NLL.

During PPO, let r_t be the outcome reward, V_t the value estimate, gamma the
discount and lambda the trace decay. On continuing transitions the estimator is:

$$
\delta_t=r_t+\gamma V_{t+1}-V_t,\qquad
A_t=\delta_t+\gamma\lambda A_{t+1}.
$$

Terminal states use zero bootstrap. Episode boundaries stop trace propagation,
and truncations retain a value bootstrap. With probability ratio rho_t between
the updated and rollout policies, the clipped policy objective is:

$$
L_{policy}=-\mathbb{E}\left[\min\left(\rho_t\widehat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right].
$$

Here epsilon is the PPO clip width and the hat denotes advantage normalization.
The total update also includes value regression, an entropy bonus and the
configured auxiliary loss. Adam updates follow global gradient-norm clipping.
The PPO configuration and evaluation protocol are described below.

`policy.zip` contains model leaves and JSON metadata. `checkpoint.zip` additionally
stores optimizer state and training progress. The bundled `policy-original.zip`
and `policy-initial.zip` are preserved historical initialization artifacts.
Their archive metadata records provenance and configuration. `bash train.sh`
starts supervised training with random parameters by default. Explicit `--init`
selects a weight archive and `--resume` selects a resumable checkpoint.

### Spatial processing and heads

The default network has width 128, six residual blocks, four attention heads,
and 2,505,006 trainable parameters. Every block has relation-conditioned graph
messages and a 512-wide feed-forward layer. Blocks 2, 4 and 6 also have global
attention. There are no tactical controllers or hand-scored plans.

Each cell receives current masked observation, last-seen owner/army/terrain,
last-seen age, observation coverage and confirmed exposure history. The enemy
coverage channel is a visible-source lower bound in general, and exact on our
own land. It is NOT obtained from the hidden opponent ownership map. Historical
ownership does not silently become current ownership. Global public totals,
half-turn phase, player seat and rule flags are retained separately.

The last eight observation/action frames and public-stat frames are encoded by
learned temporal attention with relative ages. Last-seen facts persist beyond
this window. No opponent action or future frame is supplied. This representation
can lose long-range historical correlations; that is an empirical limitation,
not a theorem that older history is unnecessary.

Four-neighbor movement and eight-neighbor vision are separate graph relations.
Messages depend on source, target, their elementwise product and edge direction.
Known mountains block movement messages, not vision or global attention.
Unexplored obstacles are uncertain, not assumed to be known mountains. Relative
displacements bias global attention without removing content-based interactions.
Shared cell processing is equivariant to consistent node reindexing; that does
not prove generalization to unseen map distributions.

A learned attention pool produces global context. Every directed neighboring
edge receives full/half-move scores from source, target, their product, direction
and context. Pass is a separate action. Only impossible observable moves are
masked. The value head uses the player's history representation, never the full
hidden state. Optional auxiliary heads predict enemy-general location, owner
marginals and log army counts. Hidden truth is used only in their targets.
These marginal predictions are not a full joint belief distribution.

## Data

### Sources and local placement

This version-controlled folder contains code, documentation and initial weights.
Download the datasets separately into `data/raw/` before running `prepare.sh`.
The official [generals.io Replay Utils repository](https://github.com/vzhou842/generals.io-Replay-Utils)
directs readers to the [developer replay page](http://dev.generals.io/replays)
for the large replay archive. The Google Drive link below is the archive used
in this experiment. Its compressed size is about 1.17 GB, and extracted or
converted data requires additional space. The GitHub repository provides the
replay utilities, while the archive is hosted separately.

Save that download as `data/raw/replays_prod.tar.gz`. Download the Parquet file
from the Hugging Face dataset below and save it as `data/raw/high_rank.parquet`.
Keep the tar archive compressed: `data.py` streams its members directly.
Generate a local inventory with the `--inspect` command in Commands, then run
`bash prepare.sh` and `bash train.sh`. The original pilot data is also external
to this source snapshot. Run `smoke.py --data data/processed/train` after preparation
when checking the network on a small batch.

Both original datasets belong under `data/raw/`:

- [Official replay archive](https://drive.google.com/file/d/1EVZlnSAN_HKPe2U2TdFIb0KQVGWeNsEU/view),
  `replays_prod.tar.gz`, 1,167,869,329 bytes. A `.gioreplay` JSON archive; inspected
  complete scan found 212,290 games, all v5; 134,570 have two players. Multiplayers
  are excluded from the 1v1 training input.
- [strakammm/generals_io_replays](https://huggingface.co/datasets/strakammm/generals_io_replays),
  `high_rank.parquet`, 61,449,177 bytes. The author reports high-rank 1v1 cleaning.
  There are 18,803 games: 10,613 v5 and 8,190 v13. It also includes bot players
  such as Human.exe, so it is not purely human.

All 10,613 v5 IDs in the high-rank set also occur in the official archive. Its
8,190 v13 games are additional data, not 18,803 additional games. Complete
inventory results are in `data/inventory.json`. The combined pool contains at
most 142,760 distinct two-player games before team/quality/rule filtering.

The official historical simulator is vendored unchanged in `vendor/replay/`,
at [552664d7c65a2fe8c8162f51236af1962eb75dbd](https://github.com/vzhou842/generals.io-Replay-Utils/tree/552664d7c65a2fe8c8162f51236af1962eb75dbd),
under its MIT license. `replay.js` adapts it for standard two-player v5/v6/v7/v13
records. Recorded move order is preserved; v5 neutral-city regeneration is
preserved. v13 records use the same supported standard board mechanics, with
array-form events normalized by the adapter. Unsupported terrain/modifiers and
other revisions are rejected, not silently treated as current games.

PPO uses the installed `generals-bots` current-rule environment. Three public
rule flags identify regeneration, old alternating priority and general trades.
Historical rule behavior need not be optimal under current rules; PPO must learn
that difference. A small supervised fit cannot establish successful transfer.

Conversion rejects invalid executed moves and unconfirmed endings. Labels that
were only executable after the opponent's same-tick action are not trained as
pre-tick legal actions. Pass frames are retained, but surrender is not labeled
as pass. Duplicate replay IDs are skipped across sources. A canonical initial
map hash assigns 85/10/5 train/validation/test splits; both seats and identical
maps stay together. This is not a disjoint-player split, nor does it guarantee
rotation/reflection families are disjoint. Processed files store current views
and separate truth targets, not repeated full history windows.

## Commands

For a standalone cloud upload and the setup/prepare/train scripts, see
[CLOUD.md](CLOUD.md). Start with supervised learning; PPO is a separate command.

Python 3.13 was used locally. Install `requirements.txt` in the chosen environment.
CUDA training requires a GPU-enabled JAX installation appropriate to the machine;
the requirements do not replace it with a hard-coded CUDA path. Node.js is needed
only to convert historical replays. Ruff is not a runtime dependency.

Run commands from this directory:

```sh
python -B data.py data/raw/replays_prod.tar.gz data/raw/high_rank.parquet --inspect --report data/inventory.json
python -B data.py data/raw/replays_prod.tar.gz data/raw/high_rank.parquet --output data/processed --report data/conversion.json
python -B test_agent.py
python -B smoke.py --data data/pilot/train
python -B train.py --data data/processed --min-stars 70
python -u -B ppo.py --init runs/supervised/policy.zip --init-pass-scale 0.03 --output runs/ppo
```

For a pilot, pass `--limit 16` to each conversion command. Without a limit,
conversion may generate a large amount of data and many map-shaped JAX
compilations. Archive members are streamed without filesystem extraction.
Data conversion is a CPU/offline task, not part of PPO sampling.

The local `.deps/` directory, when present, only holds the Parquet reader used
for this experiment. Locally, prefix conversion with
`$env:PYTHONPATH="$PWD/.deps"` in PowerShell. It is not required when pyarrow is
installed normally. Do not copy `.deps/` between operating systems.

`train.py` trains policy cross-entropy plus an adjustable auxiliary loss
(`--auxiliary 0` disables it). Outcome labels do not pretrain the value as though
human play were the current policy. Training and validation report move accuracy,
pass accuracy and action NLL separately. Rank filtering applies per player.
The best held-out action NLL selects `policy.zip`; it is not a strength rating.
`checkpoint.zip` includes Adam state, the supervised game/chunk cursor and
accumulated metrics. `--resume` restores them; shuffle order is reproducible and
the interrupted game's causal history is reconstructed. Data and training settings
must remain unchanged. `--init` loads weights only and starts a new optimizer.
PPO does not yet support exact interrupted-rollout resumption.

`ppo.py` is a single-device self-play entry point. Both seats sample the same
frozen-in-rollout policy, then PPO updates that policy. It uses terminal +/-1,
gradient clipping, a fresh on-policy buffer and proper truncation bootstrapping.
Auxiliary targets also come from these fresh rollouts (`--auxiliary 0` disables
their loss). They never change the actor's observation or sampled action.
Games can span iterations. Change map size/distance with a new invocation and
`--init`; no automatic promotion or unvalidated curriculum gate is imposed.
Defaults collect 512 environments for 512 half-turns: 262,144 environment steps
and 524,288 player samples per iteration. All samples receive one shuffled epoch,
with effective minibatch 1,024 and 16-example gradient accumulation chunks.
Gradients are clipped at global norm 0.5 after accumulation, once per optimizer
step. The rollout buffer uses CPU memory (about 8.3 GiB at these defaults);
binary features are bit-packed, floats retain float32 precision, and consecutive
observations share their history frames. Only selected minibatches enter GPU
memory. Inference runs in batches of 32; each transfer stores eight half-turns.
Direct PPO invocations default to on-demand JAX GPU allocation. An explicitly
set `XLA_PYTHON_CLIENT_PREALLOCATE` environment variable takes precedence.
`--microbatch`, `--inference-batch` and `--rollout-chunk` control accelerator peak
memory without changing the effective optimizer batch or sampled rollout size.
`--pool` defaults to 4,096 maps. GPU throughput remains unmeasured locally.

### PPO Evaluation

PPO defaults to 24x24 maps and minimum general distance 17. `--init-pass-scale`
adds `log(scale)` to the pass bias once, before sampling and optimizer creation.
The bias remains trainable; all subsequent PPO probabilities use that policy.
Use a new output directory for each run. Existing outputs are preserved.

Frozen evaluation runs every 50 iterations and at the end. Training begins with
sampling; evaluation maps and games are deferred until the first scheduled check.
Each opponent gets 128 games, using 64 independently generated held-out maps
with both starting positions, in batches of 16 games. Opponents are the original
policy, the pass-adjusted initial policy, up to two historical PPO snapshots,
and the bundled JAX `baseline.py`. Training opponents remain the current policy.
All network players use categorical sampling during these evaluations.

`--evaluate`, `--evaluation-games`, `--evaluation-environments`,
`--evaluation-seed` and `--evaluation-history` configure this work.
`--evaluate 0` disables evaluation. Evaluation can take substantially longer
than a short PPO iteration; progress is printed every 32 simulated half-turns.
Game-specific random streams keep sampling independent of batch scheduling;
hardware/backend numerical differences can still change sampled trajectories.

`references/` preserves both initial policies. `snapshots/` keeps versioned
weights every `--save` iterations and at evaluation points. `checkpoint.zip`
and `policy.zip` hold the latest saved training state and weights respectively.
`training.jsonl` records updates; `evaluation.jsonl` records opponent summaries.
`evaluation/iteration-*/` contains per-game results, map hashes and weight hashes.
`change_from_first_evaluation` compares each fixed opponent's results with its
first scheduled evaluation on the same map/color pairs; `comparison_iteration`
identifies that reference. Increasing the requested game count retains the
existing map sequence in standalone evaluations.
Timeouts are reported separately. Net win rate is `(wins - losses) / games`;
timeouts contribute zero to that descriptive statistic. The 95% intervals use
conservative map-paired Hoeffding bounds for a fixed evaluation, with no
correction for repeated checkpoint selection. No automatic promotion is applied.

Independent evaluation of a saved policy uses the same rules and inputs:

```sh
python -u -B evaluate.py runs/ppo/policy.zip --baseline --output runs/baseline-evaluation.json
python -u -B evaluate.py runs/ppo/policy.zip --opponent runs/ppo/references/initial.zip --output runs/initial-evaluation.json
```

The optional `--greedy` flag selects argmax for both network players. Keep the
selection method, map configuration, horizon and seed fixed across comparisons.

## Checks and Scope

`test_agent.py` checks observation agreement with the installed environment,
hidden-state invariance at fixed observations, history reset, action legality,
node-reindexing equivariance, historical city growth, game-level splitting,
supervised gradients, weight round trips, GAE boundaries and an actual PPO update.
`smoke.py` fits a tiny batch of real recorded moves with the full default network.
This checks trainability, not held-out skill. Its weights belong to `runs/smoke/`
and must not be mistaken for a competitively trained policy.

Deployment export, replay visualization,
joint-belief inference/search and large-scale training are outside this first
experiment. There is no claimed win-rate improvement yet.

Initial CPU checks passed all eight tests. The full default network fitted four
real recorded moves in 16 updates: action NLL 2.5092 to 0.2272, training-batch
accuracy 0% to 100%. This is intentionally an overfit check, not a validation
result. Raw measurements are in `runs/smoke/result.json`. The pilot conversion
contains nine games from the official archive and eight from the high-rank set
(15 train, 1 validation, 1 test), not a full conversion of both archives.

The cloud-readiness pass adds a ninth check: interrupted supervised training
restores exactly the same model and optimizer arrays as uninterrupted training
on the same CPU backend. The actual full-network CLI also completed a bounded
train/save/resume/validation/policy-save cycle using real replay prefixes. These
are functional checks, not new generalization or win-rate measurements.
