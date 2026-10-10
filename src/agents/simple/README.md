# GenFormer

GenFormer studies a network-architecture upgrade to the observation-memory and self-play approach used by Nebula. It uses a policy-value transformer, terminal win/loss rewards, PPO, GAE, advantage filtering and parameter EMA. The network adds cell-level representations, state-conditioned relation biases and source-target action scoring.

This directory is independent: copy it to a machine with the dependencies installed. It does not import another experiment, the graphical interface or the local C++ engine.

## Inputs

`features.py` accepts only a player's masked observation. Relative ownership is 0 for neutral/hidden, 1 for self and 2 for the opponent; visibility distinguishes neutral from hidden.

- The current observation and eight preceding received observations retain all terrain, ownership and army fields, the five public statistics, and the preceding submitted action. Timestamps distinguish skipped half-turns. Missing records have a validity mask.
- Persistent per-cell records retain the last visible terrain, owner, army and observation time, the last observed enemy army and its time, and the last confirmed exposure to enemy sight. These are historical facts, not predictions of current hidden ownership or army.
- A separate 512-observation window retains both players' public land and army counts and timestamps.
- The playable-area mask and legal-action mask are separate. An unknown obstacle is not treated as a confirmed mountain.

The model receives raw records in float32. Its numeric embedding uses both linear scaling and logarithmic scaling without clipping. The network computes visibility coverage counts directly from recorded ownership. Enemy coverage is a lower bound away from our territory because hidden enemy sight sources are unknown. Production phases are the half-turn modulo 2 and modulo 50; no future production total or action benefit is supplied.

The bounded history is not a sufficient statistic of the complete game history. Older movement sequences can be lost even though last-observed facts remain. The architecture does not claim to represent every possible history-dependent policy.

## Network

`model.py` implements `GenFormer`.

For each cell, a learned temporal attention layer reads its complete recent records and its dated memory. The current observation also has a residual path. Four public-statistic histories are independently embedded into tokens, with learned identities, and join the spatial tokens in every global attention block.

Within attention head $h$ of layer $l$, the cell-to-cell logits are

$$
L^{l,h}_{ij} = \frac{q^{l,h}_i\cdot k^{l,h}_j}{\sqrt{d_h}} + B^{l,h}_{ij}.
$$

Let $x_i^l=\operatorname{LayerNorm}(h_i^l)$ be the token representation entering attention in layer $l$, and let $I$ contain all board cells and four public-history tokens, excluding padding. The bias receives the learned global summary

$$
\alpha_i^l=\frac{\exp(s_l(x_i^l))}{\sum_{j\in I}\exp(s_l(x_j^l))},
\qquad g^l=\sum_{i\in I}\alpha_i^l x_i^l.
$$

Each $s_l$ is a learned linear score. The weights are not fixed equally and do not encode prescribed cell importance. This is still a lossy weighted sum, not a sufficient representation of the position; individual tokens remain available to attention and to the cell-conditioned bias.

For spatial tokens $i,j$, let $\phi^l_{ij}$ concatenate $R$ learned displacement templates $T^l_r(p_j-p_i)$ and three rule relations: $E^{\mathrm{known}}_{ij}$, $E^{\mathrm{possible}}_{ij}$ and $V_{ij}$. The first two describe orthogonal adjacency between confirmed traversable cells and between cells not ruled out as traversable, respectively. The third describes the local visibility neighborhood. The additive bias is

$$
B^{l,h}_{ij}=\sum_{r=1}^{R+3}
\left[c^{l,h}_r(g^l)+u^{l,h}_r(x_i^l)+v^{l,h}_r(x_j^l)\right]\phi^l_{ij,r}.
$$

The global, source-cell and target-cell coefficients are separate learned linear projections. All three start at zero, so the bias starts at zero without restricting its learned sign. These relations are not paths, danger estimates or action scores. The global projection shares its coefficients across cell pairs, while the other two condition them on each endpoint. Here source and target refer to the querying and attended cells, which need not be adjacent; they are not a proposed movement command.

Content attention remains available between all playable cells, including across mountains. Only padding is excluded from internal global attention. Local residual convolutions supplement the global computation. This preserves a general content-interaction route instead of forcing every representation through the listed relations.

An executable move has a source cell $s$, an orthogonally adjacent destination $d$, and a full/half choice $m$. Its score combines a general nonlinear path with a learned multiplicative interaction:

$$
\ell(s,d,m)=f_\theta(h_s,h_d,e_{d-s,m},z)
+\frac{q_m(h_s)^\top k_m(h_d)}{\sqrt{D}}.
$$

Here $D$ is the network width. Full and half moves have separate source and target projections; all locations share these projections. The source projection starts small to avoid overwhelming the general path at initialization. The output context $z$ is distinct from $g^l$: it is a learned projection with SiLU activation of the final tokens' masked mean, coordinatewise maximum, and the current public-statistics embedding. That output summary is unchanged. Each cell representation has already passed through global attention, so an adjacent action can depend on distant cells without adding long-range commands.

The policy emits eight actions per source cell and one global pass. No gain, threat, pursuit target, predicted opponent response or plan score is added to the logits. A categorical value head estimates the terminal outcome in $[-1,1]$.

These are finite-capacity learned encodings, not a proof of unrestricted expressiveness. In particular, temporal attention and timeline embeddings compress their inputs, and the current network configuration has a fixed padded board size.

The default model has width 128, depth 6, four attention heads and 3,683,440 parameters. Each head has 32 channels. This configuration is a capacity choice, not a measured optimum or a proven parameter saving from relation biases. The dominant block terms remain approximately $30D^2$: $4D^2$ for attention projections, $8D^2$ for the feedforward layers and $18D^2$ for the two local convolutions, with additional parameters for relation coefficients and learned pooling. Token count controls activation and attention costs, not the dimensions of these shared parameter matrices.

## Execution

The network uses Equinox, optimization uses Optax, and observations, environment transitions and training tensors remain in JAX. Parameters and optimizer state are float32. BF16 applies to matrix products and convolutions on GPU; normalization and probability reductions use float32.

Equal relative displacements share geometric-template evaluation before gathering cell-pair biases. The observation-dependent adjacency relations are built once per forward pass and reused across layers. The action head evaluates its first linear layer as $W_s s + W_t t + W_c c + b$, projecting cells before distributing them across moves. These are algebraic rearrangements, not additional decision rules.

The eight spatial orientations use precomputed forward and inverse action permutations. Legal masks, historical commands and selected actions use the same mappings. Orientation is a runtime array argument, so changing it does not require eight separately compiled collection functions. The mappings contain no learned values and do not enter checkpoints.

Collection compiles 64-step segments with `jax.lax.scan`. Each segment includes inference, action sampling, environment transitions and observation-history updates, without a Python round trip at every half-turn. A packed rollout is allocated once and each segment is written into it with buffer donation. Only scalar transition records are concatenated for GAE; there is no full-rollout spatial concatenation or copy through the finishing function. Collection reuses the observations returned by the environment. When a game reaches its time limit, its final observation supplies a bootstrap value without committing that observation to the next game's history. Progress is reported between segments.

PPO reconstructs each microbatch directly from packed device arrays inside the compiled gradient-accumulation scan. It does not first expand the entire effective minibatch. Optax clips the accumulated gradient before applying Adam. Nonfinite loss or gradients prevent that optimizer update. The outer loop reports optimizer progress and checks the optional KL limit between minibatches.

JAX compilation is part of normal execution. The first call for each shape adds compilation time; measure subsequent iterations to assess throughput. Each collection segment reports `segment_seconds`, including compilation when first encountered. Iteration logs separate `rollout_seconds`, `finish_seconds` (included in rollout time) and `ppo_seconds`. PPO reports the first completed optimizer step as well as every 32 steps, so initial compilation can be distinguished from subsequent progress. CPU correctness checks do not establish GPU memory use or speedup.

## Training

`train.py` uses one shared policy for both sides. `env.py` runs batched games through `generals-bots`. This is a single-device implementation: compiled collection and optimizer kernels are coordinated by Python, with no framework boundary between the environment and network.

Rewards are terminal +1/-1, with no economic, exploration or survival bonus. Time limits bootstrap from the final observation of the interrupted game, never the new game's initial position. Both termination and truncation stop the GAE trace. PPO stores the exact augmented inputs and sampling probabilities. Eight spatial orientations transform observations, memories and submitted actions consistently.

Network parameters stay fixed during each rollout. Finished games restart immediately, while unfinished games and their observation histories continue into the next iteration. The next rollout uses the updated policy, so a game can span several parameter updates. The rollout boundary uses a value estimate for the unobserved continuation; it is not treated as a terminal outcome.

Each iteration collects 512 half-turns from 512 environments: 262,144 environment transitions, or 524,288 player samples. Absolute normalized advantage selects 25%, leaving 131,072 samples. One PPO epoch with effective minibatches of 1,024 makes 128 optimizer steps. `--fraction 1` disables filtering. The value loss is HL-Gauss, with 128 bins and sigma 0.04; gamma is 1, GAE lambda is 0.9, PPO clipping is 0.2, the value coefficient is 0.5, and the gradient norm is clipped to 0.267. KL is logged but does not stop updates by default. Logged gradient norms are measured before clipping.

For zero-based training iteration $t$, the learning rate is $\operatorname{clip}(0.5(t+1)^{-1.1},5\times10^{-6},10^{-4})$ and the entropy coefficient is $0.05(t+1)^{-0.2}$. EMA uses decay 0.999 once per complete iteration. EMA weights are exported by default, while both current and EMA parameters remain in the checkpoint.

`--microbatch` limits the players processed by a single PPO forward/backward pass. Gradients accumulate over these chunks before clipping and updating once per effective minibatch. `--inference-batch` independently controls sampling chunks, which require no backward activations; it defaults to microbatch when omitted. Each observation frame is stored once, categorical fields and legal masks are packed, and histories are reconstructed when sampled. Army values and timestamps retain float32 precision. The default packed rollout occupies approximately 8 GiB on the selected device, excluding the map pool, current observations, model, optimizer and one temporary 64-step segment. Completed spatial segments are released after writing into the reusable buffer. Peak memory is higher than the packed size. Smaller microbatches reduce activation memory but do not reduce the rollout or effective optimizer batch.

Spawn-distance curriculum stages advance only after the current policy exceeds the configured win-rate gate against random legal movement, evaluated on 512 games every 50 iterations. This gate measures learning progress, not competitive strength. Ongoing games retain their maps when the pool changes. The default run budget allows 100,000 iterations; this is a stopping budget, not a measured requirement for strong play. Use `--until` to stop earlier without changing the schedules. The curriculum gate, map-pool settings, evaluation frequency and run budget are implementation choices, not hyperparameters specified by the paper's Table V.

There is no behavior-cloning or DAgger phase. The algorithm in `baseline.py` is used only when explicitly selected as an evaluation opponent.

The default sampling and PPO settings follow Appendix D of [Superhuman AI for Generals.io Using Self-Play Reinforcement Learning](https://arxiv.org/abs/2606.23348). The observation representation and network remain GenFormer's; matching training settings does not imply matching the paper's throughput or playing strength.

## Commands

Install a GPU-enabled JAX build appropriate to the machine before training on CUDA. `requirements.txt` declares the common dependencies; it does not install a GPU driver or choose a CUDA runtime. CPU execution needs no CUDA. `train.sh` runs the active `python3`, or the executable named by `PYTHON`, and defaults to `--device cuda:0 --microbatch 128 --inference-batch 256`. These execution settings target a 32 GiB training GPU, not a measured hardware-independent optimum. Check peak memory through collection, PPO and evaluation; larger networks need additional space. Arguments passed to the script override its defaults without changing sampling or PPO settings.

```sh
python3 -m pip install -r requirements.txt
bash train.sh
bash train.sh --output runs/genformer
bash train.sh --resume runs/train/checkpoint.eqx
bash train.sh --resume runs/genformer/checkpoint.eqx --microbatch 64 --inference-batch 128
python3 -B evaluate.py runs/genformer/policy.eqx --baseline --device cuda:0 --games 32
python3 -B evaluate.py runs/genformer/policy.eqx --opponent runs/other/policy.eqx --device cuda:0 --games 32
python3 -B agent.py runs/genformer/policy.eqx
```

Use `bash train.sh --device cpu --microbatch 32 --inference-batch 32` or run `python -B -u train.py --device cpu` directly. Network capacity, environment count, rollout length, minibatch size and history lengths can be configured through `train.py --help`. Resume restores saved settings, optimizer, iteration, curriculum and action-sampling RNG state; in-progress games restart. Microbatch and inference batch may change on resume. Resume is not bit-for-bit trajectory continuation.

`checkpoint.eqx` is resumable. `policy.eqx` contains only the exported network and input configuration. Both use native Equinox leaf serialization after a JSON metadata line. The default save interval is 10 completed training iterations; evaluation and normal completion also save. Exported policies use the existing initialization/observation/action text protocol through stdin/stdout; diagnostics go to stderr. Checkpoints must match the current network structure. No conversion of earlier architectures or PyTorch `.pt` files is implemented; start the changed architecture in a new output directory.

Evaluation plays the same maps from both colors and reports wins, losses, terminal draws and time-limit truncations separately. Omit both `--baseline` and `--opponent` to use the random legal mover. Evaluation defaults to CPU; pass `--device cuda:0` to use the GPU for both the environment and network. A completed CPU smoke test checks execution, not learning quality. GPU memory use, sustained throughput and competitive strength still require measurements on the intended machine.

## C++ Deployment

Training and evaluation remain in JAX. `export.py` copies a native `.eqx` policy or checkpoint into an equivalent float32 TorchScript module for CPU deployment. It includes observation history, dated memory, action masking and the network; it does not train or distill another policy. PyTorch is an optional export dependency, not a training dependency:

```sh
python -m pip install torch
python -B export.py runs/train/policy.eqx policy.pt
python -B export.py runs/train/checkpoint.eqx policy.pt --weights ema
```

`--weights selected` is the default. A resumable checkpoint can also select `model` or `ema`; an exported `policy.eqx` already contains only its selected weights. The resulting `.pt` file is for inference and cannot resume JAX training. Floating-point results can differ slightly between runtimes, so nearly tied actions need not always have the same argmax.

The repository's CMake configuration builds `infer.cpp` as `build/simple/simple.exe` on Windows, or `build/simple/simple` on Linux. `SIMPLE_BUILD_AGENT` defaults to `ON`. CMake uses an available Torch package or queries PyTorch through the discovered Python interpreter; it does not hard-code a machine-specific library path. The executable requires LibTorch and its runtime libraries, but no Python, JAX or CUDA at match time. Windows builds copy the Torch DLLs beside the executable. Use a PyTorch export installation compatible with the deployment LibTorch version.

The executable takes one model-path argument, whose basename is unrestricted. For example, the interface's Player field can contain:

```text
"D:\Repository\Games\generals\build\simple\simple.exe" "D:\Models\genformer.pt"
```

It reads initialization and observations from stdin, chooses the highest-scoring legal action and flushes the standard five-integer reply to stdout. Diagnostics go to stderr. Inference uses one CPU thread. The model's padded `side` bounds both input dimensions; larger maps are rejected. Quoting is necessary for paths containing spaces. The executable does not search for a model beside itself or assume the name `policy.pt`.
