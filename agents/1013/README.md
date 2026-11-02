# Agent 1013

Agent 1013 uses JAX, Equinox and Optax to train a partially observed Generals policy. A shared GNN/attention encoder supports an action policy, a value function, hidden-state predictions and opponent predictions. PPO learns from self-play against frozen versions of the current policy and its two predecessors. Potential-based economic rewards provide intermediate feedback; simulator-derived labels supervise the prediction heads.

This directory contains the complete training runtime, periodic evaluation, dependencies and initialization weights. Copy this directory to the cloud. Training requires no other repository directory, human replay dataset, Node.js installation or C++ build.

## Start and resume

Use Python 3.12 or later and an NVIDIA GPU with a compatible driver. Setup installs the pinned JAX CUDA 12 backend by default and verifies GPU availability.

```bash
bash setup.sh && bash train.sh
```

The default run starts at iteration 0, trains for 100 iterations and exits. There is no startup evaluation. Every completed iteration saves the latest full checkpoint, the latest policy and an immutable policy snapshot. Evaluation runs every 10 iterations and at completion, after saving.

- `initial/policy.zip` contains the adapted starting policy. A new run creates fresh optimizer moments, random streams and games. Existing action parameters, including the pass head, are preserved.
- `initial/original.zip` contains the original supervised policy. It is used exclusively as a fixed evaluation reference.
- `--init` selects another 1013 policy as a fresh initialization.
- `--resume` restores a full 1013 training checkpoint.

```bash
bash train.sh --resume runs/ppo/checkpoint.zip
```

The iteration limit includes completed iterations. To continue from iteration 100 to iteration 150:

```bash
bash train.sh --resume runs/ppo/checkpoint.zip --iterations 150
```

Custom runs must repeat their statistical training settings and output directory on resume. Microbatch size, inference batch size and rollout transfer chunk may change independently. A new run refuses to overwrite a nonempty output directory. Setup also accepts CUDA 13 through `bash setup.sh cuda13`. Small CPU checks can use `bash setup.sh cpu` and `ALLOW_CPU=1 bash train.sh` with appropriately reduced run dimensions.

## Runtime modules

- `ppo.py` coordinates self-play, PPO updates, the frozen opponent pool, saving and periodic evaluation.
- `environment.py` generates maps, advances batched games, collects learner transitions and computes advantages.
- `features.py` maintains observable facts and bounded history, constructs graph relations and masks legal actions.
- `model.py` defines the encoder and output heads.
- `rewards.py` computes material quantities, economic potentials and shaped rewards.
- `rollout.py` stores packed trajectories in CPU memory and reconstructs minibatches.
- `train.py` implements checkpoint persistence, the optimizer and auxiliary losses.
- `evaluate.py` measures frozen policies on held-out, seat-paired maps; `baseline.py` supplies its algorithmic reference opponent.

## Observation and memory

Let H and W denote board height and width, N = HW the number of cells, and t the half-turn index. The default board is 24 by 24, giving N = 576. Each player receives a separately masked observation. Hidden cells, visible neutral cells and previously observed cells remain distinct.

Inputs contain current spatial features, eight recent spatial frames and eight recent public-statistics frames:

$$
X_t\in\mathbb{R}^{N\times25},\qquad
R_t\in\mathbb{R}^{8\times N\times13},\qquad
U_t\in\mathbb{R}^{8\times11}.
$$

Additional inputs are the real-cell mask, history-validity mask, four-neighbor movement relation and legal-action mask. The following channel descriptions follow the implementation order.

### Spatial features

Channels 0 through 9 describe the current masked observation: visibility, own ownership, visible enemy ownership, visible neutral non-mountain status, visible mountain, visible city, visible general, ordinary fog, fog obstacle and visible army size. An unseen obstacle can be a mountain or city; its hidden type is excluded.

Channels 10 through 16 retain the last visible values of own ownership, enemy ownership, neutral status, mountain, city, general and army. These are historical observations. Ownership and army values can become stale in fog.

Channels 17 through 24 contain: ever seen, age since last visibility, own sight-source count divided by nine, observed enemy sight-source count divided by nine, cumulative confirmed enemy exposure, current own ownership, known mountain and never-seen fog obstacle. Current own ownership appears in both the current-observation and derived groups.

Army counts and observation ages use logarithmic scaling without an upper clip:

$$
\ell(a)=\frac{\log(1+\max(a,0))}{\log256}.
$$

Own sight-source counts are exact. Enemy sight-source counts use visible enemy-owned cells in the surrounding 3-by-3 neighborhood, including the queried cell itself. They are exact on current own territory, where every adjacent enemy source is visible, and lower bounds elsewhere. Cumulative exposure records whether an observed enemy source has covered a cell at any earlier time; it does not assert current enemy visibility.

### History and public statistics

Each recent spatial frame contains the ten current-observation channels and three markers for the player's preceding command: source cell, destination cell and half-army source. Pass produces no spatial markers. Each observation is appended once. Episode reset clears the history and persistent memory.

Each public-statistics frame contains: half-turn divided by 512, half-turn modulo 50 divided by 50, half-turn parity, logarithmic own and enemy total armies, own and enemy land fractions, player index, and three rule flags for regeneration, old action ordering and general trade. The current environment supplies those flags as `(False, False, True)`.

Persistent last-seen facts survive beyond the eight-frame window. The finite window and factual memory form an approximate history representation, without claiming an exact belief distribution over possible hidden states. Opponent identity, hidden opponent commands and the true opponent observation are excluded from encoder inputs.

### Relations and legal actions

The transport relation connects orthogonally adjacent real cells, excluding endpoints known to be mountains. Unknown obstacles remain uncertain in this relation. The sight relation connects all eight adjacent real cells. Parameters are shared across board positions.

An action selects a source cell, direction and full/half-army mode, or pass:

$$
\mathcal{A}=\{(u,d,b):0\leq u<N,\ d\in\{0,1,2,3\},\ b\in\{0,1\}\}\cup\{\mathrm{pass}\},\qquad
|\mathcal{A}|=8N+1.
$$

Directions are up, down, left and right. Mode 0 selects full movement; mode 1 selects half movement. The flattened index is 8u + 2d + b, and pass is 8N. Legal movement requires an owned source with at least two soldiers and a valid non-mountain destination. Every adjacent destination of an owned source is visible. Pass is always legal. The default policy has 4,609 logits.

## Network architecture

The default network has width 128, six graph blocks, four attention heads, eight history frames and 2,540,607 trainable parameters. All six blocks contain local transport/sight updates and a feed-forward residual. Blocks 2, 4 and 6 additionally contain global attention.

### Input encoding

Two independent temporal pooling modules encode per-cell history and public-statistics history. For a sequence x with valid positions J, each module projects the frames, adds learned history-position embeddings, scores them with a learned query and pools:

$$
y_j=\operatorname{GELU}(W_Tx_j+b_T+p_j),\qquad
\alpha_j=\frac{\exp(q_T^\top y_j)}{\sum_{k\in J}\exp(q_T^\top y_k)},\qquad
T(x)=\sum_{j\in J}\alpha_jy_j.
$$

All embeddings have dimension 128. The per-cell spatial projection, per-cell temporal embedding and broadcast public-history embedding are concatenated, then projected from 384 to 128 dimensions:

$$
h_u^{(0)}=W_{\mathrm{merge}}[W_XX_{t,u}+b_X\ ;\ T_{\mathrm{cell}}(R_{t,:,u})\ ;\ T_{\mathrm{public}}(U_t)]+b_{\mathrm{merge}}.
$$

Padding cells are zeroed. Temporal pooling uses valid history positions only. There is no GRU or other learned recurrent state.

### Local graph update

Each block first normalizes the embeddings. Edge features combine source, target, their elementwise product and the raw two-dimensional neighbor displacement:

$$
z_u=\operatorname{LN}(h_u),\qquad
e_{uv}=[z_u\ ;\ z_v\ ;\ z_u\odot z_v\ ;\ p_v-p_u].
$$

Each relation r has its own 386-to-128 message and gate projections. The fixed normalizers are 4 for transport and 8 for sight, including at boundaries:

$$
m_{uv}^{(r)}=\operatorname{GELU}(W_m^{(r)}e_{uv}+b_m^{(r)}),\qquad
c_{uv}^{(r)}=\operatorname{sigmoid}(W_c^{(r)}e_{uv}+b_c^{(r)}),
$$

$$
M_u^{(r)}=\frac{1}{K_r}\sum_{v\in\mathcal{N}_r(u)}c_{uv}^{(r)}\odot m_{uv}^{(r)},\qquad
h_u\leftarrow h_u+W_M[M_u^{(\mathrm{transport})}\ ;\ M_u^{(\mathrm{sight})}]+b_M.
$$

Masked edges contribute zero. The merge projection maps 256 to 128 dimensions. Source and target linear projections are computed per cell before gathering edges, and message/gate projections are fused. These changes preserve the algebraic edge functions.

### Global attention and feed-forward update

In blocks 2, 4 and 6, normalized embeddings produce queries, keys and values for four heads of dimension 32. A learned geometric bias uses signed logarithmic relative coordinates:

$$
\widetilde{\delta}_{uv}=\frac{\operatorname{sign}(p_u-p_v)\odot\log(1+|p_u-p_v|)}{4},\qquad
b_{uv}^{(a)}=w_a^\top[\widetilde{\delta}_{uv,1}\ ;\ \widetilde{\delta}_{uv,2}\ ;\ \|\widetilde{\delta}_{uv}\|_1].
$$

Attention combines content and geometry:

$$
\alpha_{uv}^{(a)}=\operatorname{softmax}_{v\in\mathcal{V}}\left(\frac{q_u^{(a)\top}k_v^{(a)}}{\sqrt{32}}+b_{uv}^{(a)}\right),\qquad
o_u^{(a)}=\sum_{v\in\mathcal{V}}\alpha_{uv}^{(a)}v_v^{(a)}.
$$

The concatenated head outputs are projected back to 128 dimensions and added residually. Global attention spans all real cells, including across mountains, allowing relationships outside local transport and sight. Pairwise attention has quadratic cost in cell count.

Every block ends with a pre-normalized feed-forward residual, expanding 128 to 512 dimensions and projecting back:

$$
h_u\leftarrow h_u+W_2\operatorname{GELU}(W_1\operatorname{LN}(h_u)+b_1)+b_2.
$$

Real-cell masks are reapplied after each block. Shared weights and relative coordinates support reuse across positions and board sizes; performance on unseen topology remains empirical.

### Production evidence and opponent representation

A final layer normalization follows the six blocks. Three additional scalars are derived from permitted observations. Let D_t be own total army minus enemy total army. At a consecutive even half-turn, the public production residual is

$$
q_t=P_{\mathrm{own},t}+\mathbf{1}_{50\mid t}(L_{\mathrm{own},t}-L_{\mathrm{enemy},t})-(D_t-D_{t-1}).
$$

For ordinary exchanges without general transfers, the resource identity gives

$$
q_t=P_{\mathrm{enemy},t}+N_{\mathrm{own},t}-N_{\mathrm{enemy},t},
$$

where P counts cities plus generals and N counts losses against neutral defenders during that transition. Known neutral losses permit an exact production calculation. Unknown neutral losses confound the residual. General transfers can invalidate the identity. The encoder supplies evidence without asserting an exact hidden-city count.

Let v_t indicate two valid consecutive observations ending on an even half-turn. The evidence vector contains the valid residual, its validity bit and visible enemy production:

$$
e_t=\left[\frac{v_tq_t}{N},\ v_t,\ \frac{P_{\mathrm{visible\ enemy},t}}{N}\right].
$$

Evidence is broadcast to every cell and combined with the normalized representation h:

$$
z_u=\operatorname{GELU}(W_z[h_u\ ;\ e_t]+b_z),\qquad
h'_u=h_u+W_oz_u+b_o.
$$

The first projection maps 131 to 128 dimensions; the second maps 128 to 128. W_o and b_o start at zero, preserving the initialization actor and value outputs up to numerical roundoff. Auxiliary supervision trains z, while policy/value gradients learn how the residual should affect decisions. This local-history opponent representation draws on the approach of [LIAM](https://arxiv.org/abs/2006.09447).

### Pooling and output heads

Learned scores pool real cells into global context g:

$$
\omega_u=\operatorname{softmax}_{u\in\mathcal{V}}(w_g^\top h'_u+b_g),\qquad
g=\sum_{u\in\mathcal{V}}\omega_uh'_u.
$$

Weights depend on the current representation; cells can receive different importance in different positions.

The actor combines source, target, their elementwise product, context and displacement. Its edge hidden layer maps 514 to 128 dimensions, followed by two full/half-army logits:

$$
l_{uv}=W_a\operatorname{GELU}(W_e[h'_u\ ;\ h'_v\ ;\ h'_u\odot h'_v\ ;\ g\ ;\ p_v-p_u]+b_e)+b_a\in\mathbb{R}^2.
$$

A linear pass head maps g to one logit. Invalid actions receive logit -1e9 before softmax. A separate linear value head maps g to an unbounded scalar estimate of the shaped return.

Three hidden-state heads use h': per-cell three-class ownership logits, per-cell normalized army prediction and one general-location logit per real cell. Ownership classes are neutral/unowned, own and enemy; general logits form a spatial categorical distribution.

The opponent-command decoder applies a direct 386-to-2 edge projection to source, target, their product and displacement from z. Its pass head uses z pooled with the same weights omega. This decoder masks only board boundaries and excludes the enemy's hidden ownership and legal-action mask.

The opponent-view decoder maps each z to ten outputs: nine binary current-view channels and one normalized army prediction. These are predictions from the player's own inputs; supervision targets remain outside the encoder.

## Self-play and collection

Let theta_x denote the policy after x completed iterations. During iteration x+1, the learner samples actions from theta_x. New episodes choose a frozen opponent uniformly from available versions:

$$
\mathcal{P}_x=\{\theta_j:\max(0,x-2)\leq j\leq x\},\qquad
\pi_x^{\mathrm{opp}}\sim\operatorname{Uniform}(\mathcal{P}_x).
$$

Iteration 1 starts with theta_0 playing against a frozen copy of theta_0. The learner samples its categorical policy; training opponents choose argmax. Each environment contributes one learner trajectory. Opponent commands supply auxiliary labels and are excluded from PPO importance ratios.

Defaults are 512 parallel games, 512 collection steps, 4,096 training maps and a 1,024-half-turn episode limit. Maps use fixed 24-by-24 dimensions and minimum general distance 17, generated reproducibly by the pinned `generals-bots` engine. The city-count range passed to the generator is `(1, max(2, side * side // 50))`; other generation details follow that engine. Reset samples another map from the pool. Learner seats alternate across environments.

Each rollout contains 262,144 learner transitions. Parameters stay fixed during collection, followed by PPO updates. An unfinished episode continues into the next iteration with the updated learner and its original frozen opponent. Time limits reset the episode but retain a value bootstrap from its final observation. Memory resets at episode boundaries.

Every iteration adds its new policy to the three-version pool. Retired versions remain allocated while unfinished episodes use them. This implementation replaces pool members once per iteration. The [2507 paper](https://arxiv.org/html/2507.06825v2#S5) describes three predecessors with a win-rate replacement threshold; that promotion threshold is not part of this trainer.

## Economic reward shaping

Let L, A and P denote owned land, total army and producing cells, with cities and generals both counted as producers. Relative material quantities are normalized and clipped:

$$
f_X(s)=\operatorname{clip}\left(\frac{\log(\max(X_{\mathrm{own}},1)/\max(X_{\mathrm{enemy}},1))}{\log4},-1,1\right),\qquad X\in\{L,A,P\}.
$$

The antisymmetric potential and shaped reward are

$$
\Phi(s)=0.3f_L(s)+0.3f_A(s)+0.4f_P(s),\qquad
r'_t=r_t^{\mathrm{win}}+\eta\left(\gamma\bar\Phi(s_{t+1})-\Phi(s_t)\right).
$$

Default shaping strength eta and discount gamma are both 1. Base reward is +1 for a win, -1 for a loss and zero otherwise. At true termination the next potential is zero. Time limits and rollout boundaries retain the continuation potential. For a complete terminated episode of length T, shaping telescopes:

$$
\sum_{t=0}^{T-1}\gamma^t(r'_t-r_t^{\mathrm{win}})=-\eta\Phi(s_0).
$$

Equal starting resources give zero initial potential. This provides intermediate feedback while preserving the complete-episode objective up to an initial-state constant. Finite rollouts, approximate value estimates and limited optimization still affect learning; the identity does not guarantee a particular PPO outcome.

The material weighting follows the 2507 paper. The ratio cap of four and counting generals alongside cities specify this implementation's normalization and zero-city handling. Actual hidden producer counts enter training rewards through the simulator and are excluded from policy inputs. There are no extra pass penalties, exploration bonuses or RND rewards.

## PPO and auxiliary learning

### Returns and advantages

Write d_t for true termination and c_t for time-limit truncation. V_old is evaluated using the behavior policy. At truncation, its successor value comes from the final observation before reset. At a rollout boundary, it comes from the continuation state. Generalized advantage estimation uses gamma = 1 and lambda = 0.95:

$$
\delta_t=r'_t+\gamma(1-d_t)V_{\mathrm{old}}(I_{t+1})-V_{\mathrm{old}}(I_t),
$$

$$
A_t=\delta_t+\gamma\lambda(1-d_t)(1-c_t)A_{t+1},\qquad
\widehat R_t=A_t+V_{\mathrm{old}}(I_t).
$$

Here I_t denotes the encoded observation and memory. The recursion stops at episode boundaries and at the end of the collected rollout. Returns use unnormalized advantages. Policy advantages are then centered and divided by their rollout-wide standard deviation, with denominator floor 1e-6.

### Policy, value and entropy losses

The likelihood ratio compares updated and behavior policies on the same input and sampled learner action:

$$
\rho_t(\theta)=\exp\left(\log\pi_\theta(a_t\mid I_t)-\log\pi_{\mathrm{old}}(a_t\mid I_t)\right).
$$

Using normalized advantages and clipping epsilon = 0.2, the minimized losses are

$$
\mathcal L_{\mathrm{policy}}=-\mathbb E_t\left[\min\left(\rho_tA_t^{\mathrm{norm}},\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t^{\mathrm{norm}}\right)\right],\qquad
\mathcal L_V=\mathbb E_t\left[(V_\theta(I_t)-\widehat R_t)^2\right].
$$

Entropy is the categorical entropy after legal masking. The logged approximate KL estimator is

$$
\widehat D_{\mathrm{KL}}=\mathbb E_t[(\rho_t-1)-\log\rho_t].
$$

KL is diagnostic and does not trigger early stopping. Value learning uses squared error without value clipping.

### Prediction losses

The simulator supplies labels at the pre-action state. Hidden ownership uses cross-entropy; hidden normalized army size uses Huber loss with threshold 1. They are averaged over currently hidden real cells, with denominator floor one. General-location cross-entropy uses the actual enemy general cell, including when visible:

$$
\mathcal L_{\mathrm{hidden}}=\mathbb E_t\left[
\frac{\sum_{u\in\mathcal H_t}\left(\operatorname{CE}(\widehat O_{t,u},O_{t,u})+\operatorname{Huber}(\widehat A_{t,u},\ell(A_{t,u}))\right)}{\max(|\mathcal H_t|,1)}
+\operatorname{CE}(\widehat G_t,G_t)
\right].
$$

If no enemy general exists, its term is zero. Hidden cells exclude padding. The army regression head is separate from the value head.

Opponent-action cross-entropy targets the frozen opponent's command. Opponent-view reconstruction averages binary cross-entropy over nine view channels and adds army Huber loss where the opponent can see:

$$
\mathcal L_{\mathrm{command}}=\mathbb E_t[-\log\widehat\pi_{\mathrm{opp}}(a_t^{\mathrm{opp}}\mid I_t)],
$$

$$
\mathcal L_{\mathrm{view}}=\mathbb E_t\left[
\frac{1}{|\mathcal V|}\sum_{u\in\mathcal V}\left(
\frac{1}{9}\sum_{k=0}^{8}\operatorname{BCE}(\widehat x_{t,u,k}^{\mathrm{opp}},x_{t,u,k}^{\mathrm{opp}})
+x_{t,u,0}^{\mathrm{opp}}\operatorname{Huber}(\widehat x_{t,u,9}^{\mathrm{opp}},x_{t,u,9}^{\mathrm{opp}})
\right)\right].
$$

The combined objective is

$$
\mathcal L=\mathcal L_{\mathrm{policy}}+0.5\mathcal L_V-0.01\mathcal H(\pi_\theta)+0.02\mathcal L_{\mathrm{hidden}}+0.02(\mathcal L_{\mathrm{command}}+\mathcal L_{\mathrm{view}}).
$$

Prediction supervision encourages the representation to retain information relevant to hidden state and opponent behavior. PPO learns how that representation should influence actions. Prediction accuracy alone does not establish playing strength. Training and deployed action selection use the same legal encoder inputs.

### Optimizer and sample counts

Each iteration shuffles all learner transitions and performs one epoch. The logical minibatch contains 1,024 transitions; gradients accumulate across 16 microbatches of 64 before one optimizer step. This produces 256 Adam updates per default iteration. The learning rate is constant at 1e-4. Accumulated gradients are clipped to global norm 0.5 before updating. Logged gradient norms are measured before clipping. Entropy and auxiliary coefficients remain constant unless explicitly configured at startup.

Across 100 iterations, the defaults process 26,214,400 learner transitions and perform 25,600 optimizer updates. These counts include repeated maps and states. There is no additional behavior-cloning phase, curriculum progression or automatic promotion based on evaluation.

## Evaluation and progress

Periodic evaluation tests the original supervised policy, this run's initialization and the algorithmic baseline. It uses separate maps, histories and random streams and leaves training weights, optimizer and opponent pool unchanged. Defaults are 128 games per opponent, with 16 parallel games, using 64 maps in both learner seats. Training-map overlap is checked by initial-state hashes.

Both neural policies sample their categorical distributions in default evaluation. Standalone `--greedy` evaluation selects argmax for both. Training opponents always use argmax.

```bash
python -B evaluate.py runs/ppo/policy.zip --baseline --output baseline-result.json
```

Win rate divides wins by all scheduled games, including unresolved time limits. Net win rate assigns +1 to wins, -1 to losses and zero to draws or unresolved games. Confidence intervals treat each map and its seat pair as one statistical unit. The Hoeffding intervals are conservative fixed-look intervals and do not certify repeated checkpoint selection on the same evaluation set.

Metrics also record mean game length, total pass fraction, pass fraction when movement was legal, opponent-action accuracy and loss, opponent-view binary loss, and both players' land/army/production at half-turns 50, 100, 150, 200 and 400. Material summaries exclude true terminal transfers and report contributing game counts. Different survival patterns can change these conditional averages.

Training losses and shaped rewards are diagnostics. Held-out opponent outcomes, unresolved-game frequency, game length and economy measurements provide the evidence for progress.

## Execution and memory

Inference and environment stepping run as batched JAX computations on the selected accelerator. Opponent inference groups environments by frozen policy slot, sharing one parameter set across each inference batch. The default inference batch is 64. Compiled collection advances in 16-step chunks before transferring packed learner transitions to CPU storage.

Binary channels are bit-packed; numeric features retain FP32 values. Recent history is reconstructed from one timeline per environment with reset masks. Only the requested minibatch is materialized and transferred. One host prefetch thread overlaps preparation with updates. Default persistent rollout storage occupies approximately 4.99 GiB of CPU RAM, plus transient batches and other process allocations.

GPU matrix products use BF16 by default. Parameters, optimizer state, normalization, logits, returns and losses remain FP32. CPU execution defaults to FP32. Model and optimizer buffers are donated to compiled updates. Progress synchronizes every 16 optimizer minibatches. The launcher enables a persistent JAX compilation cache and disables eager full-device preallocation.

Logs distinguish rollout, update and save times. Initial tracing and compilation contribute to first-iteration time. Full-size GPU throughput and peak memory require measurement on the training machine. Lowering `--microbatch` or `--inference-batch` reduces execution batch size; microbatch must divide the logical batch. These execution settings can change while statistical settings stay fixed.

## Saved state

Initialization saves a checkpoint and policy before collection. Every completed iteration updates them before evaluation and writes a separate policy snapshot:

```text
runs/ppo/
  checkpoint.zip
  policy.zip
  snapshots/iteration-00000000.zip
  snapshots/iteration-00000001.zip
  ...
  references/original.zip
  references/initial.zip
  training.jsonl
  evaluation.jsonl
  evaluation/iteration-00000010/original.json
  evaluation/iteration-00000010/initial.json
  evaluation/iteration-00000010/baseline.json
```

The policy archive contains parameters and metadata for inference or fresh initialization. The full checkpoint additionally contains Adam state, unfinished games, both players' observation memories, random streams and every active or retiring opponent slot. Maps are regenerated deterministically from the saved configuration and seed.

Each archive is written to a temporary file and atomically replaced; the archive set is saved sequentially. Evaluation records include policy and opponent hashes. An interrupted iteration can be rerun from the preceding full checkpoint. Immutable policy snapshots contain no optimizer or environment state.
