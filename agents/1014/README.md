# Agent 1014

Agent 1014 learns to choose one Generals command from the information available to a player. Its policy network, called the actor, converts observations and remembered information into a probability distribution over movements and pass. A graph encoder processes neighboring cells, and attention connects distant cells and historical observations. The actor receives the current view, persistent records, eight spatial/action frames and 512 public-statistics frames.

Training repeatedly collects games and updates the actor with proximal policy optimization (PPO). Each game pairs the learning actor with a frozen policy from an opponent pool. A separate action-value network, called the Q-critic, uses the simulator's complete state during training to estimate the returns associated with actions. Auxiliary prediction heads train the actor's shared features to describe hidden state, opponent commands and economic outcomes. A potential-based reward transformation supplies intermediate learning signals. Periodic matches determine which trained policies enter the opponent pool.

The sections below follow the computation from observations to actor outputs, auxiliary targets, critic estimates and parameter updates. The final sections describe opponent selection, evaluation, storage and execution.

This document specifies the supplied implementation. The actor has 3,611,474 parameters, including 35,474 parameters for economic prediction. The width-64 Q-critic has 339,040 parameters. Stored inputs and parameters use float32. Default GPU matrix products use bfloat16.

## Learnable relationships

The input paths and decoders support learning these relationships:

- Local force concentration, distribution and exchange opportunities from armies, ownership and directional neighbors.
- Route preferences, obstacle-sensitive movement and force aggregation from remembered terrain, transport edges and repeated graph updates.
- Coordination between distant regions from full-board attention, per-cell representations and learned global pooling.
- Phase-sensitive expansion, captures and movement from half-turn, growth phase, public material history and producer observations.
- Hidden territory and army estimates from visible facts, last-enemy records, public totals, observation ages and comparable changes.
- Enemy general localization from exploration, spatial relationships, persistent markers and the general-location objective.
- Exposure-sensitive decisions from current coverage on own territory and persistent confirmed-exposure records.
- Opponent next-command distributions from legal observations, recent commands, public history and shared opponent features.
- Source-target action preferences conditioned on local forces, terrain, remote context and full/half movement choice.
- Return-based action preferences from centralized Q values, trace targets and PPO updates.
- Action-conditioned economic trajectories from separate production, army-change and land-change targets at several future horizons.

These items describe relationships parameterized by the implementation. Each cell retains 128 features through eight residual blocks. Four blocks contain global attention. History paths encode ordered frames. Auxiliary state outputs describe per-cell ownership/army distributions and a board-wide general-location distribution. The action decoder reads shared features directly.

## Observations and memory

### Board and notation

Let $V$ be the set of real board cells, $n=|V|$ its size, and $N=HW$ the allocated grid area, with height $H$, width $W$ and padding around the real board. Cell indices $u,v$ run through the allocated grid. Player $i\in\{0,1\}$ observes opponent $j=1-i$. The half-turn index $t$ identifies an observation. The set $N_4(u)$ contains valid orthogonal neighbors, $N_8(u)$ contains valid orthogonal and diagonal neighbors, and $N_8[u]=N_8(u)\cup\{u\}$. The vector $p_u$ contains the row and column of cell $u$. Indicator brackets $[\cdot]$ evaluate to zero or one.

Vector brackets denote concatenation, $\odot$ denotes componentwise multiplication, and $\operatorname{vec}$ flattens an ordered collection. Array indices start at zero. A slice $a:b$ includes indices from $a$ through $b-1$. These conventions connect the formulas to the channel layouts in the code.

Direction indices 0,1,2,3 denote up, down, left, right. The four following offsets are diagonals. The graph constructor stores neighbors, offsets, positions and board-edge validity.

Army counts and observation ages can span several orders of magnitude. The encoder represents these nonnegative quantities with the logarithmic function $\ell$:

$$
\ell(a)=\frac{\log(1+\max(a,0))}{\log256}.
$$

For a nonnegative value, the real-arithmetic inverse is

$$
\ell^{-1}(y)=\exp(y\log256)-1.
$$

The encoder assembles four arrays. The current spatial array $X_t$ contains 28 features per cell. The spatial history $R_t$ contains $H_s=8$ frames with 13 features per cell. Short and long public histories $U_t^s,U_t^l$ contain eleven statistics per frame:

$$
X_t\in\mathbb R^{N\times28},\quad R_t\in\mathbb R^{H_s\times N\times13},\quad U_t^s\in\mathbb R^{H_s\times11},\quad U_t^l\in\mathbb R^{512\times11},\qquad H_s=8.
$$

The input also contains masks identifying real cells, recorded history frames, transport edges and legal actions. Together, these arrays and masks form the actor's encoded information $I_t$. The following subsections define their contents and update rules. Equations describe real arithmetic. Runtime float32 storage and optional bfloat16 matrix products introduce rounding.

### Current observation

The simulator provides ownership $O_t(u)$, army count $A_t(u)$, and mountain, city and general indicators $M(u),C(u),G(u)$. Ownership is $-1$ for neutral cells and the player index for owned cells. The observation adapter applies the visibility rule to these simulator fields before constructing actor inputs. Write $o_t(u)$ for own ownership, $v_t(u)$ for visibility and $e_t(u)$ for visible enemy ownership:

$$
o_t(u)=[O_t(u)=i],\qquad v_t(u)=\left[\sum_{w\in N_8[u]}o_t(w)>0\right],\qquad e_t(u)=v_t(u)[O_t(u)=j].
$$

Owning a cell reveals that cell and its eight neighbors. Using the resulting visibility mask, the adapter constructs the ten-component current observation $x_t(u)$:

$$
x_t(u)=\left[v_t,o_t,e_t,v_t[O_t=-1](1-M),v_tM,v_tC,v_tG,(1-v_t)(1-M)(1-C),(1-v_t)[M\lor C],v_t\ell(A_t)\right]_u.
$$

They encode visibility, own ownership, enemy ownership, neutral traversable ground, mountain, city, general, ordinary fog, obstacle fog and log army. Obstacle fog reveals the union of mountain and city membership. Visibility distinguishes hidden army channel zero from a visible zero-army cell.

The game also publishes each player's total armies and land count. Write these totals as $A_i,A_j$ and $L_i,L_j$. Three configuration indicators describe city regeneration, the older settlement order and general trading, denoted by $r_{\rm regen},r_{\rm old},r_{\rm trade}$. Producing cells grow every two half-turns, and owned land grows every fifty half-turns. The eleven-component public frame $U_t$ combines these quantities with elapsed time, both growth phases and the observing player's index:

$$
U_t=\left[t/512,(t\bmod50)/50,t\bmod2,\ell(A_i),\ell(A_j),L_i/n,L_j/n,i,r_{\rm regen},r_{\rm old},r_{\rm trade}\right].
$$

The runtime supplies rule flags (0,0,1). Public totals cover each player's complete territory. Spatial facts use the masked view.

### Last-seen records

A cell's last visible observation supplies a persistent record after it leaves view. Let $f_t(u)$ select the seven current channels for own, enemy, neutral, mountain, city, general and log army. Let $L_t(u)$ store that record and $T_t(u)$ store its observation time. The vector-valued $L_t(u)$ is distinct from a player's scalar land count $L_i$. The record and timestamp update together:

$$
L_t(u)=\begin{cases}f_t(u)&v_t(u)=1,\\L_{t-1}(u)&v_t(u)=0,\end{cases}\qquad T_t(u)=\begin{cases}t&v_t(u)=1,\\T_{t-1}(u)&v_t(u)=0.\end{cases}
$$

The record initializes to $L=0$ and its timestamp to $T=-1$. From these fields, the encoder derives the ever-seen indicator $S_t(u)$, encoded observation age $D_t(u)$, and known-mountain indicator $K_t(u)$:

$$
S_t(u)=[T_t(u)\ge0],\qquad D_t(u)=S_t(u)\ell(t-T_t(u)),\qquad K_t(u)=S_t(u)L_t(u)_{\rm mountain}.
$$

Static terrain and general markers persist through fog in $L_t$. A new visible observation refreshes every component. The stored ownership and army values accompany their observation age, allowing the encoder to represent both the recorded fact and its age.

A later friendly or neutral observation replaces the last-seen record. To retain the earlier enemy sighting as well, the encoder maintains a separate enemy timestamp $J_t(u)$ and log army $B_t(u)$. These fields update whenever the cell is visibly enemy-owned:

$$
J_t(u)=\begin{cases}t&e_t(u)=1,\\J_{t-1}(u)&e_t(u)=0,\end{cases}\qquad B_t(u)=\begin{cases}\ell(A_t(u))&e_t(u)=1,\\B_{t-1}(u)&e_t(u)=0.\end{cases}
$$

These enemy-specific fields initialize to $J=-1$ and $B=0$. They retain an earlier enemy sighting even as later observations update the general last-seen record $L_t$.

### Coverage and exposure

Each owned cell contributes one sight source to its surrounding nine-cell region. Counting these sources expresses how many cells currently reveal a location. Own coverage $C_t^{\rm own}$ counts all own sources, and confirmed enemy coverage $\underline C_t^{\rm enemy}$ counts visible enemy sources:

$$
C_t^{\rm own}(u)=\sum_{w\in N_8[u]}o_t(w),\qquad \underline C_t^{\rm enemy}(u)=\sum_{w\in N_8[u]}e_t(w).
$$

Own coverage is exact. Confirmed enemy coverage is exact on current own territory because every adjacent enemy sight source is visible there. Elsewhere it provides a lower bound.

Confirmed enemy coverage also establishes that the opponent has seen the location. The encoder retains this fact in a bit $E_t(u)$, which becomes one at the first confirmed exposure and remains one for the game:

$$
E_t(u)=E_{t-1}(u)\lor[\underline C_t^{\rm enemy}(u)>0],\qquad E_{-1}(u)=0.
$$

To relate this record to the underlying rule, let $v_t^{\rm enemy}(u)$ denote the opponent's actual visibility. For a complete ownership realization, at least one enemy sight source in $N_8[u]$ makes the location visible:

$$
v_t^{\rm enemy}(u)=1-\prod_{w\in N_8[u]}(1-[O_t(w)=j]).
$$

With the actor's encoded information $I_t$, the corresponding probability depends on the joint ownership of those neighboring cells:

$$
P(v_t^{\rm enemy}(u)=1\mid I_t)=1-P\left(\bigcap_{w\in N_8[u]}\{O_t(w)\ne j\}\mid I_t\right).
$$

This identity explains the relationship between visibility and hidden ownership. The implemented input channels contain confirmed coverage and the historical exposure bit $E_t$. The last-seen and last-enemy records separately retain observation times and observed contents.

### Spatial channel layout

The encoder combines the current view and persistent records into $X_t$. Channels 0-9 contain $x_t$, and channels 10-16 contain $L_t$. Let $x_t^{\rm obstacle}=x_{t,8}$ denote obstacle fog. The eleven remaining channels, at indices 17-27, are:

$$
X_t(u)_{17:28}=\left[S_t,D_t,C_t^{\rm own}/9,\underline C_t^{\rm enemy}/9,E_t,o_t,K_t,x_t^{\rm obstacle}(1-S_t),[J_t\ge0],B_t,[J_t\ge0]\ell(t-J_t)\right]_u.
$$

Every channel is multiplied by the board mask. The second own-ownership channel preserves the migrated input mapping.

### Action and observation windows

Each movement command selects a source cell, one of four orthogonal directions, and either full or half movement. Let $d\in\{0,1,2,3\}$ be the direction and $b$ the movement flag, with zero for full and one for half. The flattened movement index $a$ and the separate pass index are:

$$
a=8u+2d+b,\qquad b\in\{0,1\},\qquad a_{\rm pass}=8N.
$$

The history records the command that preceded each observation. Three per-cell markers $s_t,d_t,h_t$ identify its source, its destination, and its source multiplied by the half-movement flag. Pass produces three zero markers. Appending these markers to the ten observation channels produces the thirteen-component frame $F_t$:

$$
F_t(u)=[x_t(u),s_t(u),d_t(u),h_t(u)].
$$

The operator $\Vert$ appends a frame, and $\operatorname{tail}_k$ retains the latest $k$ frames. The spatial and public windows therefore update as follows:

$$
R_t=\operatorname{tail}_{H_s}(R_{t-1}\Vert F_t),\quad U_t^s=\operatorname{tail}_{H_s}(U_{t-1}^s\Vert U_t),\quad U_t^l=\operatorname{tail}_{512}(U_{t-1}^l\Vert U_t).
$$

Appended frames are valid. Reset initializes windows to zero, validity to false and timestamps to -1. Persistent records survive sliding-window updates within a game.

### Comparable army changes

The spatial-history encoder also receives army changes between comparable observations. Here $k$ indexes slots within the eight-frame window. Let $m_k$ indicate a recorded frame, $\tau_k$ its half-turn timestamp, and $v_{k,u}$ the visibility of cell $u$. A comparison is valid when consecutive half-turns contain visible observations of the cell:

$$
c_{k,u}=m_km_{k-1}[\tau_k=\tau_{k-1}+1]v_{k,u}v_{k-1,u}.
$$

Let $o^p_{k,u}$ indicate own or visible enemy ownership according to $p\in\{\mathrm{own},\mathrm{enemy}\}$. Multiplying both frames' ownership indicators selects changes within the same ownership category. Applying the inverse log encoding to army channel 9 gives the signed change, scaled by 256:

$$
\Delta^p_{k,u}=c_{k,u}o^p_{k,u}o^p_{k-1,u}\frac{\ell^{-1}(x_{k,u,9})-\ell^{-1}(x_{k-1,u,9})}{256}.
$$

First-frame changes and comparability are zero. Visibility entry/exit, ownership changes and skipped half-turns produce zero derived changes. Their original observations remain in F. Comparable differences include growth, transfers and exchanges. Phase and timestamp inputs accompany these changes.

### Transport and action masks

The graph encoder represents possible local transport with a binary edge mask $M^{\rm move}$. Orthogonal neighbors remain connected when both endpoints pass the known-mountain check:

$$
M^{\rm move}_{uv}=[v\in N_4(u)](1-K_t(u))(1-K_t(v)).
$$

Unknown connections remain in this latent relation until terrain is identified. For command selection, the legal-action mask $\mathcal L$ additionally requires an owned source with at least two armies. The function $\operatorname{neighbor}(u,d)$ selects the destination in direction $d$:

$$
\mathcal L(8u+2d+b)=o_t(u)[x_t(u)_9\ge\ell(2)-10^{-6}]M^{\rm move}_{u,\operatorname{neighbor}(u,d)},\qquad \mathcal L(8N)=1.
$$

Neighbors of an owned source are visible. Full and half moves share the mask. Board validity handles padding.

## Actor network

The actor first encodes current observations and history into one 128-component vector per cell. Eight residual blocks refine those vectors through local graph messages and global attention. A final shared representation supplies the policy and auxiliary prediction heads. This section follows that sequence.

In the network equations, each $W$ denotes a learned matrix, each $b$ a learned bias, and other learned vectors are identified at their first use. Subscripts distinguish separate layers, and each block has its own parameters. Affine maps include biases where displayed. The activation functions are GELU and the sigmoid $\sigma$. GELU uses JAX's default tanh approximation:

$$
\operatorname{GELU}(x)=\frac{x}{2}\left[1+\tanh\left(\sqrt{2/\pi}(x+0.044715x^3)\right)\right],\qquad \sigma(x)=\frac1{1+\exp(-x)}.
$$

Layer normalization, written $\operatorname{LN}$, normalizes a vector across its feature channels. For a vector $x$ with $d$ channels, let $\mu$ and $s^2$ be its channel mean and variance. Equinox LayerNorm applies learned scale $a$ and bias $b$:

$$
\mu=\frac1d\sum_{k=1}^d x_k,\qquad s^2=\frac1d\sum_{k=1}^d(x_k-\mu)^2,\qquad \operatorname{LN}(x)_k=a_k\frac{x_k-\mu}{\sqrt{s^2+10^{-5}}}+b_k.
$$

Attention weights and action probabilities use softmax. Its logarithm provides the likelihoods used in training:

$$
\operatorname{softmax}(x)_k=\frac{\exp(x_k)}{\sum_j\exp(x_j)},\qquad \operatorname{logsoftmax}(x)_k=x_k-\log\sum_j\exp(x_j).
$$

Runtime evaluation uses numerically stable forms. Masked positions receive the finite value -1e9.

### Temporal embeddings

The first history path learns which recent frames to emphasize. Two independently parameterized temporal encoders process the thirteen-component per-cell frames and the eleven-component short public frames. Within either encoder, $p_k^{\rm slot}$ is a learned vector identifying history slot $k$, and $q^{\rm time}$ is a learned query. The encoder projects each frame to $b_k$, assigns a normalized weight $\alpha_k$ over valid slots, and returns their weighted sum $T(F,m)$:

$$
b_k=\operatorname{GELU}(W_TF_k+b_T+p_k^{\rm slot}),\qquad \alpha_k=\operatorname{softmax}_{k:m_k=1}((q^{\rm time})^\top b_k),\qquad T(F,m)=\sum_k\alpha_kb_k.
$$

Both temporal encoders have output width 128. The cell encoder applies the formula to frames $F_k(u)$, and the public encoder applies it to public frames $U_k$. Their parameters are independent.

### Current and recent context

The current spatial embedding and both temporal embeddings each have 128 components. The public embedding is shared across cells, while the current and spatial-history embeddings vary by cell. A 384-to-128 affine map combines them into the initial cell representation $h_u^0$:

$$
h_u^0=W_I[W_XX_t(u)+b_X,T_{\rm cell}(R_t(:,u),m),T_{\rm public}(U_t^s,m)]+b_I.
$$

Two additional history paths enrich this representation. The spatial-history path below preserves ordered local changes. The public-history path then lets each cell retrieve context from the longer statistics window.

### Spatial-history branch

The temporal weighted sum summarizes recent frames at one cell. A second path explicitly combines each historical cell with its neighbors. It starts by appending the two comparable army changes and their validity bit to the thirteen-component frame. A learned projection converts these sixteen inputs into a 32-component vector $s_{k,u}$:

$$
s_{k,u}=\operatorname{GELU}(W_f[F_{k,u},\Delta^{\rm own}_{k,u},\Delta^{\rm enemy}_{k,u},c_{k,u}]+b_f)\in\mathbb R^{32}.
$$

For the same history slot, the encoder aggregates the four- and eight-neighbor features alongside the cell's own features. The result $r_{k,u}$ retains 32 components and is multiplied by frame validity $m_k$:

$$
r_{k,u}=m_k\operatorname{GELU}\left(W_n\left[s_{k,u},\frac14\sum_{v\in N_4(u)}s_{k,v},\frac18\sum_{v\in N_8(u)}s_{k,v}\right]+b_n\right).
$$

Board-valid neighbors participate. The divisors remain four and eight at boundaries. Concatenating the eight resulting vectors in time order preserves their slot identities. A learned projection produces the 128-component spatial-history contribution $S_u^{\rm hist}$:

$$
S_u^{\rm hist}=W_s\operatorname{vec}_{k=1}^{H_s}(r_{k,u})+b_s\in\mathbb R^{128}.
$$

At H_s=8 the projection has 256 inputs. Direction-sensitive graph updates follow this historical aggregation.

### Public-history branch

The long public history captures changes over a broader time range. The encoder divides its 512 frames into eight ordered segments of 64 frames. In the segment formula, $U$ denotes this long history and $m$ its frame-validity vector. Each frame contributes eleven public values multiplied by its validity bit, followed by that validity bit itself. Each segment therefore has $64\times12=768$ inputs. Two affine maps and a learned segment-position vector $p_j^c$ produce a 128-component segment token $c_j$, where $j$ indexes the eight segments:

$$
c_j=W_2^c\operatorname{GELU}(W_1^c\operatorname{vec}[mU,m]_j+b_1^c)+b_2^c+p_j^c\in\mathbb R^{128}.
$$

Each cell uses its already defined representation $h_u^0$ to query these segment tokens. A segment participates when any of its frames is valid. For each of four attention heads $a$, learned projections produce a 32-component cell query, segment key and segment value. The weights $\alpha_{u,j}^a$ measure that cell's attention to each segment. Concatenating the four weighted sums gives the retrieved public context $C_u^{\rm hist}$:

$$
\alpha_{u,j}^a=\operatorname{softmax}_{j\text{ valid}}\left(\frac{(W_Q^ah_u^0+b_Q^a)^\top(W_K^ac_j+b_K^a)}{\sqrt{32}}\right),\qquad C_u^{\rm hist}=\operatorname{concat}_{a=1}^{4}\sum_j\alpha_{u,j}^a(W_V^ac_j+b_V^a).
$$

### Combined history representation

Both additional history paths now have the same width as $h_u^0$. The encoder adds the spatial contribution and an affine projection of the retrieved public context to form $\widetilde h_u^0$, the input to the graph blocks:

$$
\widetilde h_u^0=h_u^0+S_u^{\rm hist}+W_OC_u^{\rm hist}+b_O.
$$

Padding is zeroed before and after the history branch. Runtime observations supply at least one valid current frame.

### Graph blocks

Eight width-128 blocks update the combined representation. Each block contains local messages and a 128-to-512-to-128 feed-forward residual. Blocks 2,4,6,8 also contain global attention. In the equations for one block, $h_u$ denotes its input. The first block receives $\widetilde h_u^0$, and each later block receives the preceding block's output.

Local messages use two relations indexed by $r\in\{4,8\}$. Normalize the input as $z_u^{\rm local}=\operatorname{LN}(h_u)$. For a source-destination pair, the edge feature $f_{uv}$ combines both endpoints, their componentwise interaction and their coordinate offset. Separate affine maps produce a message $m_{uv}^r$ and a componentwise gate $g_{uv}^r$:

$$
f_{uv}=[z_u^{\rm local},z_v^{\rm local},z_u^{\rm local}\odot z_v^{\rm local},p_v-p_u],\qquad m_{uv}^r=\operatorname{GELU}(W_m^rf_{uv}+b_m^r),\qquad g_{uv}^r=\sigma(W_g^rf_{uv}+b_g^r).
$$

Let $M_{uv}^4=M_{uv}^{\rm move}$, the known-terrain transport mask, and let $M_{uv}^8$ identify valid geometric eight-neighbor edges. Each relation sums its gated messages into $b_u^r$. A projection of both sums gives the residual update $h_u^L$:

$$
b_u^r=\frac1r\sum_{v\in N_r(u)}M_{uv}^rg_{uv}^r\odot m_{uv}^r,\qquad h_u^L=h_u+W_M[b_u^4,b_u^8]+b_M.
$$

### Global attention

The attention-enabled blocks next connect each cell to every real cell. For each head $a$, separate affine projections of $\operatorname{LN}(h_u^L)$ produce the 32-component query $Q_u^a$, key $K_u^a$ and value $V_u^a$. These attention vectors have head superscripts to distinguish them from cell sets and value estimates elsewhere in the document.

A learned geometry term augments the query-key score. Define the transformed displacement $d_{uv}$ and its three-component geometry feature $\xi_{uv}$:

$$
d_{uv}=\frac{\operatorname{sign}(p_u-p_v)\odot\log(1+|p_u-p_v|)}4,\qquad \xi_{uv}=[d_{uv,x},d_{uv,y},|d_{uv,x}|+|d_{uv,y}|].
$$

The coordinates $x,y$ here select the two displacement components. Each head has a learned three-component vector $w_a$. Softmax turns the content and geometry scores into attention weights $\beta_{uv}^a$, and the weighted values produce the residual output $h_u^A$:

$$
\beta_{uv}^a=\operatorname{softmax}_{v\in V}\left(\frac{Q_u^{a\top}K_v^a}{\sqrt{32}}+w_a^\top\xi_{uv}\right),\qquad h_u^A=h_u^L+W_A\operatorname{concat}_a\sum_v\beta_{uv}^aV_v^a+b_A.
$$

Local-only blocks carry the local result forward as $h_u^A=h_u^L$. Every block then applies a channel-mixing feed-forward layer: normalization, a projection to 512 components, GELU, and a projection back to 128 components. Adding this result to the block's current features gives its output $h_u^{\rm next}$:

$$
h_u^{\rm next}=h_u^A+W_2\operatorname{GELU}(W_1\operatorname{LN}(h_u^A)+b_1)+b_2.
$$

Padding is zeroed after each block. Full-board attention connects all real cells. The bias uses transformed coordinate displacement and its L1 magnitude. The board retains N separate 128-component vectors.

### Production evidence

The actor also receives a compact accounting statistic computed from public army changes. Let $P_i^{\rm obs}$ count observed own producing cells, where a producing cell is a city or general, and let $\underline P_j^{\rm obs}$ count currently visible enemy producing cells. The own count is complete because own territory is visible. The enemy count is a visible lower bound. Using these counts and consecutive public totals, define the production residual $q_t^{\rm prod}$:

$$
q_t^{\rm prod}=P_i^{\rm obs}+[t\bmod50=0](L_i-L_j)-\left((A_i-A_j)_t-(A_i-A_j)_{t-1}\right).
$$

An even half-turn is a producer-growth step. Let $\nu$ indicate two valid consecutive observations ending on such a step. The three-component evidence vector $e_t^{\rm prod}$ contains the normalized residual, its validity indicator and the visible enemy producer count:

$$
e_t^{\rm prod}=[\nu q_t^{\rm prod}/n,\nu,\underline P_j^{\rm obs}/n].
$$

The residual has an accounting interpretation under matching settlement counts and equal attrition in opposing exchanges. Let $P_j$ be the actual enemy producer count, and $N_i,N_j$ be the two players' losses to neutral captures during the transition. Under those conditions:

$$
q_t^{\rm prod}=P_j+N_i-N_j.
$$

The network receives the arithmetic statistic and validity flag. Capture ordering, neutral losses and general transfers determine its interpretation in a particular transition.

### Shared features and pooling

Let $h_u^8$ denote the output of the eighth graph block. Final normalization produces $f_u$. The production branch then joins each cell's features with $e_t^{\rm prod}$, producing an intermediate vector $z_u$ and the final shared representation $\bar h_u$:

$$
f_u=\operatorname{LN}_{\rm final}(h_u^8),\qquad z_u=\operatorname{GELU}(W_B[f_u,e_t^{\rm prod}]+b_B),\qquad \bar h_u=f_u+W_Rz_u+b_R.
$$

The first map has 131 inputs and 128 outputs. The residual map has 128 inputs and outputs. To give action decoding a board-level summary, a learned vector $w_g$ and scalar bias $b_g$ score each real cell. Softmax converts these scores into pooling weights $\omega_u$. The global context $g$ is the resulting weighted sum:

$$
\omega_u=\operatorname{softmax}_{u\in V}(w_g^\top\bar h_u+b_g),\qquad g=\sum_u\omega_u\bar h_u.
$$

The common bias $b_g$ cancels in the softmax. Individual cell representations remain available beside $g$, so action decoding can combine local features with this learned global summary.

### Policy and actor value

The policy scores full and half movement for each orthogonal source-target pair. Its input joins four 128-component vectors and the two coordinate offsets, giving 514 components. A hidden layer produces the 128-component movement feature $v_{uv}^{\rm move}$, and a second affine map produces two action scores, called logits:

$$
v_{uv}^{\rm move}=\operatorname{GELU}(W_{\pi,1}[\bar h_u,\bar h_v,\bar h_u\odot\bar h_v,g,p_v-p_u]+b_{\pi,1}),\qquad \ell_{u,v,:}=W_{\pi,2}v_{uv}^{\rm move}+b_{\pi,2}.
$$

The two logits correspond to full and half movement in that order. A learned vector $w_p$ and bias $b_p$ map $g$ to the pass logit. Flattening movement logits in the defined action order and appending pass gives the vector $\ell$. Let $\theta$ denote all actor parameters. Applying the legal-action mask and softmax yields the policy $\pi_\theta$. A separate readout with learned vector $w_V$ and bias $b_V$ produces the scalar $V_{\rm actor}$, which will receive a return-prediction target during training:

$$
\ell_{\rm pass}=w_p^\top g+b_p,\qquad \pi_\theta(a\mid I_t)=\operatorname{softmax}_a(\operatorname{mask}_{\mathcal L}(\ell))_a,\qquad V_{\rm actor}=w_V^\top g+b_V.
$$

Invalid action logits receive -1e9. There are 8N+1 actions. The learner samples categorically and frozen opponents select argmax.

## Auxiliary predictions

The policy and the auxiliary heads read features produced by the same actor encoder. Auxiliary heads predict quantities for which the simulator can supply training labels: hidden state, the opponent's command and subsequent economic outcomes. Their prediction losses update the shared representation along with their own decoder parameters. At action selection time, the policy reads the shared features and global context directly. The auxiliary outputs serve as training readouts.

The following subsections define each target before its decoder and loss. The actor's scalar value readout receives its target in the Q-trace section, where future rewards are combined into a return estimate. The actor-objective section then combines all these losses.

### State targets

The state head predicts three related quantities. For each cell it predicts ownership and a conditional distribution over army counts. Across the board it predicts the enemy general's location. This gives the shared features supervision for both local hidden resources and a location that determines victory.

Ownership labels are relative to the observing player. Define $C_u^O=0$ for neutral, $C_u^O=1$ for own, and $C_u^O=2$ for enemy ownership. These three class indices are obtained from simulator ownership $O_u$ by:

$$
C_u^O=[O_u=i]+2[O_u=j].
$$

For army prediction, the head assigns probabilities to 64 count intervals. Their class indices run from 0 to 63 because indexing starts at zero. The next subsection defines those intervals. Write $C_u^A$ for the resulting army class and $G_*\in V$ for the enemy general's cell.

### Army bins

The bins preserve exact small counts and use progressively wider intervals for larger armies. Sixty-three ascending integer boundaries separate the 64 bins. Denote these boundaries by $b_1,\ldots,b_{63}$. The first sixteen boundaries are the integers 1 through 16, and the remaining 47 follow a rounded geometric sequence from 17 to 65,536:

$$
(b_1,\ldots,b_{63})=(1,\ldots,16)\Vert\left(\operatorname{round}\left(17(65536/17)^{k/46}\right)\right)_{k=0}^{46}.
$$

For the stored army label $y=\ell(A_u)$, the class index counts how many boundaries the army has crossed. The implementation places each comparison halfway between neighboring integer army counts to preserve assignments under floating-point log encoding:

$$
C_u^A=\sum_{k=1}^{63}\left[y\ge\frac{\log(1+b_k-0.5)}{\log256}\right].
$$

Counts 0 through 16 have individual classes. Larger classes widen geometrically. Class 63 contains counts at least 65,536. Half-integer thresholds stabilize integer assignment under floating-point log encoding. Actor visible-army inputs retain their original logarithmic values.

### State decoder

The state decoder reads the final cell representation $\bar h_u$. Each cell needs one score for general location, three ownership scores and three groups of 64 army scores, one group for each possible owner. This gives $1+3+3\times64=196$ outputs per cell. A single affine map produces the score vector $s_u$:

$$
s_u=W_{\rm state}\bar h_u+b_{\rm state}\in\mathbb R^{196}.
$$

Under the zero-based slice convention, component 0 is the general-location score, slice 1:4 contains ownership scores, and slice 4:196 contains conditional army scores. Reshaping the latter into three rows of 64 entries assigns a row to each owner. Softmax then produces an ownership distribution and an army distribution conditional on each ownership class:

$$
p_\theta(C_u^O=o\mid I_t)=\operatorname{softmax}(s_u[1:4])_o,\qquad p_\theta(C_u^A=b\mid C_u^O=o,I_t)=\operatorname{softmax}(\operatorname{reshape}(s_u[4:196],3,64)_{o,:})_b.
$$

Here $o\in\{0,1,2\}$ and $b\in\{0,\ldots,63\}$ index the classes just defined. Multiplying the ownership probability by its conditional army probability gives a joint distribution within the cell. Summing over ownership gives the marginal army distribution:

$$
p_\theta(C_u^O=o,C_u^A=b\mid I_t)=p_\theta(C_u^O=o\mid I_t)p_\theta(C_u^A=b\mid C_u^O=o,I_t),
$$

$$
p_\theta(C_u^A=b\mid I_t)=\sum_{o=0}^{2}p_\theta(C_u^O=o\mid I_t)p_\theta(C_u^A=b\mid C_u^O=o,I_t).
$$

The general-location score has a different normalization axis. Each cell contributes one score, and softmax across all real cells produces one board-wide distribution:

$$
p_\theta(G_*=u\mid I_t)=\frac{\exp(s_u[0])}{\sum_{v\in V}\exp(s_v[0])}.
$$

Thus ownership and army predictions describe alternatives within each cell, while general prediction describes alternative locations across the board. All three readouts use cell features that already incorporate neighboring cells, global attention and history.

### State likelihoods

The ownership and army objectives focus supervision on hidden cells whose terrain is still eligible for inference. Define the eligible set $\mathcal U_t$ using visibility $v_t$ and remembered mountain status $K_t$:

$$
\mathcal U_t=\{u\in V:v_t(u)=0\ \land\ K_t(u)=0\}.
$$

For each eligible cell, the simulator supplies the ownership label $C_u^O$ and army class $C_u^A$. The ownership loss $L_O$ scores the probability assigned to the true owner. The army loss $L_B$ scores the probability assigned to the true army class in that owner's conditional distribution. Both losses average over eligible cells:

$$
L_O=\frac{\sum_{u\in\mathcal U_t}-\log p_\theta(C_u^O\mid I_t)}{\max(|\mathcal U_t|,1)},\qquad L_B=\frac{\sum_{u\in\mathcal U_t}-\log p_\theta(C_u^A\mid C_u^O,I_t)}{\max(|\mathcal U_t|,1)}.
$$

All three conditional army distributions are computed from actor features. The true owner selects the distribution that contributes to the army loss for that sample. Empty eligible sets contribute zero.

Let $g_*$ be the simulator's enemy-general cell label, with $-1$ marking an unavailable label. The location loss $L_G$ scores the board-wide probability assigned to that cell:

$$
L_G=\begin{cases}-\log p_\theta(G_*=g_*\mid I_t)&g_*\ge0,\\0&g_*=-1.\end{cases}
$$

The location distribution covers every real cell, including visible cells. The shared inputs and labels provide the observed evidence for localization. Adding the three terms gives the state-prediction loss $L_{\rm hidden}$:

$$
L_{\rm hidden}=L_O+L_B+L_G.
$$

### Opponent command

The opponent-command task asks what command the frozen opponent selects at the current half-turn. Its target $a_t^-$ comes from running that opponent on its own observation history. The learner predicts this command from its own encoded information $I_t$.

The decoder reads the production-branch features $z_u$ defined in the shared-features section. It scores each source-target pair using both endpoints, their product and their offset. The actor's pooling weights $\omega_u$ also summarize these features into $g^-$. A learned vector $w_-$ and scalar bias $b_-^p$ map that context to the pass score. Superscript minus identifies opponent-command quantities:

$$
\ell^-_{u,v,:}=W_-[z_u,z_v,z_u\odot z_v,p_v-p_u]+b_-,\qquad g^-=\sum_u\omega_uz_u,\qquad \ell^-_{\rm pass}=w_-^\top g^-+b_-^p.
$$

The command space uses the same full, half and pass indexing as the policy. Its mask includes all valid on-board orthogonal pairs, allowing the predictor to assign probability to commands from hidden enemy cells. Softmax produces the command distribution $\widehat\pi^-$. The supervised loss $L_{\rm opponent}$ is the negative log probability of the recorded command $a_t^-$:

$$
\widehat\pi^-(a\mid I_t)=\operatorname{softmax}_a(\operatorname{mask}_{\rm geometry}(\ell^-))_a,\qquad L_{\rm opponent}=-\log\widehat\pi^-(a_t^-\mid I_t).
$$

### Economic trajectory prediction

The economic head learns how production, surviving armies and territory change after a chosen action. The head shares the actor's board representation and movement features. Its supervised targets come from subsequent simulated play. The actor's victory objective trains those same shared features, connecting economic consequences with return-based action selection. Production and consumption remain separate target components, allowing the shared representation to describe both expansion and costly attacks.

For the learning player $i$ and opponent $j$, write $A_p(t),L_p(t),P_p(t)$ for player $p$'s total armies, owned cells and owned producing cells at time $t$. Producing cells are cities or generals. The game applies movement before income. For a transition ending at half-turn $t+1$, the generated army count $b_p(t)$ is therefore computed from the post-move producer and land counts:

$$
b_p(t)=[(t+1)\bmod2=0]P_p(t+1)+[(t+1)\bmod50=0]L_p(t+1).
$$

Let $d_t$ indicate a true game-ending transition, such as a general capture. A time-limit truncation ends data collection for a game and is handled separately below. Define the six-component economic flow $f_t^{\rm econ}$ in this fixed order: learner production, opponent production, learner army change, opponent army change, learner land change and opponent land change.

$$
f_t^{\rm econ}=(1-d_t)\left[b_i(t),b_j(t),A_i(t+1)-A_i(t),A_j(t+1)-A_j(t),L_i(t+1)-L_i(t),L_j(t+1)-L_j(t)\right].
$$

The decisive transition contributes zero in all channels. This convention keeps the final territory confiscation outside the economic accounting window. Earlier production, exchanges and captures retain their realized values. For ordinary transitions, production minus army change measures net army depletion. That depletion includes fighting and neutral captures. General transfers also affect the accounting. The six separate targets preserve these distinctions for the shared network to interpret using its spatial history.

The head predicts these flows over three horizons, covering 25, 50 and 200 half-turns. Let $h\in\{25,50,200\}$ denote a prediction horizon and $\tau_t$ the first true-terminal transition at or after $t$, taking $\tau_t=+\infty$ while such a transition is absent. A completed trajectory has zero economic flow after its true terminal. The six-component window total $S_{t,h}^{\rm econ}$ and its signed logarithmic encoding $y_{t,h}$ are:

$$
S_{t,h}^{\rm econ}=\sum_{k=t}^{\min(t+h-1,\tau_t)}f_k^{\rm econ},\qquad
y_{t,h}=\operatorname{sign}(S_{t,h}^{\rm econ})\frac{\log(1+|S_{t,h}^{\rm econ}|)}{\log256}.
$$

The encoding acts componentwise. Positive and negative changes retain their signs. Each target describes the continuation generated by the rollout learner and its selected frozen opponent. The model learns a robust regression of these encoded outcomes conditioned on the learner's available information and selected command.

A training label requires a complete window total. Let $m_{t,h}^{\rm econ}$ indicate that this total is available. A window is valid when all $h$ transitions are recorded or a true terminal completes its remaining future through zero flows. At a time limit, the final observed transition remains usable. Windows extending beyond that time limit or the available rollout receive validity zero. These masks separate complete economic outcomes from partially observed futures. Host-side prefix sums and first-boundary indices construct all three horizons from one flow timeline.

The decoder conditions its prediction on the selected learner command $a_t$. It reuses global context $g$ and the movement feature $v_{uv}^{\rm move}$ defined by the policy decoder. For a move, let $e(a_t)=v_{uv}^{\rm move}$, and let $q(a_t)$ be the two-component one-hot vector for full or half movement. Pass uses zero vectors for both. Their concatenation with $g$ has $128+128+2=258$ components. Two affine maps produce the prediction $\widehat y_t$. Its eighteen outputs form three rows for horizons 25, 50 and 200, each containing the six quantities in the flow's defined order:

$$
\widehat y_t=\operatorname{reshape}_{3\times6}\left(W_2^E\operatorname{GELU}\left(W_1^E[g,e(a_t),q(a_t)]+b_1^E\right)+b_2^E\right).
$$

The layer dimensions are 258-to-128-to-18. The decoder evaluates the selected action during training. Standard action selection uses the existing policy path. Auxiliary gradients reach the shared board encoder, learned pooling and movement feature layer. The final policy-logit layer continues to receive its policy and entropy gradients.

The loss compares predictions with the signed-log targets. For an encoded prediction error $z$, define the unit-threshold Huber penalty $\rho$. This penalty is quadratic near zero and linear for larger absolute errors. For sample $t$, the economic loss $L_{{\rm econ},t}$ averages the six component penalties within each horizon and then averages over valid horizons. Here $c\in\{1,\ldots,6\}$ labels the six target components, and the horizon subscript selects its corresponding prediction row:

$$
\rho(z)=\begin{cases}\tfrac12z^2&|z|\le1,\\|z|-\tfrac12&|z|>1,\end{cases}\qquad
L_{{\rm econ},t}=\frac{\sum_{h\in\{25,50,200\}}m_{t,h}^{\rm econ}\,\frac16\sum_{c=1}^{6}\rho(\widehat y_{t,h,c}-y_{t,h,c})}{\max(\sum_hm_{t,h}^{\rm econ},1)}.
$$

Samples with zero valid horizons contribute zero. The actor objective adds 0.02 times the sample mean of this loss. The option `--economy-loss` controls that coefficient. The reward, potential function and Q-trace targets retain their existing definitions.

Training logs include `economy_prediction_loss`, `economy_valid_fraction` for each collected horizon and `economy_prediction_by_horizon` for the optimized samples. Each horizon diagnostic reports its Huber error over valid labels and its valid-label fraction. A horizon with zero valid labels reports a null error. These errors use encoded targets, while the existing paired-match economy metrics report actual land, armies and producer counts.

## Centralized Q-critic

The Q-critic supplies action-dependent return estimates for policy training. Let $\phi$ denote its parameters and $Z_t$ its context, comprising the learner's encoded information, the simulator's complete board and the frozen opponent's current command. The value $Q_\phi(Z_t,a)$ estimates the shaped return associated with learner action $a$ in that context. The reward and return construction are defined in the next section.

This critic is an independent network with width 64, three graph blocks and four heads. Block 2 contains global attention. It has its own parameters and optimizer. The actor's inputs remain the player-visible encoded information $I_t$, and the centralized context supplies training estimates through the critic.

### Critic encoding

The critic begins with the learner's 28 spatial channels $X_t(u)$. A six-component vector $\operatorname{truth}_6(u)$ adds true own, enemy, neutral, mountain, city and general indicators. True log army adds one component. Three opponent-command markers, written $\operatorname{command}_3^-(u)$, use the same source, destination and half-movement convention as the actor's action history. These additions give a 38-component spatial input $x_u^Q$:

$$
x_u^Q=[X_t(u),\operatorname{truth}_6(u),\ell(A_u),\operatorname{command}_3^-(u)]\in\mathbb R^{38}.
$$

The critic also receives the learner's histories. It masks and flattens the eight spatial frames at each cell. For the long public history, it uses the same eight 64-frame segments defined for the actor. A segment's 768 values are projected to a sixteen-component token $c_j^Q$. Joining all eight tokens with the eleven current public statistics produces 139 inputs for the 64-component public summary $q^U$:

$$
c_j^Q=\operatorname{GELU}(W_c^Q\operatorname{vec}[mU,m]_j+b_c^Q),\qquad q^U=W_U^Q[\operatorname{vec}_{j=1}^{8}c_j^Q,U_t]+b_U^Q.
$$

Separate affine projections bring the spatial input and flattened spatial history to width 64. Adding the shared public summary and applying GELU gives the initial critic cell representation $h_u^{Q,0}$:

$$
h_u^{Q,0}=\operatorname{GELU}(W_X^Qx_u^Q+b_X^Q+W_R^Q\operatorname{vec}(mR(:,u))+b_R^Q+q^U).
$$

Critic transport edges use true mountains. Its blocks apply the graph equations with width 64, head dimension 16 and feed-forward width 256. Final LayerNorm produces the cell vectors $f_u^Q$. Learned scalar pooling, following the actor's pooling formula with independent parameters, produces global context $g^Q$.

### Action values

The critic uses the same source-target structure as the actor's action decoder. Four 64-component vectors and the two coordinate offsets give 258 inputs. A 64-component hidden layer leads to two scalar action values, for full and half movement. A separate affine readout of $g^Q$ supplies the pass value. Together these outputs form $Q_\phi(Z_t,a)$ for every action:

$$
Q_{u,v,:}=W_{Q,2}\operatorname{GELU}(W_{Q,1}[f_u^Q,f_v^Q,f_u^Q\odot f_v^Q,g^Q,p_v-p_u]+b_{Q,1})+b_{Q,2},\qquad Q_{\rm pass}=w_Q^\top g^Q+b_Q.
$$

Final move and pass layers initialize to zero. Actor legal probabilities determine which values participate in expectation. The opponent command in the critic context comes from frozen-policy argmax using that opponent's legal history.

## Rewards and optimization

One training iteration first collects a rollout with fixed actor and critic parameters. The rollout contains transitions, rewards, sampled learner commands and critic estimates. A backward computation converts these records into return targets and advantages. PPO then updates the actor using those fixed targets, and regression updates the critic. This section defines each step in that order.

### Material and potential

The reward transformation uses a scalar summary of relative material, called a potential. For player $p$, define land $L_p$, total armies $A_p$, and producing cells $P_p$ directly from the complete simulator state:

$$
L_p=\sum_u[O_u=p],\qquad A_p=\sum_u[O_u=p]A_u,\qquad P_p=\sum_u[O_u=p][C_u\lor G_u].
$$

For each material quantity $X\in\{L,A,P\}$, a clipped logarithmic ratio $f_X$ compares the two players. Counts are floored at one before division. The ratio reaches its clipping boundary at a fourfold advantage or disadvantage. A weighted sum gives the learner's potential $\Phi_i$, and the opponent's potential has the opposite sign:

$$
f_X=\operatorname{clip}\left(\frac{\log(\max(X_i,1)/\max(X_j,1))}{\log4},-1,1\right),\qquad \Phi_i=0.3f_L+0.3f_A+0.4f_P,\qquad\Phi_j=-\Phi_i.
$$

Let $r_t^{\rm outcome}$ be +1 for victory, -1 for defeat and zero for other transitions. Write $\Phi_t$ for the learner's potential at time $t$, $\gamma$ for the return discount, and $\eta$ for shaping strength. The true-terminal indicator $d_t$ sets the next potential to zero when the game ends. Adding the discounted potential change produces the training reward $r'_t$:

$$
r'_t=r_t^{\rm outcome}+\eta\left(\gamma(1-d_t)\Phi_{t+1}-\Phi_t\right),\qquad\gamma=1,\quad\eta=1.
$$

The logs separate this transformation into a continuation contribution for ongoing games and a reset contribution at true termination. Define these components as $r_t^{\rm continuation}$ and $r_t^{\rm reset}$:

$$
r_t^{\rm continuation}=\eta(1-d_t)(\gamma\Phi_{t+1}-\Phi_t),\qquad r_t^{\rm reset}=-\eta d_t\Phi_t,\qquad r'_t=r_t^{\rm outcome}+r_t^{\rm continuation}+r_t^{\rm reset}.
$$

Time-limit truncations retain continuation potential. For a complete episode with $T$ transitions ending in true termination, consecutive potential terms cancel in the discounted sum. The total shaping contribution is therefore determined by the initial potential:

$$
\sum_{t=0}^{T-1}\gamma^t(r'_t-r_t^{\rm outcome})=-\eta\Phi_0.
$$

### Q traces

The collected actor distribution $\pi_{\rm old}$ and critic $Q_{\rm old}$ remain fixed throughout a rollout. For each recorded context $Z_t$, define $V_t$ as the expectation of the critic's action values under the collected policy. Define $q_t$ as the critic value of the sampled learner command $a_t$:

$$
V_t=\sum_a\pi_{\rm old}(a\mid I_t)Q_{\rm old}(Z_t,a),\qquad q_t=Q_{\rm old}(Z_t,a_t).
$$

These estimates provide the baseline and sampled-action value for a backward return calculation. Let $c_t$ mark time-limit truncation, and let $\lambda=0.95$ control how much later residuals contribute. The next value $V_{t+1}$ uses the final observation before reset at truncation and the continuing observation at the rollout boundary.

For a rollout ending at index $T$, initialize the accumulated residual $\mathcal E_T=0$. At each earlier step, $\delta_t^Q$ compares the sampled-action value with the immediate reward plus the next expected value. The trace $\mathcal E_t$ combines this residual with the next trace while respecting terminal and truncation boundaries:

$$
\delta_t^Q=r'_t+\gamma(1-d_t)V_{t+1}-q_t,\qquad \mathcal E_t=\delta_t^Q+\gamma\lambda(1-d_t)(1-c_t)\mathcal E_{t+1},\qquad\lambda=0.95.
$$

Adding the accumulated residual to $q_t$ gives the return target $Y_t$. Subtracting the expected baseline $V_t$ gives advantage $\widehat A_t$, which describes how the sampled action's target compares with the collected policy's expected value. PPO uses its rollout-normalized version $\widetilde A_t$:

$$
Y_t=q_t+\mathcal E_t,\qquad\widehat A_t=Y_t-V_t,\qquad\widetilde A_t=\frac{\widehat A_t-\operatorname{mean}(\widehat A)}{\max(\operatorname{std}(\widehat A),10^{-6})}.
$$

Normalization spans the rollout. Targets $Y_t$ retain their reward scale. The reference-policy estimator keeps collected expectations, targets and advantages fixed during actor updates. Critic regression follows actor optimization.

### Actor value target

The actor's scalar value readout $V_{\rm actor}(I_t)$ predicts the same return target from player-visible information. Its squared-error loss $L_{V,\rm aux}$ adds a return-prediction task to the shared actor representation:

$$
L_{V,\rm aux}=(V_{\rm actor}(I_t)-Y_t)^2.
$$

The centralized Q-critic supplies the collected action-value estimates used above. The actor value head contributes this auxiliary regression loss during actor training.

### Actor objective

As actor parameters $\theta$ change, each recorded command receives a new probability. Let $\Delta_t$ be the change in its log probability, and $\rho_t$ the ratio of its updated probability to its collection-time probability:

$$
\Delta_t=\log\pi_\theta(a_t\mid I_t)-\log\pi_{\rm old}(a_t\mid I_t),\qquad\rho_t=\exp(\Delta_t).
$$

The PPO policy loss $L_{\rm policy}$ weights this ratio by normalized advantage and clips the ratio to the interval [0.8,1.2] in the surrogate term. Policy entropy $\mathcal H$ measures the spread of the full action distribution and contributes an exploration incentive. Each mean below runs over the current training batch:

$$
L_{\rm policy}=-\operatorname{mean}_t\min\left(\rho_t\widetilde A_t,\operatorname{clip}(\rho_t,0.8,1.2)\widetilde A_t\right),\qquad\mathcal H=-\operatorname{mean}_t\sum_a\pi_\theta(a\mid I_t)\log\pi_\theta(a\mid I_t).
$$

The actor objective adds the value, hidden-state, opponent-command and economic prediction losses defined above to the policy and entropy terms. With the default coefficients:

$$
L_{\rm actor}=L_{\rm policy}-0.01\mathcal H+0.5\operatorname{mean}L_{V,\rm aux}+0.02\operatorname{mean}L_{\rm hidden}+0.02\operatorname{mean}L_{\rm opponent}+0.02\operatorname{mean}L_{\rm econ}.
$$

Options `--entropy`, `--value-auxiliary`, `--auxiliary`, `--opponent-loss` and `--economy-loss` set these coefficients. The actor performs one shuffled epoch by default.

### Shared gradients

All actor losses are differentiated in one objective. To express this with configurable weights, let $c_E,c_V,c_H,c_O,c_{\rm econ}$ be the entropy, value, state, opponent-command and economic coefficients. Let an overbar denote the batch mean of a per-sample loss. The gradient is the sum of the weighted task gradients:

$$
\nabla_\theta L_{\rm actor}=\nabla_\theta(L_{\rm policy}-c_E\mathcal H)+c_V\nabla_\theta\overline L_{V,\rm aux}+c_H\nabla_\theta\overline L_{\rm hidden}+c_O\nabla_\theta\overline L_{\rm opponent}+c_{\rm econ}\nabla_\theta\overline L_{\rm econ}.
$$

Each decoder receives gradients from the losses that use its outputs. Shared encoder parameters receive contributions from all heads connected to them. This is how supervised predictions influence later policy decisions: they change the cell features and context that the policy reads. During execution, action probabilities continue to come directly from the policy decoder.

### Critic and optimizers

After actor optimization, the critic fits its sampled-action estimates to the fixed return targets $Y_t$. Its loss $L_Q$ is half the mean squared error, optimized over two shuffled epochs:

$$
L_Q=\frac12\operatorname{mean}_t(Q_\phi(Z_t,a_t)-Y_t)^2.
$$

Both optimizers use logical batches $B$ of 1,024 samples. To limit device memory, each batch is divided into microbatches $B_k$. Weighting each microbatch's mean-loss gradient by its sample fraction recovers the logical-batch gradient $g^{\rm grad}$. Global-norm clipping gives $g_c$:

$$
g^{\rm grad}=\sum_k\frac{|B_k|}{|B|}\nabla L(B_k),\qquad g_c=g^{\rm grad}\min\left(1,\frac{0.5}{\|g^{\rm grad}\|_2}\right).
$$

At zero norm, $g_c=0$. Independent Adam optimizers use learning rate $10^{-4}$, moment decay rates 0.9 and 0.999, and denominator stabilizer $10^{-8}$. At optimizer step $k$, let $m_k,v_k$ denote the first and second gradient moments, initialized to zero. Their bias-corrected forms are $\widehat m_k,\widehat v_k$. Squares and divisions act componentwise:

$$
m_k=0.9m_{k-1}+0.1g_c,\quad v_k=0.999v_{k-1}+0.001g_c^2,\quad\widehat m_k=\frac{m_k}{1-0.9^k},\quad\widehat v_k=\frac{v_k}{1-0.999^k},
$$

$$
\theta_{k+1}=\theta_k-10^{-4}\frac{\widehat m_k}{\sqrt{\widehat v_k}+10^{-8}}.
$$

The critic applies the same rule to $\phi$ using its own moments. Logs report gradient norms before clipping.

### KL control

KL control limits the actor's movement away from the policy that collected the rollout. A fixed random subset of at most 2,048 rollout samples serves as a probe throughout the update. Using the sampled-action log-probability change $\Delta_t$ defined above, the code computes the following estimator of the Kullback-Leibler divergence from the collected policy to the updated policy:

$$
\widehat D_{\rm KL}=\operatorname{mean}_t[\exp(\Delta_t)-1-\Delta_t].
$$

Before optimization, the current and collected policies should assign matching probabilities. The pre-update check accepts finite estimates at most $10^{-4}$. Checks then run every 16 actor optimizer steps and at the final actor step. A probe above 0.03 ends the actor update for that iteration, and critic regression proceeds afterward. Options `--target-kl`, `--kl-every` and `--kl-samples` configure this control. The log field `kl` averages minibatch measurements during optimization, while `probe_kl` records the latest measurement on the fixed probe.

## Opponents and evaluation

Training and automatic evaluation use the current opponent pool. Every iteration evaluates the candidate against each pool member, then uses those results to decide admission and retirement. PPO-20 and PPO-50 initialize the pool and remain eligible for retirement under the same rules as later members.

### Iteration and admission

The default run has 512 parallel environments. Each iteration collects 512 half-turns per environment, yielding $512\times512=262{,}144$ learner transitions. Games draw from a generated pool of 4,096 maps and have a 1,024-half-turn time limit. The run completes after 100 iterations by default. Several games can contribute to one environment's rollout when games finish and reset during collection.

Maps use side 24, minimum general distance 17 and city generation range `(1, max(2, side * side // 50))`. The learner alternates player seats. Each new game selects uniformly from the admitted frozen policies and retains that opponent for its duration. The learner samples actions from its policy distribution, and the frozen opponent selects its highest-probability action. Ongoing games preserve their state and both players' memories between iterations while the learner receives updated parameters.

The supplied PPO-20 and PPO-50 archives initialize the opponent pool. The actor under training is the candidate for future admission. After every iteration, that candidate plays 64 seat-paired games against each current pool member, using 16 concurrent games. Both players select argmax to measure deployment performance. Each iteration draws fresh held-out maps, shared across its opponent matchups. A seat pair uses the same map twice with the learner occupying opposite seats. Let $K$ be the number of pool members and $w_j$ the candidate's win rate against member $j$. The admission score averages these member-specific win rates:

$$
w_{\rm pool}=\frac1K\sum_{j=1}^{K}w_j,\qquad\operatorname{admit}=[w_{\rm pool}\ge0.50].
$$

When the score reaches 0.50, admission adds a frozen copy of the candidate. The pool holds at most three eligible opponents, ordered by admission time, with PPO-20 preceding PPO-50 initially. A pool with spare capacity keeps all existing members. For a full pool, let $W_j$ and $L_j$ be the candidate's wins and losses against member $j$, and let $a_j$ be that member's admission order. Define the defeated set $D$ and retiring member $j^*$ by:

$$
D=\{j:W_j>L_j\},\qquad
j^*=\begin{cases}
\arg\min_{j\in D}a_j,&D\ne\varnothing,\\
\arg\min_{j\in\{1,\ldots,K\}}a_j,&D=\varnothing.
\end{cases}
$$

This rule retires the oldest defeated opponent first. When all opponents are defeated, it retires the oldest pool member. The empty-set case also retires the oldest member, covering admission at the 50% threshold with tied individual matchups. Each successful full-pool admission replaces one member. Active games finish against their original opponent, so retired parameter slots remain available until those games release them. When the admission score falls below the threshold, the pool stays unchanged and the candidate continues training from its current parameters and optimizer state. The promotion log lists the defeated versions and the removed version.

### Evaluation records

Automatic evaluation consists of the pool-admission matches described above. Standalone evaluation allows a separately chosen opponent and defaults to argmax for both players. Its optional `--sample` mode measures the learner's categorical policy against an argmax opponent as a separately labeled diagnostic. Training retains categorical learner sampling. Match records use names such as `promotion-ppo20.json`, and admission decisions use `promotion.json`. Each result in `evaluation.jsonl` identifies both players' selection modes. Pool scores describe performance against that iteration's roster. Comparisons across changing rosters require attention to the opponent identities.

For one opponent matchup, let $G$ be the game count, $M=G/2$ the map-pair count, $W$ the number of learner wins and $L$ the number of learner losses. The reported win rate $\widehat w$ and net win rate $\widehat n$ use all games in their denominators. A Hoeffding radius $\epsilon$ is computed using the independent map pairs:

$$
\widehat w=W/G,\qquad\widehat n=(W-L)/G,\qquad\epsilon=\sqrt{\frac{\log40}{2M}}.
$$

Each map pair contributes an average win indicator in [0,1] and an average net score in [-1,1]. Their different ranges give the following 95% fixed-look intervals, denoted by $CI_w$ and $CI_n$:

$$
CI_w=[\max(0,\widehat w-\epsilon),\min(1,\widehat w+\epsilon)],\qquad CI_n=[\max(-1,\widehat n-2\epsilon),\min(1,\widehat n+2\epsilon)].
$$

Draws and truncations contribute zero wins and zero net score. Independent maps are sampling units, each containing both seats. The interval interpretation applies to a fixed evaluation look.

### Metrics

Evaluation also records behavior during the games. Across all $G$ games, let $T$ be the total active half-turn count, $P$ the learner's pass count, $P_o$ its passes when a movement was available, and $M_o$ the number of states with at least one available movement. These counts define average game length, overall pass frequency and optional-pass frequency:

$$
\operatorname{mean\_ticks}=T/G,\qquad\operatorname{pass\_fraction}=P/\max(T,1),\qquad\operatorname{optional\_pass\_fraction}=P_o/M_o.
$$

The optional-pass metric is null when $M_o=0$. Opponent-command accuracy is the fraction of active frames in which the prediction's argmax matches the frozen opponent's command. Negative log-likelihood (NLL) metrics average the corresponding prediction losses over active frames. State metrics report ownership NLL, conditional army-bin NLL and general-location NLL separately.

Economy metrics record learner and opponent land, armies and producing cells at half-turns 50, 100, 150, 200 and 400. Each average uses the games still active after that step, and its accompanying ongoing-game count identifies the population. Later milestones therefore describe the surviving games at that time.

Training additionally reports critic explained variance $\operatorname{EV}_Q$. It compares variation in the residual $Y-q$ with variation in the return target $Y$, using the collected sampled-action estimates $q$:

$$
\operatorname{EV}_Q=1-\frac{\operatorname{Var}(Y-q)}{\max(\operatorname{Var}(Y),10^{-8})}.
$$

Reward logs sum outcome, continuation and terminal-reset components. Reconciliation error is total shaped reward minus their total. Terminal-reset means divide by completed games. Truncated-potential means divide by truncations. Collection logs also report mean material, raw advantage standard deviation, chosen-Q statistics and target statistics.

## Storage and execution

The training process separates CPU rollout storage from device computation. Collection advances the games on the execution device and transfers compact records to a host timeline. Training reconstructs only the history windows needed for each batch. This organization keeps repeated observations shared in storage while preserving the input arrays defined earlier.

### Packed rollout

Of the 28 spatial channels in $X_t$, twenty are binary indicators. Let $\mathcal B$ list their zero-based channel indices in packing order:

$$
\mathcal B=(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,17,21,22,23,24).
$$

The other eight channels contain encoded counts, ages or coverage values. Their channel indices form the ordered list $\mathcal R$, and their stored values retain float32 precision:

$$
\mathcal R=(9,16,18,19,20,25,26,27).
$$

The binary channels and several masks fit into one 32-bit unsigned integer per cell. For a zero-based binary vector $b$ and starting bit position $s$, the packing operation is:

$$
\operatorname{bits}(b,s)=\sum_k\operatorname{uint32}(b_k)2^{s+k}.
$$

Bits 0-19 store the spatial channels in $\mathcal B$'s order, bit 20 stores the board mask, bits 21-24 store transport edges, bits 25-28 store full-move legality, and bits 29-31 store action markers. Reconstruction extracts each bit with a shift and one-bit mask. Since full and half moves share legality, each decoded movement-validity bit supplies both action modes. Pass validity is appended as true.

The six true spatial indicators used by the critic occupy six bits in a separate uint8 value. Its army and opponent-command inputs reuse the records stored for auxiliary labels. Public statistics are stored once per step and shared by reconstructed history windows. During collection, each full Q vector is reduced to the policy expectation and sampled-action value required by the trace equations.

At the default rollout size, economic flows, targets and validity masks add 24.75 MiB. The complete host rollout buffer occupies approximately 6.04 GiB, including those additions. This figure describes CPU data storage. Device memory separately holds active games, live histories, network parameters and temporary computation arrays.

### Batches and precision

The default inference batch contains 64 observations, and each actor training microbatch contains 64 samples. The critic has a separate `--critic-microbatch` setting, defaulting to the greatest common divisor of the logical batch size and 128. With the default logical batch of 1,024, the critic therefore processes 128 samples per microbatch. Both microbatch sizes divide the logical batch, and sample-weighted gradient accumulation preserves its loss definition.

Collection transfers records in chunks of 16 steps. Host prefetch reconstructs upcoming batches while the device computes the current batch. At time-limit truncations, collection gathers the affected environments into fixed-size inference batches and computes their bootstrap values from the final observations. Padding fills the last batch, and its outputs contribute zero. At the rollout boundary, every continuing environment receives a bootstrap value.

The execution option `--activation-checkpoint` controls intermediate activation storage during differentiation. The default `full` recomputes the actor history branch and all actor and critic graph blocks. The `blocks` setting retains the actor history activations and recomputes graph blocks. The `none` setting retains activations throughout both networks. Retaining more activations trades device memory for less recomputation. These settings preserve the parameter structure and loss equations. Floating-point reduction order can produce small numerical differences between execution configurations.

Automatic precision selects bfloat16 GPU matrix products and float32 CPU execution. Parameters, probability operations and reductions retain float32. Compiled computations are cached under `JAX_COMPILATION_CACHE_DIR`, which defaults to `.jax-cache`. `XLA_PYTHON_CLIENT_PREALLOCATE` defaults to `false`, allowing device allocations to grow as needed.

## Running and checkpoints

Use Python 3.12 or later. Setup installs pinned dependencies and checks the selected backend:

~~~bash
bash setup.sh && bash train.sh
~~~

CUDA 12 is the default backend. The setup arguments `cuda13` and `cpu` select the corresponding alternatives. Small CPU checks use `ALLOW_CPU=1` with explicitly reduced training dimensions.

Training starts at iteration zero from `initial/ppo20.zip` with fresh optimizers and a zero-output Q-critic. The archive `initial/ppo50.zip` supplies the second pool member. The first operation is rollout collection. Each completed iteration saves a full checkpoint, the latest policy and an immutable actor snapshot, then evaluates the candidate against the current pool. Evaluation and pool changes trigger another full-state save.

To resume all training state, provide the full checkpoint. The second command below extends the total iteration limit to 150:

~~~bash
bash train.sh --resume runs/ppo/checkpoint.zip
bash train.sh --resume runs/ppo/checkpoint.zip --iterations 150
~~~

The limit counts completed iterations. Statistical settings and the output directory are checked against checkpoint metadata. Actor and critic microbatch sizes, inference-batch size, transfer-chunk size and activation checkpointing may change on resume. Fresh runs require an empty output directory.

Checkpoints record league protocol `every-iteration-beaten-first-v1`. Full-state resume requires this protocol. A policy from an earlier protocol can initialize a fresh run through `--init`, with fresh optimizers and the initial opponent pool. Evaluation frequency is one assessment per iteration. The options `--promotion-games` and `--evaluation-environments` control games per member and parallel evaluation lanes.

The following examples use smaller runtime batches for a fresh run and evaluate a saved policy against PPO-20:

~~~bash
bash train.sh --microbatch 32 --inference-batch 32 --rollout-chunk 8
python -B evaluate.py runs/ppo/policy.zip --opponent initial/ppo20.zip --games 128 --lanes 16 --output runs/ppo/manual-ppo20.json
~~~

For a resumed run with additional device memory available, the following command retains actor history activations and uses 128-sample critic microbatches. The startup log records both microbatch sizes and the activation setting. Compare steady-state `rollout_seconds`, `update_seconds` and `critic_seconds` after compilation to measure the effect on the target GPU:

~~~bash
bash train.sh --resume runs/ppo/checkpoint.zip --activation-checkpoint blocks --critic-microbatch 128
~~~

The `none` activation setting enables a further memory-for-computation tradeoff. The `full` setting and `--critic-microbatch 64` reproduce the previous activation and microbatch configuration.

### Archive format

The runtime reads archive format `1014-economy-v1`. A policy archive contains actor parameters, configuration and metadata for policy evaluation or initialization. A full checkpoint additionally contains both optimizers, Q-critic parameters, unfinished games, both players' memories, pseudorandom-number generator state, NumPy shuffle state and frozen opponent parameters. These additional records restore the ongoing training process.

Offline migration preserved policy, pass, value, opponent-command, general-location and ownership weights from PPO-20 and PPO-50. Two added graph blocks initialize as identity residuals and new history branches initially contribute zero. Conditional army classification starts from fresh parameters. Metadata records source hashes and migration comparisons. The supplied files initialize fresh training under this architecture. Continuing a different architecture's full state requires a separate offline conversion.

The economic-head migration preserves every existing actor parameter and adds the two economic decoder layers. The supplied initial archives include this migration. Startup remains `bash train.sh` from iteration zero with fresh optimizers. Older full checkpoints require an explicit offline conversion of their optimizer and opponent-bank trees as well as their actor trees.

### Output files

- runs/ppo/checkpoint.zip stores the latest resumable state using atomic replacement.
- runs/ppo/policy.zip stores the latest actor and metadata.
- runs/ppo/snapshots/iteration-XXXXXXXX.zip stores immutable actor snapshots including iteration zero.
- runs/ppo/references/ stores the initial PPO-20 and PPO-50 opponent archives for match provenance.
- runs/ppo/promoted.zip stores the latest admitted candidate.
- runs/ppo/training.jsonl, evaluation.jsonl and evaluation/ store metrics and paired-match records.

## Module responsibilities

- features.py defines observations, graph indices, persistent memory, windows and masks.
- model.py defines actor history encoding, graph/attention blocks and prediction decoders.
- critic.py defines centralized Q values, traces and critic optimization.
- environment.py defines maps, batched games, views, labels and collection.
- rollout.py defines packed host storage and history reconstruction.
- train.py defines native archive persistence, optimizers and auxiliary losses.
- ppo.py defines policy optimization, KL probes, opponent admission and orchestration.
- evaluate.py defines paired matches and their metrics.
- rewards.py defines material, potential and reward components.
- economy.py defines economic flow accounting, finite-window labels and prediction loss.
- setup.sh, train.sh and requirements.txt define environment setup and launch.

## References

- [Superhuman AI for Generals.io Using Self-Play Reinforcement Learning](https://arxiv.org/html/2606.23348v1) provides context for public-score history and historical observations.
- [Artificial Generals Intelligence](https://arxiv.org/html/2507.06825v2) provides context for material shaping and historical-policy pools.
- [GAE Falls Short in Imperfect-Information Self-Play Reinforcement Learning](https://arxiv.org/html/2605.19235v1) provides context for centralized Q-boosting and VRPO. This implementation uses fixed rollout-reference expectations throughout each actor update.
