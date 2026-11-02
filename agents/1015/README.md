# Agent 1015

Agent 1015 learns to choose one Generals command from the information available to a player. Its policy network, called the actor, converts observations and remembered information into a probability distribution over movements and pass. A graph encoder processes neighboring cells, and attention connects distant cells and historical observations. The actor receives the current view, persistent facts and aligned readings of one 512-half-turn observation archive.

This version initializes the actor from the bundled `initial/r1.zip`, a lossy specialization of 1014-33. It retains the goal-conditioned gathering auxiliary head and the frozen observation-based potential. `potential.py` computes 22 inputs and 35 interaction products, with an intercept giving 58 fitted coefficients in `potential.json`. Equations and offline measurements appear in [POTENTIAL.md](POTENTIAL.md). The potential maintains its reward-specific factual state. The default seed is 1015. Checkpoint metadata binds resume to the potential schema and coefficient file.

Training repeatedly collects games and updates the actor with proximal policy optimization (PPO). Each game pairs the learning actor with a frozen policy from a three-member opponent pool, initially `r1`, `r2` and `r3`. All three initial members receive the bundled initializer's parameters. The learner begins with a separate trainable copy of `r1`. A separate action-value network, called the Q-critic, uses the simulator's complete state during training to estimate returns. Auxiliary heads train shared actor features to describe hidden state, opponent commands, economic outcomes and local gathering efficiency. Every iteration includes pool-admission matches and a separate fixed evaluation against the immutable initial `r1`. Only pool-admission results control membership changes.

The sections below follow the computation from observations to actor outputs, auxiliary targets, critic estimates and parameter updates. The final sections describe opponent selection, evaluation, storage and execution.

This document specifies the supplied implementation. At width 128, depth 8, four heads, a 32-frame statistical interval and 32 ordered recent frames, the actor has 3,691,031 parameters and the independent width-64 Q-critic has 370,448 parameters. Parameters use float32. Default GPU matrix products use bfloat16.

The archive retains 512 observations plus one boundary observation for differences. Interval boundaries move forward every half-turn. `--period 32` reads sixteen aligned intervals, while any positive divisor of 512 selects another granularity. Persistent facts carry information across the window boundary. The same archive reader serves live inference, PPO reconstruction and critic training.

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
- Destination- and budget-conditioned gathering efficiency from exact small-subgraph transport targets, including tied full, half and pass alternatives.

These items describe relationships parameterized by the implementation. Each cell retains 128 features through eight residual blocks. Four blocks contain global attention. The unified reader encodes ordered intervals and all public frames. Auxiliary state outputs describe per-cell ownership/army distributions and a board-wide general-location distribution. The action decoder reads shared features directly.

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

The current spatial array $X_t\in\mathbb R^{N\times28}$ contains current observations and persistent facts. A single archive stores records $F_s=(x_s,a_{s-1},U_s,E_s)$, where $x_s$ is the ten-channel observation, $a_{s-1}$ is the player's preceding command, $U_s\in\mathbb R^{11}$ contains public statistics, and $E_s$ contains deterministic event flags defined below.

The archive has capacity 513. Its last 512 records form the query window, and the preceding record supplies the first difference. The recorded count identifies valid rows after an episode begins. Current board, transport and action masks accompany the archive. These values form the actor's information $I_t$. Equations describe real arithmetic, with float32 storage and optional bfloat16 matrix products in the implementation.

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

### Unified observation archive

A movement command uses source cell $u$, direction $d\in\{0,1,2,3\}$ and mode $b\in\{0,1\}$, with full movement at zero and half movement at one:

$$
a=8u+2d+b,\qquad a_{\rm pass}=8N.
$$

For each recorded observation, the archive stores the preceding own command. The reader derives source, destination and half-source markers $s_s(u),d_s(u),h_s(u)$. Pass produces zero markers. Ordinary consecutive observations enable these markers.

Let $\Vert$ append a record and $\operatorname{tail}_k$ retain the last $k$ records. Archive update and its query window are

$$
\mathcal A_t=\operatorname{tail}_{513}(\mathcal A_{t-1}\Vert F_t),\qquad
\mathcal W_t=\operatorname{tail}_{512}(\mathcal A_t).
$$

An episode reset clears the archive and persistent facts. A rollout boundary preserves both. The physical ring cursor changes the storage position, while chronological query order follows time.

### Comparable army changes

Let $m_s$ indicate a recorded frame, $\tau_s$ its timestamp, and $v_s(u)$ cell visibility. A comparable transition requires two recorded consecutive observations and visibility in both:

$$
c_s(u)=m_sm_{s-1}[\tau_s=\tau_{s-1}+1]v_s(u)v_{s-1}(u).
$$

For category $p\in\{\mathrm{own},\mathrm{enemy}\}$, let $o_s^p(u)$ denote own or visible-enemy ownership. Define the category-specific valid comparison $q_s^p(u)$ and scaled army change $\Delta_s^p(u)$:

$$
q_s^p(u)=c_s(u)o_s^p(u)o_{s-1}^p(u),\qquad
\Delta_s^p(u)=q_s^p(u)\frac{\ell^{-1}(x_s(u)_9)-\ell^{-1}(x_{s-1}(u)_9)}{256}.
$$

A missing predecessor gives zero comparison validity. The boundary observation supplies the predecessor for the oldest query frame. The latest source, destination and half markers, both latest category changes and latest comparability form the six-component vector $D_t(u)$. Interval queries additionally expose both change sums, both valid comparison counts and the comparability count.

### Ordered recent observations and commands

Let $R$ be the configured recent read length, with default $R=32$ and allowed values from 1 through 512. This read selects the last $R$ rows of the shared archive. The statistical interval length and $R$ are independent configuration values. Current configuration exposes four times the eight-frame observation and command sequence used by the historical baseline, alongside the additional long-window spatial statistics.

For each timestamp $s$, let $\epsilon_s(u)$ contain six factual flags in this order: ever occupied by a player, first discovery, own gain, own loss, consecutive observation and real board cell. Let $k_s=m_sm_{s-1}[\tau_s=\tau_{s-1}+1]$ indicate that the preceding command belongs to a consecutive transition. The pass flag is $p_s=k_s[a_{s-1}=8N]$. The ordered per-cell vector has 24 components:

$$
q_s(u)=\left[x_s(u),s_s(u),d_s(u),h_s(u),\Delta_s^{\rm own}(u),\Delta_s^{\rm enemy}(u),c_s(u),\epsilon_s(u),k_s,p_s\right].
$$

Here $x_s(u)$ contributes the ten original observation channels. The next three components identify the source, destination and half-source of the same submitted command. Their shared timestamp preserves the source-destination pairing. The two changes and comparison-validity flag occupy the next three components, followed by six event flags and two command flags. Invalid frames and padding cells produce zero vectors. Public row $U_s$ refers to this same observation timestamp, and the stored command $a_{s-1}$ precedes it.

Every selected frame reaches the learned encoder before temporal compression. Increasing $R$ expands this ordered read while retaining the same archive allocation. The rolling statistics continue to summarize the entire 512-frame window. A recent length of at least eight covers the baseline's original ordered input extent. Values below eight provide an explicitly reduced-input configuration.

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

### Rolling interval statistics

These queries connect recurring local behavior with control, production, exposure and public material. Let $b$ be the configured interval length and $K=512/b$. Index intervals by $j=0,\ldots,K-1$ from oldest to newest. Their half-turn ranges are

$$
J_{t,j}=\{t-511+jb,\ldots,t-512+(j+1)b\}.
$$

Let $n_J$ count recorded observations in interval $J$. A position $k_J(s)=s-(t-511+jb)+1$ runs from 1 to $b$. Empty positions before the episode remain masked. Every new observation shifts every interval boundary by one half-turn. Statistics are extracted from the archive at query time.

A persistent bit $B_s(u)$ records that a cell has been observed under either player's ownership. During a live standard 1v1 game, occupied cells remain player-owned. Current own ownership $o_s(u)$ is fully observed. These rules define known enemy ownership $e_s^*(u)$:

$$
B_s(u)=B_{s-1}(u)\lor o_s(u)\lor e_s(u),\qquad e_s^*(u)=B_s(u)\land\neg o_s(u).
$$

Observed neutral ownership supplies the neutral category. The remaining real cells have unknown ownership. Army statistics use visible observations. The 28-channel current view retains last-seen records. The current encoder also receives $B_t$ and $e_t^*$.

Each queried interval produces 48 statistics per cell:

| Channels | Contents |
|---|---|
| 0 to 3 | Own, known-enemy, observed-neutral and unknown-ownership counts |
| 4 to 11 | Visible own-army count, sum, maximum, first, last, first position, last position, peak position |
| 12 to 19 | The same eight statistics for visible enemy armies |
| 20 to 27 | The same eight statistics for visible neutral armies |
| 28 | Visible observation count |
| 29 to 31 | Submitted source, destination and half-source command counts |
| 32 to 33 | Latest source and destination command positions |
| 34 to 36 | Observed own gain, own loss and first-ever sight counts |
| 37 | Realized own production |
| 38 to 42 | Own exposure count, exposed-army sum, maximum, latest position and peak position |
| 43 to 47 | Own and enemy comparable army-change sums, their valid comparison counts, and comparability count |

For ownership category $c$, let $m_{s,u,c}$ indicate a visible observation of that category. Let $a_{s,u}$ denote its observed army count, reconstructed by rounding $\exp(\log(256)x_{s,u,9})-1$ to an integer. Its count and sum are

$$
C_{J,u,c}=\sum_{s\in J}m_{s,u,c},\qquad S_{J,u,c}=\sum_{s\in J}m_{s,u,c}a_{s,u}.
$$

Maximum, first and last refer to valid observations. Their positions use $k_J(s)$, and equal maxima retain the latest position. Empty groups contain zero values and a zero count. Source and destination records describe the submitted command preceding the observation. Consecutive records establish own gain $[o_s\land\neg o_{s-1}]$ and loss $[o_{s-1}\land\neg o_s]$. First sight uses $v_s(u)[\text{cell previously unseen}]$.

The configured simulator applies growth after movement. On continuing consecutive observations, realized production uses current own ownership and current producing terrain $p_s(u)=[C(u)\lor G(u)]$:

$$
Y_{J,u}=\sum_{s\in J}[s>0]o_s(u)\left([50\mid s]+[2\mid s]p_s(u)\right).
$$

The spawn observation contributes zero production. Live policy inputs use ongoing observations. The collector clears terminal histories before their next use. Production describes already-paid growth.

Own exposure is $z_s(u)=o_s(u)[\sum_{v\in N_8(u)}e_s(v)>0]$. Every enemy sight source adjacent to own land is visible, making this flag exact. Exposure count and army sum are

$$
E_{J,u}=\sum_{s\in J}z_s(u),\qquad A^E_{J,u}=\sum_{s\in J}z_s(u)a_{s,u}.
$$

Exposed maximum and peak position are selected jointly among exposed observations. A short large-army passage remains recorded after the current garrison becomes small.

The reader also obtains every public frame in the interval, its mean and its first and last recorded values. These quantities are derived from the same archive. Public army channels retain their original log encoding.

### Statistical normalization

Counts divide by $\max(n_J,1)$. A visible-army sum $S$ becomes $S/\max(C,1)$ using its category's observation count $C$. Army means, maxima and endpoints use $\ell$. Positions divide by $b$. Production, exposed-army sum and exposed maximum use $\ell$. Each comparable-change sum divides by its own valid comparison count, clamped below at one. This yields $f_{J,u}\in\mathbb R^{48}$.

Let $U_J$ be the ordered public frames, $m_J$ their validity flags, and $\bar U_J,U_J^{\rm first},U_J^{\rm last}$ their mean and endpoints. The context vector $c_J\in\mathbb R^{12b+35}$ is

$$
c_J=[\operatorname{vec}(m_JU_J,m_J),\bar U_J,U_J^{\rm first},U_J^{\rm last},n_J/b,(t-t_J^{\rm last})/512].
$$

The complete public sequence and its summaries share one storage source. Interval summaries describe long-term spatial behavior, and the ordered recent read supplies frame-level observations and command pairs. Both reads use the same archive. The older part of the spatial window reaches the model through statistics, while the public sequence retains all 512 ordered rows.

## Actor network

The actor uses width 128, eight residual graph blocks and four attention heads. Every second block includes global attention. A shared per-cell representation supplies movement, pass, scalar value and auxiliary predictions.

For an affine map $L(x)=Wx+b$, the implementation casts matrix products to the configured compute type and returns float32 results. GELU and LayerNorm use the library definitions. Let $\mathcal N(x)$ denote LayerNorm over the feature dimension.

### Unified history encoding

The learned maps $F$ and $P$ project interval statistics and public context to 32 components. A learned vector $p_j$ identifies interval order. First project each cell's statistics:

$$
z_{j,u}=\operatorname{GELU}(F(f_{J_{t,j},u})).
$$

A learned affine map $M$ combines this vector with four- and eight-neighbor means. Board-valid neighbors contribute, and the divisors stay four and eight at boundaries:

$$
r_{j,u}=[z_{j,u},\tfrac14\sum_{v\in N_4(u)}z_{j,v},\tfrac18\sum_{v\in N_8(u)}z_{j,v}],
$$

$$
h_{j,u}^{\rm hist}=[n_{J_{t,j}}>0]\operatorname{GELU}(M(r_{j,u})+P(c_{J_{t,j}})+p_j).
$$

An affine map $O$ projects the ordered interval vectors to actor width, producing the long-window encoding $L_t^{\rm long}(u)$:

$$
L_t^{\rm long}(u)=O(\operatorname{vec}_{j=0}^{K-1}h_{j,u}^{\rm hist}).
$$

The recent read uses independent affine maps $F_R$, $M_R$ and $O_R$. The first two produce 32-component vectors. The third projects the ordered sequence to actor width. Let $\mathcal C(z)_u=[z_u,\tfrac14\sum_{v\in N_4(u)}z_v,\tfrac18\sum_{v\in N_8(u)}z_v]$ denote the same board-valid spatial concatenation used above. For each recent timestamp, the encoder computes

$$
z^R_{s,u}=\operatorname{GELU}(F_R(q_s(u))),\qquad
h^R_{s,u}=m_s\operatorname{GELU}(M_R(\mathcal C(z^R_s)_u)).
$$

Each spatial exchange uses one timestamp, so neighboring observations and command markers interact before the ordered temporal projection. The recent and long-window encodings combine as

$$
L_t^{\rm recent}(u)=O_R(\operatorname{vec}_{s=t-R+1}^{t}h^R_{s,u}),\qquad
L_t(u)=L_t^{\rm long}(u)+L_t^{\rm recent}(u).
$$

The learned temporal projection has separate weights for each ordered position. Statistical averaging occurs only in the long-window summary path. The recent path learns its temporal compression from frame-level inputs.

The current input adds the six immediate command/change features $D_t(u)$ and two persistent ownership facts to the existing spatial features. An affine map $S$ projects its 36 components:

$$
h_u^0=\operatorname{GELU}(S([X_t(u),D_t(u),B_t(u),e_t^*(u)])+L_t(u)).
$$

Padding cells are zeroed. Fresh runs restore the initializer's history parameters and train the shared backbone from the first update.

### Graph blocks

Eight width-128 blocks update the combined representation. Each block contains local messages and a 128-to-512-to-128 feed-forward residual. Blocks 2,4,6,8 also contain global attention. In the equations for one block, $h_u$ denotes its input. The first block receives $h_u^0$, and each later block receives the preceding block's output.

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

The policy and the auxiliary heads read features produced by the same actor encoder. The simulator supplies labels for hidden state, opponent commands and subsequent economic outcomes. A small exact transport solver supplies gathering labels from visible friendly cells. These prediction losses update the shared representation along with their own decoder parameters. At action selection time, the policy reads the shared features and global context directly. The auxiliary outputs serve as training readouts.

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

### Goal-conditioned gathering

The gathering task teaches the encoder to compare transport schedules with a shared action budget. A query specifies a connected patch of current friendly cells, one destination and an integer budget. Growth, opponent actions and transfers across the patch boundary are held fixed for this auxiliary problem. PPO continues to determine the strategic timing and destination of actual movement.

Let $G_P=(P,E_P)$ be the induced four-neighbor graph of the selected patch. Its size $m$ is between two and eight. The destination $r$ occupies local index zero, $a_u$ is the currently observed army count at patch cell $u$, and $k$ is the number of available commands. Each task chooses $k$ uniformly from $1,\ldots,m-1$. Full movement transfers $a_u-1$ armies from a source with at least two armies. Half movement transfers $\lfloor a_u/2\rfloor$. Passing leaves the armies unchanged and consumes one command.

Define $M_k(G_P,a,r)$ as the largest destination army achievable within $k$ commands. A connected-set enumeration gives its exact value:

$$
M_k(G_P,a,r)=a_r+\max_{\substack{r\in S\subseteq P\\G_P[S]\text{ connected}\\|S|-1\leq k}}\sum_{u\in S\setminus\{r\}}(a_u-1).
$$

The set $S=\{r\}$ supplies the option of retaining the current army. For any movement schedule, the used-edge component containing $r$ has at most $k+1$ vertices. Army conservation and the final one-army garrison at each other used cell bound its destination mass by the displayed expression. A maximizing connected set of minimum cardinality attains that bound: choose a rooted spanning tree and send armies from children to parents in postorder. Every selected non-root subtree has positive net contribution, because removing a subtree with a nonpositive contribution would improve the chosen set or reduce its cardinality. Each resulting full transfer is therefore legal. This argument also covers initially empty friendly cells.

For a legal first command $c$, let $T_c(a)$ denote the army vector immediately after that command. Its exact gathering value $Q_k^g(c)$ measures the best destination mass attainable afterward. Normalize masses by $Z=\max(1,\sum_{u\in P}a_u)$ and define the full set of optimal commands $\mathcal C^*$:

$$
Q_k^g(c)=M_{k-1}(G_P,T_c(a),r),\qquad y_c=Q_k^g(c)/Z,\qquad y_M=M_k(G_P,a,r)/Z,\qquad \mathcal C^*=\{c:Q_k^g(c)=M_k(G_P,a,r)\}.
$$

The labels cover full movement, half movement and pass. This preserves every tied optimal first command. The connected-set expression involves at most $2^8$ subsets, so the host computes labels directly without running additional games.

The gathering head reads the actor's shared cell representations $\bar h_u$. Its query provides the patch indices, patch edges, destination marker and budget. All patch arrays are padded to eight cells, with validity indicator $b_u$. Define the initial query-conditioned representation $z_u^{(0)}$, and let $j_u$ indicate the destination:

$$
z_u^{(0)}=b_u\operatorname{GELU}\left(W_c[\bar h_u,\bar h_r,j_u,k/7]+b_c\right),\qquad j_u=[u=r].
$$

Two message steps exchange these representations along patch edges. Their weights are shared between steps. With $N_P(u)$ denoting the patch neighbors of $u$, the update for $s=0,1$ is:

$$
\mu_u^{(s)}=\frac14\sum_{v\in N_P(u)}z_v^{(s)},\qquad z_u^{(s+1)}=b_u\left(z_u^{(s)}+\operatorname{GELU}\left(W_m[z_u^{(s)},\mu_u^{(s)}]+b_m\right)\right).
$$

A learned scalar readout $w_g,b_g$ assigns patch pooling weights $\alpha_u$. These weights produce context $g_P$. Each directed patch edge receives a two-component readout for full and half movement, using its geometric offset $\Delta p_{uv}$. Separate scalar outputs predict the pass value and optimal destination mass:

$$
\alpha_u=\operatorname{softmax}_{u\in P}\left(w_g^\top z_u^{(2)}+b_g\right),\qquad g_P=\sum_{u\in P}\alpha_u z_u^{(2)},
$$

$$
[\widehat q_{uv,\rm full},\widehat q_{uv,\rm half}]=W_q\operatorname{GELU}\left(W_e[z_u^{(2)},z_v^{(2)},z_u^{(2)}\odot z_v^{(2)},g_P,\Delta p_{uv}]+b_e\right)+b_q,\qquad [\widehat q_{\rm pass},\widehat M]=W_f g_P+b_f.
$$

All hidden vectors in this head have width 128. Eight cells, four directions and two movement modes give 64 movement outputs. Appending pass gives 65 command values, and the last output predicts $y_M$. Legal commands within the patch form the set $\mathcal C$. Define a comparison distribution $p_g$ with temperature $\tau=0.1$ and the per-task loss $L_g$:

$$
p_g(c)=\frac{\exp(\widehat q_c/\tau)}{\sum_{d\in\mathcal C}\exp(\widehat q_d/\tau)},\qquad L_g=\frac1{|\mathcal C|}\sum_{c\in\mathcal C}(\widehat q_c-y_c)^2+(\widehat M-y_M)^2-0.1\log\sum_{c\in\mathcal C^*}p_g(c).
$$

The regression terms teach mass prediction. The final term teaches first-command preference while preserving ties. At deployment the policy continues to use its ordinary action decoder. The gathering query and decoder are evaluated only during actor updates.

After each rollout, an independent random stream selects up to 4,096 transition indices. Each selected observation supplies a uniformly selected friendly destination. The patch grows by repeatedly choosing a random adjacent friendly cell until a randomly selected size limit from two through eight is reached, or its component is exhausted. Patches with fewer than two cells are skipped. The sampled tasks use current own-cell flags and army counts from the stored actor inputs. Their labels describe the static patch problem defined above.

Let $B$ be the number of rollout transitions, $S_g$ the accepted task indices and $N_g=|S_g|$. A task is stored only at its selected index. The sparse sampling weight $w_t$ preserves the task-average scale under shuffled PPO minibatching:

$$
w_t=\begin{cases}B/N_g&t\in S_g,\quad N_g>0\\0&\text{otherwise},\end{cases}\qquad \overline L_g=\frac1B\sum_{t=1}^B w_t L_{g,t}=\frac1{N_g}\sum_{t\in S_g}L_{g,t}\quad(N_g>0).
$$

An empty task set contributes zero gathering loss. `--gathering-samples` sets the sampled transition count, and `--gathering-loss` sets its actor-loss coefficient, default 0.02. A separate sparse task bank and index lookup avoid storing 65 labels for every rollout transition. Logs report task count, label-generation time, storage, gathering loss, normalized mass error, optimal-command accuracy and normalized regret. The latter uses $y_M-y_{\widehat c}$ for the head's selected command $\widehat c$. These are training-task diagnostics. Fixed-match results measure the behavior of the actual policy.

## Centralized Q-critic

The Q-critic supplies action-dependent return estimates for policy training. Let $\phi$ denote its parameters and $Z_t$ its context, comprising the learner's encoded information, the simulator's complete board and the frozen opponent's current command. The value $Q_\phi(Z_t,a)$ estimates the shaped return associated with learner action $a$ in that context. The reward and return construction are defined in the next section.

This critic is an independent network with width 64, three graph blocks and four heads. Block 2 contains global attention. It has its own parameters and optimizer. The actor's inputs remain the player-visible encoded information $I_t$, and the centralized context supplies training estimates through the critic.

### Critic encoding

The critic uses the same archive, interval reader and recent length as the actor. Its history encoder has independent parameters, 16-component interval and recent-frame embeddings and a width-64 output $L_t^Q(u)$.

Define $\operatorname{truth}_6(u)$ as true own, enemy, neutral, mountain, city and general indicators. Let $\operatorname{command}_3^-(u)$ contain the frozen opponent's source, destination and half markers. True army is $A_u$. The critic's current input has 44 components:

$$
x_u^Q=[X_t(u),\operatorname{truth}_6(u),\ell(A_u),\operatorname{command}_3^-(u),D_t(u)].
$$

An independent affine map $S_Q$ combines current context with the history encoding:

$$
h_u^{Q,0}=\operatorname{GELU}(S_Q(x_u^Q)+L_t^Q(u)).
$$

Critic transport edges use true mountains. Its three graph blocks have width 64, four heads and feed-forward width 256. The second block includes global attention. Final LayerNorm produces $f_u^Q$. Independent learned pooling produces context $g^Q$.

### Action values

The critic uses the same source-target structure as the actor's action decoder. Four 64-component vectors and the two coordinate offsets give 258 inputs. A 64-component hidden layer leads to two scalar action values, for full and half movement. A separate affine readout of $g^Q$ supplies the pass value. Together these outputs form $Q_\phi(Z_t,a)$ for every action:

$$
Q_{u,v,:}=W_{Q,2}\operatorname{GELU}(W_{Q,1}[f_u^Q,f_v^Q,f_u^Q\odot f_v^Q,g^Q,p_v-p_u]+b_{Q,1})+b_{Q,2},\qquad Q_{\rm pass}=w_Q^\top g^Q+b_Q.
$$

Final move and pass layers initialize to zero. Actor legal probabilities determine which values participate in expectation. The opponent command in the critic context comes from frozen-policy argmax using that opponent's legal history.

## Rewards and optimization

One training iteration first collects a rollout with fixed actor and critic parameters. The rollout contains transitions, rewards, sampled learner commands and critic estimates. A backward computation converts these records into return targets and advantages. PPO then updates the actor using those fixed targets, and regression updates the critic. This section defines each step in that order.

### Material and fitted potential

The reward transformation uses a frozen scalar potential based on the learner's legal current observation and factual history. Let $I_t^i$ denote that history for learner $i$. Economic inputs describe public army and land ratios, own production and an observable enemy-production interval. Resource-distribution inputs describe separation across owned connected components and mobile-army-weighted shortest-path distance within those components. Information inputs describe explored terrain, current vision, own-land exposure and persistent general-location knowledge. Time, growth countdowns, map dimensions and resource totals supply context.

Let $b(I_t^i)\in\mathbb R^{58}$ contain the intercept, 22 encoded inputs and 35 pairwise products defined in [POTENTIAL.md](POTENTIAL.md). Let $w$ be the frozen coefficient vector loaded from `potential.json`. The fitted logit $z_t^i$ and potential are

$$
z_t^i=w^\top b(I_t^i),\qquad \Phi_t^i=\tanh(z_t^i/2).
$$

Each player's information supports its own potential estimate. Training computes the learner's estimate and caches it in separate reward memory. A transition advances that memory from the following legal observation. Exact transport distances use blocked BFS on the current owned graph. The potential stays in $[-1,1]$ and all fitted coefficients stay fixed throughout RL. Actor observations and action selection retain their existing definitions.

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

The actor objective adds the value, hidden-state, opponent-command, economic and gathering prediction losses defined above to the policy and entropy terms. With the default coefficients:

$$
L_{\rm actor}=L_{\rm policy}-0.01\mathcal H+0.5\operatorname{mean}L_{V,\rm aux}+0.02\operatorname{mean}L_{\rm hidden}+0.02\operatorname{mean}L_{\rm opponent}+0.02\operatorname{mean}L_{\rm econ}+0.02\overline L_g.
$$

Options `--entropy`, `--value-auxiliary`, `--auxiliary`, `--opponent-loss`, `--economy-loss` and `--gathering-loss` set these coefficients. The actor performs one shuffled epoch by default.

### Shared gradients

All actor losses are differentiated in one objective. To express this with configurable weights, let $c_E,c_V,c_H,c_O,c_{\rm econ},c_g$ be the entropy, value, state, opponent-command, economic and gathering coefficients. Let an overbar denote the batch mean, including the sparse sampling weights for gathering. The gradient is the sum of the weighted task gradients:

$$
\nabla_\theta L_{\rm actor}=\nabla_\theta(L_{\rm policy}-c_E\mathcal H)+c_V\nabla_\theta\overline L_{V,\rm aux}+c_H\nabla_\theta\overline L_{\rm hidden}+c_O\nabla_\theta\overline L_{\rm opponent}+c_{\rm econ}\nabla_\theta\overline L_{\rm econ}+c_g\nabla_\theta\overline L_g.
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

Training draws opponents from the current pool. Every iteration evaluates the candidate against each pool member and uses those results to decide admission and retirement. The pool begins full with `r1`, `r2` and `r3`, in that order. All three members follow the same retirement rules as later members. A separate immutable `r1` reference supplies fixed evaluation throughout the run, including after the training pool retires its initial `r1` member.

### Iteration and admission

The default run has 512 parallel environments. Each iteration collects 512 half-turns per environment, yielding $512\times512=262{,}144$ learner transitions. Games draw from a generated pool of 4,096 maps and have a 1,024-half-turn time limit. The run completes after 100 iterations by default. Several games can contribute to one environment's rollout when games finish and reset during collection.

Maps use side 24, minimum general distance 17 and city generation range `(1, max(2, side * side // 50))`. The learner alternates player seats. Each new game selects uniformly from the admitted frozen policies and retains that opponent for its duration. The learner samples actions from its policy distribution, and the frozen opponent selects its highest-probability action. Ongoing games preserve their state and both players' memories between iterations while the learner receives updated parameters.

All three members receive the current-format parameters from `initial/r1.zip`. The optional `--init` argument selects another current-format initializer. Their archives are written to `references/r1.zip`, `references/r2.zip` and `references/r3.zip`. The learner starts from a separate array copy of `r1`, preserving the immutable reference when PPO donates its learner buffers. Each reference records its source archive hash and migration metadata. Full checkpoints preserve all frozen members. Training samples eligible members uniformly. Identical initial members share the same initial match results, so the first accepted candidate retires `r1`. Later decisions prioritize defeated members, breaking ties by admission order.

The actor under training is the candidate for future admission. After every iteration, that candidate plays 64 seat-paired games against each current pool member, using 16 concurrent games. Both players select argmax to measure deployment performance. Each iteration draws fresh held-out maps, shared across its opponent matchups. A seat pair uses the same map twice with the learner occupying opposite seats. Let $K$ be the number of pool members and $w_j$ the candidate's win rate against member $j$. The admission score averages these member-specific win rates:

$$
w_{\rm pool}=\frac1K\sum_{j=1}^{K}w_j,\qquad\operatorname{admit}=[w_{\rm pool}\ge0.50].
$$

When the score reaches 0.50, admission adds a frozen copy of the candidate and retires one of the three eligible members. Initial ordering is `r1`, `r2`, `r3`, followed by admitted iterations. Let $W_j$ and $L_j$ be the candidate's wins and losses against member $j$, and let $a_j$ be that member's admission order. Define the defeated set $D$ and retiring member $j^*$ by:

$$
D=\{j:W_j>L_j\},\qquad
j^*=\begin{cases}
\arg\min_{j\in D}a_j,&D\ne\varnothing,\\
\arg\min_{j\in\{1,\ldots,K\}}a_j,&D=\varnothing.
\end{cases}
$$

This rule retires the oldest defeated opponent first. When all opponents are defeated, it retires the oldest pool member. The empty-set case also retires the oldest member, covering admission at the 50% threshold with tied individual matchups. Each successful full-pool admission replaces one member. Active games finish against their original opponent, so retired parameter slots remain available until those games release them. When the admission score falls below the threshold, the pool stays unchanged and the candidate continues training from its current parameters and optimizer state. The promotion log lists the defeated versions and the removed version.

### Evaluation records

Automatic evaluation has two independent roles. Fixed evaluation plays 128 games against the immutable `r1` reference every iteration, using 64 map pairs and 16 concurrent games. Both players choose argmax. Its seed is `seed + 1000000`, and every iteration reuses these held-out maps for comparable measurements. The reference is generated once in the run directory. A parameter fingerprint is checked on resume, and evaluation records include its archive SHA-256. These matches produce monitoring records only. Training batches and promotion scores use their own data.

Pool-admission evaluation uses the current three opponents and fresh maps with seed `seed + 2000000 + iteration`. The code checks map identities to keep its maps separate from both training and fixed evaluation. Admission uses only these pool-match results. Fixed evaluation tracks progress against the initial `r1` policy. Its score measures that particular matchup, while the evolving pool provides additional opponents.

Standalone evaluation allows a separately chosen opponent and defaults to argmax for both players. Its optional `--sample` mode measures the learner's categorical policy against an argmax opponent as a separately labeled diagnostic. Training retains categorical learner sampling. Fixed matches use `fixed-r1.json`, pool matches use names such as `promotion-r1.json`, and admission decisions use `promotion.json`. Each result in `evaluation.jsonl` identifies its `kind`, opponent hash and both players' selection modes. Pool scores describe performance against that iteration's roster. Fixed scores retain the same reference and maps.

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

### Packed observation archive

Each cell and observation stores nine binary observation flags and six event flags in a uint16. Event flags identify ever-player-owned, first sight, own gain, own loss, a consecutive transition and a real board cell. If $x_{s,u,k}$ is observation flag $k$ and $E_{s,u}$ the six-bit event word, packing is

$$
F_{s,u}=\sum_{k=0}^{8}x_{s,u,k}2^k+2^9E_{s,u}.
$$

A separate float32 preserves the observed log army exactly. Each frame also stores one int32 own command and eleven float32 public values. Thus an $N$-cell archive uses

$$
513(6N+48)+8\quad\text{bytes}.
$$

For 576 cells this is 1,797,560 bytes per player. The 512-environment, two-player live archives occupy approximately 1.714 GiB. Current features and persistent records bring one complete player memory to 1,893,753 bytes, or approximately 1.806 GiB for both players in all 512 environments. Allocator workspace, games, parameters and activations add further memory. The runtime stores each archive once inside its player's memory.

This layout uses bounded bit-packed rows on the accelerator. It preserves every observed float32 army and every observation flag. Compression relative to a dense 13-float per-cell frame is approximately 8.7-fold for the spatial part. Query tensors are temporary and exist within inference or training microbatches.

### Shared PPO timelines

Host storage keeps one observation timeline per learner lane, with 512 prefix rows and the newly collected observations. A sampled time selects 513 chronological rows and its recorded count. This reconstructs the same input as live inference, including the predecessor of the oldest query frame. A reset limits validity to the new episode, and a rollout boundary preserves the preceding timeline.

Current persistent spatial features use one uint32 for twenty binary channels, board validity, four transport edges and four action-validity bits. The seven float32 channels are indices $(16,18,19,20,25,26,27)$. Current observed army, channel 9, comes from the archive. Full and half movement share legality, and pass is always legal. Critic truth flags occupy a separate uint8, and true armies and opponent commands reuse auxiliary labels.

PPO and critic minibatches share the same archive format and reader. The reader generates rolling statistics, aligned public context and the requested recent frames inside the compiled computation. Each lane stores one shared spatial timeline. Economic labels and the sparse gathering task bank retain their existing definitions.

After collection, the four timeline arrays (packed flags, observed armies, commands and public values) upload once to the execution device. Each sampled transition carries its time index, environment index and recorded count. Inside the actor, critic or KL microbatch, these indices gather the same 513 chronological rows used by live inference. The logical minibatch carries current features, labels and indices. Its histories materialize at microbatch scope. The fixed KL subset shares the same device timeline, and all optimization passes reuse it until the iteration finishes.

Let $T$ be rollout length, $E$ environment count and $N$ padded cell count. The shared device timeline occupies

$$
M_{\rm timeline}=(T+512)E(6N+48)\quad\text{bytes}.
$$

For $T=E=512$ and $N=576$, this is 1.711 GiB uploaded once per iteration. Expanding one archive per sample for one actor pass and two critic passes would transfer approximately 1,316.6 GiB of historical rows. Current features, labels and indices still transfer per batch. Device-local gathering and statistical reads also remain part of each forward computation. The optimization preserves sample order, microbatch weighting, losses and optimizer steps. Logs report `timeline_gib` and `timeline_seconds` for the shared upload.

The recent decoder expands only $R$ frames into 24 channels. Its float32 output occupies $4BRN\cdot24$ bytes for a microbatch of $B$ observations. At $B=64$, $R=32$ and $N=576$, this tensor is 108 MiB. Expanding all 512 frames would make the same tensor 1,728 MiB. These figures describe this single logical tensor, and total device memory additionally includes projections, backward activations, statistics, parameters and compiler workspace. Neighbor features accumulate one direction at a time, keeping the eight-neighbor dimension outside the intermediate activation layout.

### Batches and precision

The default inference batch contains 64 observations, and each actor training microbatch contains 64 samples. The critic has a separate `--critic-microbatch` setting, defaulting to the greatest common divisor of the logical batch size and 128. With the default logical batch of 1,024, the critic therefore processes 128 samples per microbatch. Both microbatch sizes divide the logical batch, and sample-weighted gradient accumulation preserves its loss definition.

Collection transfers records in chunks of 16 steps. Host prefetch packs upcoming transition features, labels and window indices while the device computes the current batch. At time-limit truncations, collection gathers the affected environments into fixed-size inference batches and computes their bootstrap values from the final observations. Padding fills the last batch, and its outputs contribute zero. At the rollout boundary, every continuing environment receives a bootstrap value.

The execution option `--activation-checkpoint` controls intermediate activation storage during differentiation. The default `full` recomputes the actor history branch and all actor and critic graph blocks. The `blocks` setting retains the actor history activations and recomputes graph blocks. The `none` setting retains activations throughout both networks. Retaining more activations trades device memory for less recomputation. These settings preserve the parameter structure and loss equations. Floating-point reduction order can produce small numerical differences between execution configurations.

Automatic precision selects bfloat16 GPU matrix products and float32 CPU execution. Parameters, probability operations and reductions retain float32. Compiled computations are cached under `JAX_COMPILATION_CACHE_DIR`, which defaults to `.jax-cache`. `XLA_PYTHON_CLIENT_PREALLOCATE` defaults to `false`, allowing device allocations to grow as needed.

## Running and checkpoints

Use Python 3.12 or later. Setup installs pinned dependencies and checks the selected backend:

~~~bash
bash setup.sh && bash train.sh
~~~

CUDA 12 is the default backend. The setup arguments `cuda13` and `cpu` select the corresponding alternatives. Small CPU checks use `ALLOW_CPU=1` with explicitly reduced training dimensions.

Training starts at iteration zero from the bundled `initial/r1.zip`, with fresh optimizers and a zero-output Q-critic. The learner and initial pool members `r1`, `r2` and `r3` all receive these actor parameters. The initializer path is resolved relative to `ppo.py` and appears in the startup log. A missing file immediately stops startup with an upload instruction. Each reference is preserved inside the run directory. The first operation is rollout collection. Every completed iteration saves a full checkpoint, latest policy and actor snapshot, then performs fixed and pool-admission evaluations. Evaluation and pool changes trigger another full-state save.

Start a fresh run with:

~~~bash
bash train.sh
~~~

This path initializes the learner and all three frozen opponents from the supplied actor. Each pool member has its own identity and follows the same retirement rule. Iteration starts at zero, both optimizers are fresh and the Q-critic starts with zero-output heads. Fixed evaluation uses this run's immutable `r1` copy. The initializer is a lossy specialization of 1014-33: compatible trunk and decoder weights are copied, and the new spatial/history encoder is calibrated on legal teacher trajectories. The gathering head retains random initialization. The archive and its companion `initial/r1.json` record the source hash, fitting procedure and measured errors. Behavioral differences are part of this initialization. The cloud runtime uses the current network and carries all required weights in the initializer.

The offline calibration uses 104 observations from two teacher self-play maps and 70 observations from a third map for validation. All maps have side 24 and minimum general distance 17. Five hundred encoder-only updates minimize the squared difference between teacher and student per-cell representations after subtracting each representation's channel mean. A common positive input offset keeps the new GELU input in its approximately linear region. Pre-normalized residual blocks and final LayerNorm make this channel-constant offset irrelevant in exact arithmetic. The temporal encoding approximation and floating-point computation still change behavior. On the validation trajectories, top-1 action agreement is 92.9% and mean teacher-to-student policy KL is 0.232. These measurements describe one held-out map and support a limited initialization check.

To resume all training state, provide the full checkpoint. The second command below extends the total iteration limit to 150:

~~~bash
bash train.sh --resume runs/ppo/checkpoint.zip
bash train.sh --resume runs/ppo/checkpoint.zip --iterations 150
~~~

The limit counts completed iterations. Statistical settings and the output directory are checked against checkpoint metadata. Actor and critic microbatch sizes, inference-batch size, transfer-chunk size and activation checkpointing may change on resume. Fresh runs require an empty output directory.

Checkpoints record league protocol `seeded-three-beaten-first-v1`. The run directory contains the reference required for resume. Resume checks the league protocol, network configuration, training settings, potential and immutable reference fingerprint. Runs created by the earlier random-initialization protocol require their matching code version for resume. The options `--width`, `--depth`, `--heads`, `--period` and `--recent` specify the network and must match the initializer. Period is a positive divisor of 512. Recent length ranges from 1 through 512 and defaults to 32. The options `--promotion-games`, `--fixed-games` and `--evaluation-environments` control evaluation sizes and concurrency. Fixed evaluation always uses the run's initial `r1`.

The following examples use smaller runtime batches for a fresh run and evaluate a saved policy against its frozen initializer:

~~~bash
bash train.sh --microbatch 32 --inference-batch 32 --rollout-chunk 8
python -B evaluate.py runs/ppo/policy.zip --opponent runs/ppo/references/r1.zip --games 128 --lanes 16 --output runs/ppo/manual-r1.json
~~~

For a resumed run with additional device memory available, the following command retains actor history activations and uses 128-sample critic microbatches. The startup log records both microbatch sizes and the activation setting. Compare steady-state `rollout_seconds`, `update_seconds` and `critic_seconds` after compilation to measure the effect on the target GPU:

~~~bash
bash train.sh --resume runs/ppo/checkpoint.zip --activation-checkpoint blocks --critic-microbatch 128
~~~

The `none` activation setting enables a further memory-for-computation tradeoff. The `full` setting and `--critic-microbatch 64` reproduce the previous activation and microbatch configuration.

### Archive format

The runtime reads archive format `1015-unified-memory-v2`, which includes the ordered-recent encoder and its configured read length. A policy archive contains actor parameters, configuration and metadata for policy evaluation or initialization. A full checkpoint additionally contains both optimizers, Q-critic parameters, unfinished games, both players' memories, pseudorandom-number generator state, NumPy shuffle state and frozen opponent parameters. These additional records restore the ongoing training process. The run directory also preserves the immutable initial reference archives used for evaluation provenance.

Fresh runs copy the bundled current-format actor into the learner and all three initial opponents, and initialize a fresh critic from PRNG keys. Full-state resume restores the matching archive layout, encoders, optimizers and runtime memory. The source folder is self-contained and carries both the initializer and frozen potential coefficients alongside the code.

### Output files

- runs/ppo/checkpoint.zip stores the latest resumable state using atomic replacement.
- runs/ppo/policy.zip stores the latest actor and metadata.
- runs/ppo/snapshots/iteration-XXXXXXXX.zip stores immutable actor snapshots including iteration zero.
- runs/ppo/references/ stores immutable `r1.zip`, `r2.zip` and `r3.zip` archives for match provenance and fixed evaluation.
- runs/ppo/promoted.zip stores the latest admitted candidate.
- runs/ppo/training.jsonl, evaluation.jsonl and evaluation/ store metrics and paired-match records.

## Verification

CPU checks cover exact archive reconstruction, window rollover beyond 512 frames, query periods 1, 2, 32, 128 and 512, ordered reads of 1, 8, 32, 63 and 512 frames, source-destination markers, army-change validity, persistent facts, episode reset within a host timeline, live-versus-PPO input and logit agreement, hidden-state isolation, gathering labels and finite gradients in both history paths. A direct comparison with the historical encoder reproduces its current cells, masks, eight ordered observation/command frames and 512 public rows from the new inputs. A two-iteration small-board workflow exercises bundled initialization through the default command path, missing-initializer rejection, actor and critic updates, fixed evaluation, pool assessment and full checkpoint resume. The checks compare all three initial parameter trees and remove the bundled initializer before resuming from the run's reference copy. Retirement checks cover an oldest-member tie, a later defeated member and the next oldest member after admission. Frozen references retain their parameters across learner updates.

A 24-by-24 CPU spot check with a full archive and the default actor measured approximately 5.4 ms per archive reading and 63.6 ms per complete actor forward call after compilation. The timing averages ten calls for one observation at a time. GPU throughput and peak training allocation require measurements on the target device.

Indexed-window tests compare reconstructed inputs exactly against host-expanded windows across episode reset, rollover and repeated sample indices. Actor updates, critic updates and fixed-subset KL agree within float32 tolerances. A 128-sample, 576-cell host packing spot check measured approximately 72 ms and 222.2 MiB per expanded batch, versus 0.90 ms and 2.76 MiB per indexed batch. This measures packing and payload size independently of GPU execution.

## Module responsibilities

- features.py defines observations, graph indices, persistent memory, windows and masks.
- model.py defines actor history encoding, graph/attention blocks and prediction decoders.
- critic.py defines centralized Q values, traces and critic optimization.
- environment.py defines maps, batched games, views, labels and collection.
- rollout.py defines packed host storage and history reconstruction.
- memory.py defines the packed rolling archive and aligned interval queries.
- train.py defines native archive persistence, optimizers and auxiliary losses.
- ppo.py defines policy optimization, KL probes, opponent admission and orchestration.
- evaluate.py defines paired matches and their metrics.
- rewards.py defines material, potential and reward components.
- economy.py defines economic flow accounting, finite-window labels and prediction loss.
- gathering.py defines exact static transport labels, sparse query sampling and auxiliary loss.
- potential.py and potential.json define the observation-based potential and its frozen coefficients.
- setup.sh, train.sh and requirements.txt define environment setup and launch.

## References

- [Superhuman AI for Generals.io Using Self-Play Reinforcement Learning](https://arxiv.org/html/2606.23348v1) provides context for public-score history and historical observations.
- [Artificial Generals Intelligence](https://arxiv.org/html/2507.06825v2) provides context for material shaping and historical-policy pools.
- [GAE Falls Short in Imperfect-Information Self-Play Reinforcement Learning](https://arxiv.org/html/2605.19235v1) provides context for centralized Q-boosting and VRPO. This implementation uses fixed rollout-reference expectations throughout each actor update.
