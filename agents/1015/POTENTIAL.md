# Observation-Based Potential in Agent 1015

## Purpose and Information Boundary

This experiment fits a compact description of economy, resource distribution and information to the recorded final outcomes of Bot-involved games. The feature definitions were agreed before fitting. Agent 1015 uses the frozen coefficients in `potential.json`. Agent 1014 retains its existing implementation and parameters.

Let $p$ denote the player being evaluated and let $q$ denote the opponent. Let $t$ denote the current half-turn. The information set $I_t^p$ contains player $p$'s legal observations through half-turn $t$. The proposed potential has the form

$$
\Phi_p(I_t^p)=2\sigma(z_\theta(I_t^p))-1,
\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

The scalar $z_\theta$ is a fitted outcome logit. Each player's potential uses that player's information. Different information sets can produce different beliefs, so the two potentials have independently evaluated outputs. Every true terminal potential receives the value zero during reward shaping.

The fitting data was reconstructed at every half-turn. Each example uses current public statistics, the fog-limited board and factual memory accumulated from consecutive legal observations. Complete-state production counts served as validation targets and separately labeled oracle comparisons. The training implementation in `potential.py` follows the same information boundary.

## Economic Features

Let $A_p$ and $L_p$ be player $p$'s public total army and land count. Let $P_p$ be the number of producing cells currently owned by player $p$, including cities and the general. Own production is directly countable from the current observation.

Define the existing economic ratio encoding

$$
r(a,b)=\operatorname{clip}\left(
\frac{\log(\max(a,1)/\max(b,1))}{\log4},-1,1\right).
$$

The land and army features are

$$
e_L=r(L_p,L_q),\qquad e_A=r(A_p,A_q).
$$

Enemy production is represented by an observable interval $[P_q^-,P_q^+]$. This gives an interval for the production ratio

$$
e_P^-=r(P_p,P_q^+),\qquad e_P^+=r(P_p,P_q^-).
$$

The fitted basis uses its center $e_P$, half-width $w_P$ and a certainty flag $k_P$

$$
e_P=\frac{e_P^-+e_P^+}{2},\qquad
w_P=\frac{e_P^+-e_P^-}{2},\qquad
k_P=\mathbf1[P_q^-=P_q^+].
$$

The center is an interval encoding. Its role is accompanied by the half-width and certainty flag. The five economic inputs are $e_L,e_A,e_P,w_P,k_P$.

### Observable Production Bounds

Terrain memory stores the latest visible terrain type for every previously revealed cell. Current visible enemy producers provide a lower bound. The upper bound also counts hidden cells whose public fog category and remembered terrain permit an enemy city or general. Both bounds are intersected with the public enemy land count and the continued existence of one live enemy general.

Consecutive ordinary transitions also constrain the interval. The implementation conservatively expands the previous production interval by two in either direction before intersecting it with current bounds. A changed own-general location identifies the special general-exchange case and bypasses these ordinary-transition constraints.

Public army changes provide another constraint on production half-turns. Define the oriented public army difference $\Delta_t=A_{p,t}-A_{q,t}$. Let $N_p$ and $N_q$ be troop losses to neutral defenders during the transition into half-turn $t$. With $b_t=\mathbf1[50\mid t]$, ordinary-transition accounting gives

$$
\Delta_t-\Delta_{t-1}=P_p-P_q+b_t(L_p-L_q)-N_p+N_q.
$$

Define $c_t=P_p+b_t(L_p-L_q)-(\Delta_t-\Delta_{t-1})$. Upper bounds $U_p,U_q$ on neutral losses then imply

$$
P_q=c_t-N_p+N_q\in[c_t-U_p,c_t+U_q].
$$

The loss bounds use the preceding observation. For an observed source with $a$ troops, an observed neutral target with $d$ defenders permits at most $\min(a-1,d)$ neutral losses. An unknown target permits the conservative upper bound $a-1$. Hidden enemy troops are bounded by public army minus visible enemy army. At most one action per player makes the maximum candidate loss a valid bound for the transition. These bounds prioritize validity over tightness.

Every reconstructed observation checks that the actual production count lies inside its observable interval. Across the current corpus, 481,846 player observations passed this check. At sampled positions, 5.30% of intervals are exact. The mean width is 57.41 production cells. These broad intervals identify a limitation of the current conservative inference procedure.

An engine-checked two-frame example illustrates the ambiguity. Both states have enemy army 10 and enemy land 2 at half-turn 101. The enemy makes the same move from a nine-army general into a hidden neutral city. In one state, the city has one defender and the enemy already owns another city. In the other, the target has zero defenders and the enemy owns an ordinary land cell in place of that other city. The same map terrain produces identical observations from the distant player. After the move and half-turn-102 growth, both enemy totals are army 12 and land 3. The producing-cell counts are respectively three and two. Their one-unit growth difference exactly offsets their one-unit neutral-loss difference. Verification checked equality of every public statistic and every observed cell in both frames.

## Resource Distribution

Let $G_p$ be the graph whose vertices are currently owned cells and whose edges join four-neighbor owned cells. Empty owned garrisons remain vertices. For a vertex $u$ with army $a_u$, define mobile mass $m_u$ and total mobile mass $M$

$$
m_u=\max(a_u-1,0),\qquad M=\sum_u m_u.
$$

Let $\mathcal C$ be the connected components of $G_p$, and let $M_c=\sum_{u\in c}m_u$ for a component $c$. The separation feature is

$$
S=1-\sum_{c\in\mathcal C}\left(\frac{M_c}{M}\right)^2.
$$

It measures the probability that two independently sampled units of mobile mass lie in different owned components.

For vertices in the same component, let $d_{G_p}(u,v)$ be the shortest-path length through currently owned cells. The conditional transport-distance feature is

$$
D=\frac{\sum_{c\in\mathcal C}\sum_{u,v\in c}m_um_vd_{G_p}(u,v)}{\sum_{c\in\mathcal C}M_c^2}.
$$

The sum includes ordered pairs and self-pairs. This makes $D$ an expected distance conditional on the sampled masses being connected. A single stack gives zero distance. Two equal stacks separated by two owned edges give $D=1$ because half of the ordered mass pairs are self-pairs.

When $M=0$, both features take zero. A separate availability flag $h_M=\mathbf1[M>0]$ distinguishes this case from a concentrated mobile army. Distance $D$ retains raw edge units. These three inputs describe current transport geometry. Additional moves that capture connecting cells can change both $S$ and $D$.

## Information Features

Let $N$ be the number of board cells, including mountains. Let $V_t^p$ be the set currently visible to player $p$. Let $K_t^p$ be the set whose concrete terrain has been observed by that player at some time through $t$. Let $E_t^p$ be the currently owned cells covered by the opponent's vision.

The three area features are

$$
k_T=|K_t^p|/N,\qquad v_T=|V_t^p|/N,\qquad x_T=|E_t^p|/\max(L_p,1).
$$

The engine reveals all eight neighbors of every owned cell. For an owned cell, every possible adjacent enemy vision source is therefore visible to its owner. This permits exact calculation of $E_t^p$ using current observations.

Two persistent flags describe general-location knowledge. The flag $g_E$ becomes one when the enemy general is observed. The flag $g_H$ becomes one when the own general is covered by enemy vision. Both flags retain one after the event. Memory updates occur at every half-turn, including intermediate frames between fitted examples.

The five information inputs are $k_T,v_T,x_T,g_E,g_H$. Concrete hidden enemy positions remain outside their inputs.

## Context

Context lets identical local statistics have different associations at different game scales and phases. Define $\ell(x)=\log(1+x)$. Let $H,W$ be the board dimensions. Nine context inputs are used

$$
\tau=\ell(t),\qquad
u_L=(50-(t\bmod50))/50,\qquad
u_P=(2-(t\bmod2))/2,
$$

$$
d_s=\ell(\min(H,W)),\qquad d_l=\ell(\max(H,W)),
$$

$$
a_\Sigma=\ell(A_p+A_q),\qquad l_\Sigma=\ell(L_p+L_q),
$$

$$
p_\Sigma=\frac{\ell(P_p+P_q^-)+\ell(P_p+P_q^+)}2,\qquad
w_\Sigma=\frac{\ell(P_p+P_q^+)-\ell(P_p+P_q^-)}2.
$$

The countdowns refer to the next growth event following the current observation. Sorted board dimensions preserve rotation invariance. The two production-total terms retain interval uncertainty.

## Combination Rules

The main vector $x$ contains five economic inputs, three distribution inputs, five information inputs and nine context inputs. Its dimension is 22. An intercept adds one coefficient.

Let $E=\{e_L,e_A,e_P\}$ be the economic-center group, let $R=\{S,D\}$ be the spatial group and let $J=\{k_T,v_T,x_T,g_E,g_H\}$ be the information group. Let $B$ contain all thirteen economic, distribution and information inputs.

The candidate basis adds the following fixed products

| Product family | Terms | Count |
|---|---|---:|
| Economic relationships | Each unordered pair in $E$ | 3 |
| Distribution and resources | Every $re$ for $r\in R,e\in E$ | 6 |
| Information and army balance | Every $je_A$ for $j\in J$ | 5 |
| Stage-dependent interpretation | Every $b\tau$ for $b\in B$ | 13 |
| Resource scale | Every $ea_\Sigma$ for $e\in E$ | 3 |
| Distance and map dimensions | $Dd_s,Dd_l$ | 2 |
| Growth timing | $e_Lu_L,e_Pu_P$ | 2 |
| Spatial joint effect | $SD$ | 1 |

Let $\mathcal P$ denote this set of 35 products. The fitted logit is

$$
z_\theta=\theta_0+\sum_{i=1}^{22}\theta_i x_i+\sum_{(i,j)\in\mathcal P}\theta_{ij}x_ix_j.
$$

The full model has 58 coefficients. For example, the slope associated with separation depends on army balance, land balance, production balance, time and transport distance. Consequently, an individual coefficient is one part of a conditional relationship. Its sign alone has a limited interpretation.

## Fitting and Partitions

The corpus contains 617 validated games involving the selected Bot accounts. Both legal player perspectives enter the fit. This balances outcome labels within every game and prevents a constant Bot-win prior from dominating the headline accuracy. Opponent perspectives include human or unidentified players. The supervision consists of positions and final results. Account names and player identities serve reporting only.

Every game has equal total weight. Within a game, sampled positions and both seats have equal weight. Samples start at half-turn 11 and follow stride 11, while memory consumes every half-turn. The full dataset has 43,138 player positions.

The initial-map hash preserves the previous partition assignment. Both seats and all sampled positions of a map remain in one partition.

| Partition | Games and distinct maps | Player positions |
|---|---:|---:|
| Training | 428 | 29,948 |
| Validation | 87 | 5,578 |
| Test | 100 | 7,612 |

Let $b_{ni}$ be basis term $i$ for sample $n$, let $w_n$ be its game-balanced weight, and let $y_n\in\{0,1\}$ be the recorded outcome. Training-only root-mean-square scales are

$$
s_i=\max\left(\sqrt{\sum_{n\in\mathrm{train}}w_nb_{ni}^2},0.05\right).
$$

The optimizer fits scaled coefficients $\alpha$ with logit $z_n=\sum_i\alpha_i b_{ni}/s_i$ and objective

$$
\mathcal L(\alpha)=\sum_{n\in\mathrm{train}}w_n\left[\log(1+e^{z_n})-y_nz_n\right]
+\frac{\lambda}{2}\sum_i\alpha_i^2.
$$

The exported coefficients are $\theta_i=\alpha_i/s_i$. Validation cross-entropy selects $\lambda$ from $\{10^{-4},10^{-3},10^{-2},10^{-1}\}$ and selects among economic-only, additive and interaction candidates. The selected model uses interactions and $\lambda=10^{-4}$.

The old formula and its three-weight refit use the original complete-state production statistic as explicitly labeled oracle comparisons. An additional old-form comparison substitutes the observable production-interval center and calibrates its scalar on the training partition.

## Results

Brier error is the weighted mean of $(\sigma(z_n)-y_n)^2$. Lower values indicate better outcome-probability predictions on this corpus.

| Model | Test cross-entropy | Test Brier | Test accuracy |
|---|---:|---:|---:|
| Constant balanced prior | 0.6931 | 0.2500 | 50.00% |
| Original 0.3/0.3/0.4, oracle production | 0.6643 | 0.2360 | 62.74% |
| Three-weight refit, oracle production | 0.6019 | 0.2126 | 61.81% |
| Original weights, observable production center | 0.6878 | 0.2474 | 54.11% |
| Fitted observable economic inputs | 0.6300 | 0.2224 | 61.06% |
| All 22 additive inputs | 0.6119 | 0.2143 | 62.32% |
| All inputs plus 35 products | **0.5650** | **0.1948** | **69.80%** |

Accuracy is measured on both perspectives here. The earlier economic-only report used Bot perspectives. Complementary oracle probabilities preserve Brier error under this perspective expansion, while threshold ties and seat-specific class frequencies can change accuracy.

Map-clustered paired bootstrapping with 5,000 resamples gives the following Brier differences, selected model minus comparison

| Comparison | Difference | 95% percentile interval |
|---|---:|---:|
| Original oracle formula | -0.04124 | [-0.05892, -0.02267] |
| Three-weight oracle refit | -0.01783 | [-0.03746, 0.00190] |
| Observable economic model | -0.02761 | [-0.04123, -0.01386] |
| Additive observed model | -0.01950 | [-0.03068, -0.00770] |

The map partition has already been examined in earlier experiments. These intervals summarize sampling variation within this exploratory corpus. Independent new games provide a stronger confirmation of generalization.

Results vary by Bot account. On ResBot perspectives, Brier improves from approximately 0.280 for the old formula to 0.199. On Average Joe perspectives, the selected model gives approximately 0.197, compared with 0.189 for the three-weight refit. On the other Joe account, the corresponding values are approximately 0.190 and 0.165. SB network contributes only three test games and gives approximately 0.223 versus 0.131.

Conditional economic-logit slopes describe associations in the sampled strategies and game stages. Examining these slopes helps interpret how the fitted formula supplies PPO shaping signals.

In the fitted sample relationships, negative partial slopes occur for some economic inputs. A partial derivative holds the other encoded inputs fixed, including totals, time and information. It describes the fitted observational relation. Causal claims about the result of adding troops require an additional argument. This distinction is particularly relevant when an attacking player wins after spending an early army advantage.

## Reward Shaping and Verification

The frozen potential is $\Phi=2\sigma(z)-1$. With discount $\gamma$, the shaping increment is

$$
F_t=\gamma\Phi_{t+1}-\Phi_t,
$$

with $\Phi_{t+1}=0$ at a true terminal state. Agent 1015 applies this potential difference to its learner rewards. The current package initializes its actor randomly and includes a gathering auxiliary decoder. Its PPO, Q traces and economic prediction retain their existing equations. The README specifies the complete training and evaluation workflow. Offline outcome prediction and the effect of shaping on learning are separate measurements.

Verification covers numerical graph examples, quarter-turn invariance, persistent terrain and exposure flags, current exposure against simulator truth, hidden enemy-general relocation, hidden army/ownership rearrangement, production-interval accounting, sample alignment with the earlier experiment, disjoint map partitions, balanced outcome labels, exact interaction products and parameter round-trip predictions.

## Training Integration

`potential.py` ports the offline equations to JAX. `observe()` first masks the simulator board and retains the public totals. `update()` consumes that legal view and advances a separate potential memory. `basis()` creates the same 58 named terms used by the fit. `evaluate()` computes the frozen potential.

The transport calculation performs exact breadth-first searches from positive-mobile-mass cells in blocks of at most 32 sources. Each block retains Boolean frontiers and visited nodes and accumulates distance-weighted pair mass. The algorithm preserves the fitted definition of $D$ while bounding its working arrays to source-block size times board area per learner. Actor observations retain their existing history and channels.

Each environment stores potential memory only for its learner seat. The memory contains remembered terrain, general-knowledge flags, production bounds, previous public army difference, neutral-loss bounds, previous general location, previous tick and the cached current potential. Each transition computes the following potential once. True termination uses zero. A horizon truncation uses the continuing potential before the environment resets. Fresh episodes receive fresh memory and a newly computed initial potential.

Full checkpoints serialize this memory with the unfinished game. Training settings include both `potential_schema` and `potential_sha256`. Full-state resume requires these identities to match. Fresh training constructs three reproducible random networks, `r1`, `r2` and `r3`, for the initial frozen pool. The learner begins from a separate trainable copy of `r1`. Independent fixed evaluation compares every completed iteration with the immutable initial `r1`. A full checkpoint resumes the entire training process.

Start a fresh run from this folder with:

```bash
bash train.sh
```

Use a new output folder when keeping an earlier run:

```bash
bash train.sh --output runs/observed
```

Resume a checkpoint produced by this version with:

```bash
bash train.sh --resume runs/observed/checkpoint.zip --output runs/observed
```

The earlier 26-feature potential uses a different coefficient schema and runtime state. Its checkpoints belong to that earlier experiment. This folder contains the JAX implementation, frozen coefficients and initialization archives required for training. Replay reconstruction and coefficient fitting were completed before packaging. Training loads `potential.json` directly.

