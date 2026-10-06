# Training

The 2026 paper provides the target algorithm for the agent in `src/agents/nebula/`. PyTorch handles the neural network and optimization, while the existing C++ engine supplies self-play games through `arena`. Both training and deployment use `features.hpp` to construct observations from each player's own field of view.

## Running

The Python dependencies are listed in the root `requirements.txt`. The build places the graphical application and its runtime libraries in `build/interface/`. The training library and optional Nebula executable use `build/nebula/`. The independent `simple` program stays in `build/`.

From the repository root on Windows:

```powershell
.\scripts\build.bat
.\scripts\train.bat --profile check
```

On Linux or macOS:

```sh
bash scripts/build.sh -DNEBULA_BUILD_INTERFACE=OFF
bash scripts/train.sh --profile check
```

`check` uses a small network and two brief updates to exercise sampling, optimization and saving. `paper` selects the seven-layer model and large rollout configuration. GPU training selects a CUDA-enabled PyTorch installation through `--device cuda`.

```powershell
.\scripts\train.bat --profile paper --device cuda --output runs/nebula
.\scripts\train.bat --resume runs/nebula/checkpoint.pt --updates 100000
```

The output directory contains `checkpoint.pt` and `metrics.jsonl`. A checkpoint stores model weights, EMA weights, optimizer state, update count, random state and curriculum stage. Resuming begins fresh games at the saved stage and continues the optimization schedule. `--updates` specifies the total target update count.

The paper-sized rollout contains $512\times512\times2$ player transitions and uses about 25.9 GiB of host memory for board, mask and temporal buffers. Model activations, optimizer state and minibatch transfers require additional memory. The default execution check uses about 3 MiB for these buffers.

## Observation and Model

The input is padded to $24\times24$. Thirty-eight spatial channels contain visible armies, public scores, remembered terrain, inferred sight, last-seen enemy armies, coordinates and seven successive army differences for each side. Two further inputs contain 512 observations of enemy army and land totals. The seven-step spatial history advances on each received observation, while last-seen age follows the elapsed half-turn count.

The network embeds $3\times3$ patches into 448 dimensions. Seven pre-normalized attention blocks use eight heads, SiLU feed-forward layers and an intermediate width of 1344. Two independent temporal encoders add army and land tokens beside a learned value token. The complete network has 15,351,633 parameters.

Each patch produces four full moves, four half moves and a pass for each cell. The action mask uses owned source cells, visible mountains and board boundaries. Pass remains available at every padded position to preserve the reference action distribution. The value head predicts 128 bins spanning $[-1,1]$.

## Optimization

One current policy controls both sides of every sampled game. Capturing the opposing general supplies $+1$ to the winner and $-1$ to the loser. Other transitions carry zero reward. The engine also supplies its normal terminal adjudications.

GAE combines reward and a bootstrapped value estimate:

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
A_t=\delta_t+\gamma\lambda A_{t+1},\qquad \gamma=1,\quad\lambda=0.9.
$$

A terminal transition uses a zero bootstrap. A time-limit transition uses the value of its final board. Both boundaries end the recursive advantage trace.

Each update normalizes advantages and selects the largest 25 percent by absolute value. The selected samples train both the clipped PPO objective and the HL-Gauss value objective. PPO uses clipping $0.2$, value weight $0.5$ and gradient-norm limit $0.267$. HL-Gauss integrates a Gaussian of standard deviation $0.04$ across each bin, then normalizes the resulting probabilities. Value targets are clipped to the head's support before this calculation.

The learning rate and entropy coefficient follow

$$
\eta_k=\operatorname{clip}\left(\frac{0.5}{(k+1)^{1.1}},5\cdot10^{-6},10^{-4}\right)
$$

$$
\beta_k=\max\left(0.001,\frac{0.05}{(k+1)^{0.2}}\right).
$$

EMA updates once per training iteration with decay $0.999$. Deployment uses these averaged weights.

## Curriculum and Scope

The first stage uses general-to-general BFS distance 3 through 4. Later stages use 4 through 8, 6 through 13, 11 through 17 and 17 through 28. Evaluation against a random legal mover gates advancement at a win rate of $0.6$. This evaluation measures curriculum readiness. Competitive strength calls for separate matches against capable opponents and authorized ladder play.

The current implementation runs optimization on one selected device. Its C++ arena creates maps through the local mainstream generator, then relocates generals to satisfy the stage distance. Map generation, the serial CPU environment loop and single-device training are adaptations of the paper's JAX environment and four-device execution. The released recipe uses a large reusable map pool, while this implementation generates maps as games reset. The current terrain and city distribution follows the local engine.

The network, observation dimensions and principal PPO settings match the paper and the corresponding released configuration. The paper describes a starting distance upper bound of four, while the current released YAML begins with distance two through six. This implementation uses three through four, preserving the local engine's general-separation assumptions. The current repository also contains a policy-guidance regularizer. This implementation follows the paper's entropy objective. The truncation implementation explicitly bootstraps the final observation and retains that transition for training.

CPU execution checks establish functional correctness. Reproducing the reported playing strength remains a training and evaluation milestone at the required compute scale.

## Deployment

The Python strategy reads the existing initialization and observation protocol and writes five-integer actions. In the graphical application's Player field, a command can be:

```text
python src/agents/nebula/agent.py runs/nebula/checkpoint.pt
```

Quote each executable or file path that contains spaces. Relative paths resolve from the application's working directory. An explicit `--library PATH` selects an arena library in a custom build directory. The model input size determines the largest supported board for its checkpoint, and the strategy reports larger boards through `stderr`.

For C++ inference, export the EMA model:

```powershell
python src/agents/nebula/export.py runs/nebula/checkpoint.pt runs/nebula/policy.pt
```

The optional `nebula` target uses LibTorch. With a PyTorch installation that includes C++ development files, its CMake prefix can be discovered directly:

```powershell
$torchPrefix = python -c "print(__import__('torch').utils.cmake_prefix_path)"
.\scripts\build.bat -DNEBULA_BUILD_AGENT=ON "-DCMAKE_PREFIX_PATH=$torchPrefix"
```

The corresponding Player command is:

```text
build/nebula/nebula.exe runs/nebula/policy.pt
```

The C++ target uses TorchScript as the currently tested bridge to LibTorch. PyTorch classifies TorchScript as a legacy export path. Training remains ordinary PyTorch, so the export boundary can later move to a newer runtime. Python and C++ inference share the same feature encoder and quantization step.

## Sources

- [2026 paper](2606.23348v1.pdf)
- [Author repository](https://github.com/strakam/AverageJoe)
- [Released model configuration](https://github.com/strakam/AverageJoe/blob/main/configs/custom/L_7d_gae90.yaml)
- [Observation and action encoding](https://github.com/strakam/AverageJoe/blob/main/networks/common.py)
- [PyTorch C++ loading](https://docs.pytorch.org/tutorials/advanced/cpp_export.html)
