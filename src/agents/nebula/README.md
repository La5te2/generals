# Nebula

Nebula implements the agent described in the 2026 paper. PyTorch handles the neural network and optimization, while `generals-bots` supplies batched self-play games through JAX. During sampling, `train.py` runs the model to select actions. It then uses the collected games to update the model through PPO. Training uses `features.py` and C++ deployment uses `features.hpp` to encode the same information from each player's own field of view.

## Running

Copy this directory to the training machine. Keep its Python files, `requirements.txt` and the appropriate launch script; `infer.cpp` and `features.hpp` are not needed for training. No other repository files or C++ build are required. Git must be installed because `requirements.txt` pins `generals-bots` to a Git commit.

Run the following commands from inside the copied directory, using the Python environment selected for training:

```sh
python -m pip install -r requirements.txt
```

On Linux or macOS, a short CPU execution check is:

```sh
bash train.sh --profile check --device cpu
```

On Windows:

```powershell
python -B train.py --profile check --device cpu
```

`check` uses a small network and two brief updates to exercise sampling, optimization and saving. The default profile, `paper`, selects the seven-layer model and large rollout configuration. The scripts use `python3` on Unix and `python` on Windows; set `PYTHON` to an executable path to override this. They preserve the calling directory, so relative output and resume paths resolve from where the command is run.

For NVIDIA GPU training on Linux, install a CUDA-enabled PyTorch build using the [PyTorch installation selector](https://pytorch.org/get-started/locally/), and a compatible CUDA-enabled JAX build using the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html). Both libraries must have GPU support. For a CUDA 13-compatible machine, the JAX command is:

```sh
python -m pip install "jax[cuda13]>=0.11.2,<0.12"
python -c "import torch, jax; print('PyTorch CUDA:', torch.cuda.is_available()); print('JAX devices:', jax.devices())"
bash train.sh --device cuda --output runs/nebula
```

The check should report PyTorch CUDA availability and a JAX GPU device. Native Windows and macOS can run the CPU path; the CUDA command above targets Linux. `--device auto`, the default, selects an available device; `--device cuda` requires GPU support and reports an error if it is missing.

To resume training:

```sh
bash train.sh --device cuda --resume runs/nebula/checkpoint.pt --updates 100000
```

The output directory contains `checkpoint.pt` and `metrics.jsonl`. A checkpoint stores model weights, EMA weights, optimizer state, update count, random state and curriculum stage. Resuming begins fresh games at the saved stage and continues the optimization schedule. `--updates` specifies the total target update count.

The paper-sized rollout contains $512\times512\times2$ player transitions. Its compressed observation buffer uses about 5.09 GiB and stays in CPU memory by default. `--storage device` keeps it on the training device instead. The buffer stores shared history entries once, packs binary markers into bits and reconstructs the original inputs for each selected minibatch without changing their precision. On CUDA, only that minibatch's host data is pinned for transfer. The buffer is reused between updates; scalar training data, model activations, optimizer state and environment data require additional memory. The default execution check uses about 0.71 MiB for the observation buffer.

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
\beta_k=\frac{0.05}{(k+1)^{0.2}}.
$$

EMA updates once per training iteration with decay $0.999$. Deployment uses these averaged weights.

## Curriculum and Scope

The first stage uses general-to-general BFS distance 3 through 4. Later stages use 4 through 8, 6 through 13, 11 through 17 and 17 through 28. Evaluation against a random legal mover gates advancement at a win rate of $0.6$. This evaluation measures curriculum readiness. Competitive strength calls for separate matches against capable opponents and authorized ladder play.

The current implementation runs the JAX environment and PyTorch model on one selected CPU or GPU device, sharing arrays through DLPack. `env.py` generates a reusable map pool through `generals-bots` and refreshes it periodically or when the curriculum advances. Ongoing games retain their boards, while completed games restart from the current pool. The environment retains neutral cities, fog of war and general trades. The paper's four-device training remains a separate extension.

The network, observation dimensions and principal PPO settings match the paper and the corresponding released configuration. The paper describes a starting distance upper bound of four, while the current released YAML begins with distance two through six. This implementation uses three through four, preserving the local engine's general-separation assumptions. The current repository also contains a policy-guidance regularizer. This implementation follows the paper's entropy objective. The truncation implementation explicitly bootstraps the final observation and retains that transition for training.

CPU execution checks establish functional correctness. Reproducing the reported playing strength remains a training and evaluation milestone at the required compute scale.

## Deployment

The deployed agent runs through `infer.cpp`, using LibTorch on CPU with one computation thread. It reads initialization and observations through the process protocol and replies with five-integer actions. To prepare its model, export the EMA weights:

```sh
python -B export.py runs/nebula/checkpoint.pt runs/nebula/policy.pt
```

`checkpoint.pt` resumes Python training; `policy.pt` is the exported TorchScript model loaded by the C++ executable. Exporting needs only this folder and its Python dependencies. Copy the resulting `policy.pt` to the deployment machine.

Building the executable requires the full repository. The `nebula` target builds by default and uses LibTorch. CMake first searches the configured library paths, then queries the selected Python interpreter for PyTorch's CMake package location. A separate LibTorch installation can be selected through `CMAKE_PREFIX_PATH` or `Torch_DIR`.

From the repository root, build the executable on Windows:

```powershell
.\scripts\build.bat -DNEBULA_BUILD_AGENT=ON
```

On Linux or macOS, a deployment-only build uses:

```sh
bash scripts/build.sh -DNEBULA_BUILD_INTERFACE=OFF -DNEBULA_BUILD_AGENT=ON
```

The corresponding Player command on Windows is:

```text
build/nebula/nebula.exe "path/to/policy.pt"
```

Linux and macOS use `build/nebula/nebula` as the executable path. Quote each executable or model path that contains spaces. Relative paths resolve from the application's working directory. The model input size determines the largest supported board for its checkpoint, and the agent reports larger boards through `stderr`.

TorchScript carries the exported model from PyTorch to LibTorch. The JAX training encoder and C++ deployment encoder produce matching features, and both paths apply the same board quantization before model evaluation. Deployment uses the executable, model file and LibTorch runtime libraries.

## Sources

- [2026 paper](https://arxiv.org/abs/2606.23348)
- [Author repository](https://github.com/strakam/AverageJoe)
- [Released model configuration](https://github.com/strakam/AverageJoe/blob/main/configs/custom/L_7d_gae90.yaml)
- [Observation and action encoding](https://github.com/strakam/AverageJoe/blob/main/networks/common.py)
- [PyTorch C++ loading](https://docs.pytorch.org/tutorials/advanced/cpp_export.html)
