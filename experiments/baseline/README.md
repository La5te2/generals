# 固定模型基线评估

配套文档：[研究归档](../../docs/research/2026-11-defense-baseline/README.md)、[协议与结果](../../docs/research/2026-11-defense-baseline/baseline.md)。工具仅做冻结策略评估，复用现有`agents/1014`。规则bot适配器从合法输入重建观测，使用固定依赖中的Random、Expander、Hunter；不向bot提供隐藏状态。

## 先核对已有结果

在仓库根目录执行，不需要权重或第三方包：

```sh
python3 experiments/baseline/verify.py
python3 experiments/baseline/verify.py --json
python3 -m unittest discover -s experiments/baseline -p 'test_*.py' -v
```

核验包括20份证据文件的SHA256、七组共1,600条逐局记录、座位配对、地图一致性、扩样去重、模型身份、胜率及名义区间；合并P2在内存中计算。命令只读归档。

## 配置CPU环境

使用Python 3.12，从仓库根目录执行：

```sh
python3.12 -m venv /tmp/generals-baseline-venv
/tmp/generals-baseline-venv/bin/python -m pip install -r experiments/baseline/requirements.txt
```

核心数值包固定到原评估使用的版本，模拟器/bot依赖固定到Git提交。完整历史版本和原模型SHA256见归档`manifest.json`。从维护者取得三份快照，放在同一个目录，例如`/tmp/generals-baseline-models/`。

评估器在导入前检查模型和`agents/1014`源码hash；后续模型/代码版本若不同，不能冒称此次历史重跑。仓库更新后需要使用本PR对应的提交。显式传入`--checkpoint`可以评估同架构的新学习方快照，输出标记`custom_learner=true`并记录实际hash；参照对手仍须匹配原快照。

## 逐配对复现

先做两局完整对战，检查安装和模型加载；它们不替代正式评估：

```sh
JAX_PLATFORMS=cpu /tmp/generals-baseline-venv/bin/python experiments/baseline/run.py \
  --pair P5 --models-dir /tmp/generals-baseline-models \
  --games 2 --lanes 2 --seed 930001 --output runs/baseline-reproduction/smoke-P5.json
```

对初批P1—P5分别使用默认200局、seed930001运行；P6同样200局、seed930001。例如：

```sh
JAX_PLATFORMS=cpu /tmp/generals-baseline-venv/bin/python experiments/baseline/run.py \
  --pair P2 --models-dir /tmp/generals-baseline-models \
  --output runs/baseline-reproduction/P2-first.json

JAX_PLATFORMS=cpu /tmp/generals-baseline-venv/bin/python experiments/baseline/run.py \
  --pair P2 --models-dir /tmp/generals-baseline-models --games 400 --seed 930002 \
  --output runs/baseline-reproduction/P2-second.json

JAX_PLATFORMS=cpu /tmp/generals-baseline-venv/bin/python experiments/baseline/run.py \
  --pair P6 --models-dir /tmp/generals-baseline-models \
  --output runs/baseline-reproduction/P6.json
```

| 配对 | 学习方 | 固定对手 |
| --- | --- | --- |
| P1 | 33 | 39 |
| P2 | 33 | ppo20 |
| P3 | 33 | Random |
| P4 | 33 | Expander |
| P5 | 33 | Hunter |
| P6 | 39 | ppo20 |

每局交换座位的一半与另一半属于同一地图，不能计成两张独立地图。对战仍按真实tick维护原8帧空间历史和512步公开历史，保持原合法动作掩码和终局规则。

新输出包含逐局记录和双方模型身份，拒绝覆盖已有文件。`verify.py`核对的是随仓库提供的历史证据，不自动认可新的复现文件；比较新输出时须核对seed、完整地图hash与座位，以及逐局胜负/步数。CPU耗时取决于硬件，完整面板不是快速测试。

适配器还可单独跑200tick随机自对战，检查回退率、动作合法性及实现一致性：

```sh
JAX_PLATFORMS=cpu /tmp/generals-baseline-venv/bin/python experiments/baseline/check_adapter.py \
  --output runs/baseline-reproduction/adapter.json
```

四个改进方向的历史训练代码、全部checkpoint与轨迹不在此目录内；本次交付提供它们的过程说明和结果摘录，不能仅凭本PR重训四个方向。
