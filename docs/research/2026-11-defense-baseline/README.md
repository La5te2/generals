# 防守改进尝试与基线评估归档

整理日期：2026-11-02。对应 [方向清单 #1](https://github.com/La5te2/generals/issues/1)、[阶段进展 #2](https://github.com/La5te2/generals/issues/2)、[四方向过程 #3](https://github.com/La5te2/generals/issues/3)。

接手后依次尝试B1、B2、B4、B12，均未取得替换原33模型的采用依据；随后补测固定基线。本目录保存过程、结论边界和精选证据，基线工具位于 [experiments/baseline](../../../experiments/baseline/README.md)。

- [四个方向的过程与未采用原因](attempts.md)：动机、具体修订、主要结果及停止依据。
- [基线协议、结果与复现](baseline.md)：七组共1,600局正式评估，包含P2追加400局和P6。
- [本次整理验证记录](validation.json)：离线核验、8项测试、P2/P5各两局完整对照与200tick适配器检查。
- [证据清单与模型身份](manifest.json)：原始文件/摘录的SHA256、三份模型的SHA256、依赖版本和代码版本。
- [基线逐局数据](data/baseline/)：保留原结果和协议的字节，不复制重复的合并结果。
- [四方向精选结果](data/attempts/)：按字段摘录，保留原文件SHA256；不等同于完整训练档案。

从仓库根目录执行，Python标准库即可核对基线计数、地图配对、扩样去重和模型身份，并复算区间：

```sh
python3 experiments/baseline/verify.py
python3 -m unittest discover -s experiments/baseline -p 'test_*.py' -v
```

这两个命令不加载模型、不训练、不启动对战。全部七组基线的胜负和共享地图比较可独立复算；四方向仅验证已导出文件的完整性，不在本PR重跑训练和探针拟合。

目前没有候选权重替代原33。原模型相对ppo20的胜率为349/600（58.17%），这是既有模型间的表现差异；本阶段的新增成果包括测量工具、实验结果和诊断，不能将该差异计作本阶段算法提升。ppo20与33的训练来源关系仍缺少33元数据中的`source_sha256`直接证明。

33与39在共享100张图上对ppo20战绩相同，且直接互战接近50%。这支持“本次未观察到进一步收益”，不构成策略等价检验，也没有证明四个方向未采用的共同原因就是饱和。

完整checkpoint、逐tick轨迹、旧源码快照和历史部署记录继续保存在维护者本地归档。模型获取方式及复现限制见 [基线说明](baseline.md)。
