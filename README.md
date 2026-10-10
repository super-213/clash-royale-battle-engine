# Clash Royale Battle Engine

这个仓库允许你把皇室战争的原生战斗引擎当成 API 来调用，极大方便了 AI 训练。同时，为了方便你自己配置这个仓库，我在开发过程中把 agent 总结的经验蒸馏成了 3 个 skill，放在 `.agents/skills` 目录下。

本仓库是对 [FirstLight-CR](https://gitlab.com/firstlight3/FirstLight_CR) 逆向工作加以改进后的产物，极大简化了项目结构和通信接口，把代码从上万行压缩到几百行。如果你也希望训练 AI 来玩皇室战争，这个项目提供的干净接口非常适合作为你的训练起点。

强化学习训练现在可直接使用当前仓库的 `NativeBatchVecEnv`：一个原生进程、
一条 batch 连接同时推进 16 场战斗，并在单场结束时只复位对应槽位。观测已经按
CNN 空间张量、实体 Transformer token、LSTM 时序历史和 masked hierarchical action
head 拆分；对应的可训练 PPO policy 也位于当前仓库。接口与验证命令见
[VEC_ENV.md](VEC_ENV.md)。

默认正式训练策略已经升级为 Auxiliary World-Model RecurrentPPO。手牌 token 会查询
实体、历史和对手 belief，历史 token 会查询当前实体；CNN 始终保留原生 `32×18`
空间 feature map，动作头让每张牌直接查询576个空间位置并生成独立热力图。随后
Actor/Critic LSTM hidden state 跨决策持续传递，单场结束时只重置对应 VecEnv 槽位，
并使用 sequence minibatch 和 truncated BPTT。`native_model_v3` 还提供塔血、圣水、
单位生成/死亡、战斗血量、目标和出牌等逐决策变化特征。

第六阶段的辅助世界模型也已接入默认 PPO：共享编码器会同时预测下一决策的塔血变化、
实体死亡与位置、伤害来源类别、对手公开圣水区间、可能手牌/下一张牌，以及默认约
2秒后的局面价值。辅助任务只增加训练 loss，不替代 PPO Actor/Critic，也不会把未来
标签加入推理观测。默认 v6 还将实际成功执行的卡牌与落点加入动力学辅助预测，
当前对手信息推断保持纯状态输入；旧 v5 类保留用于加载、续训和消融。

`train_native.py` 与一键入口共用默认 v6 profile、训练日志、自博弈和固定评估流程。
默认使用 M5 Pro 的 `mps` GPU；MPS 不可用时立即报错。两个入口均支持 `--resume`，
恢复模型、优化器、历史对手池、评级和 RNG；续训从新对局开始。操作见 [TRAINING.md](TRAINING.md)。

推荐通过交互式一键入口开始正式训练：

```bash
./train.sh
```

入口会询问训练配置、轮次、16 路并行环境、MPS/CPU、随机种子、检查点间隔和
可选 PPO 高级参数。每次运行都会在 `artifacts/training_runs/` 下创建独立目录，
实时保存 `console.log`、`status.json`、JSONL/CSV 指标、TensorBoard 数据、模型
检查点、最终模型、训练摘要以及损失、奖励和吞吐率 SVG 曲线。Agent 可直接读取
`artifacts/training_runs/latest.json` 指向的 `status.json`，无需解析终端滚动输出。

默认训练配置已经启用自博弈联赛：0% 合法随机对手、30% 当前策略镜像、70%
历史快照。每局固定一个对手，按设置的轮次间隔发布不可变模型快照。历史池最多
保留 20 个快照：新快照先作为挑战者进入池，满额时根据自博弈 Elo 淘汰评级最低的
旧快照；活跃对手从较强快照和最新快照中挑选。仍在进行对局的淘汰快照会在对局
结束后删除。运行目录的 `self_play/` 保存逐局结果、Elo、matchup 矩阵和对手池。

训练还会运行独立固定评估赛程：round 0、周期检查点和最终模型始终面对相同 seed、
相同双方角色以及固定的随机/round-0 对手。默认保存各轮更新后的快照，在训练结束后
依次评估，避免重置进行中的训练对局。评估不进入 PPO rollout，
结果写入 `evaluation/`，并生成 `charts/fixed_evaluation.svg`。因此跨轮次、跨模型比较
应优先看固定评估的 `score_rate`，而不是随自博弈对手池共同变化的训练胜率。

模型不是写死在脚本中的。复制 `training_profiles/native_hybrid_ppo.json` 并修改
`policy` 的 `module:Class` 即可接入同一 observation/action schema 的新模型；更换
算法、环境或完整训练流程时，分别修改 `algorithm`、`environment` 或 `trainer`。
`trainer` 接口为 `trainer(config: dict, run_dir: pathlib.Path)`，因此其他框架也能
继续使用同一个交互入口和运行目录约定。

`./train.sh` 会自动启动已有的 `Pixel_10_Pro` 模拟器并初始化引擎；训练结束或中断后，
可单独关闭模拟器：

```bash
./stop_emulator.sh
```

测试战斗引擎的速度（下面的数据是在我的笔记本电脑运行的结果，只运行1个战斗所以速度较慢）：
```
python -m clash-royale-battle-engine.benchmark

result=1 crowns=(0, 1) final_tick=3835
attempts=62 executions=62 wire_bytes=475842
reset=0.035s rollout=0.376s total=0.411s
simulated=187.2s speed=497.9x real-time
ticks_per_second=9957.5
```
