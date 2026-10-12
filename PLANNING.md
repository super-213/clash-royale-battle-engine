# 多步世界模型与搜索训练

`native_planning` 是独立于 PPO 的实验训练路线：公开观测编码 → 随机空间动力学
→ sampled PUCT → 真实引擎 → 序列回放 → 搜索策略、价值与动力学训练。
默认 `native_hybrid_ppo` 不变；已有 v5/v6 checkpoint 继续用原来的类加载。

## 启动与恢复

```bash
./train.sh --profile native_planning

# 只检查配置，不启动模拟器
./train.sh --profile native_planning --defaults --dry-run

# 引擎已启动时的直接入口；默认使用 MPS，CPU 必须显式指定
conda run --no-capture-output -n clash-royale python train_native.py \
  --profile native_planning --output artifacts/planning_runs/run01

conda run --no-capture-output -n clash-royale python train_native.py \
  --resume artifacts/planning_runs/run01/final_model.zip \
  --rounds 20 --output artifacts/planning_runs/resumed01
```

复制 `training_profiles/native_planning.json` 可修改完整设置。规划专用参数包括：

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `planning_budget` | 64 | 每场、每次真实决策最多调用多少次模型转移；0 为纯策略提议 |
| `planning_depth` | 6 | 最大未来决策层数；默认每层0.5秒 |
| `planning_candidates` | 16 | 根节点候选上限，始终保留等待和高概率动作 |
| `planning_internal_candidates` | 8 | 内部节点候选上限 |
| `planning_outcome_samples` | 2 | 每个动作的随机结果样本数；重复样本保留其权重 |
| `planning_chance_codes` | 16 | 学习型公开结果代码数 |
| `planning_warmup_rounds` | 2 | learner 先不搜索，采集初始模型数据的轮数 |
| `planning_unroll` / `planning_n_steps` | 5 / 5 | 预测状态连续训练长度 / 真实回报 bootstrap 长度 |
| `planning_burn_in` | 16 | 从回放历史重建循环状态的前缀长度 |
| `planning_batch_size` / `planning_updates` | 8 / 64 | 每次序列数 / 每轮网络更新数 |
| `planning_replay_capacity` | 8192 | 回放转移数上限；原始观测与后继观测约占1GB，另有对象和模型开销 |
| `planning_reanalyze_every` | 8 | 每多少次更新重新搜索一个已存根节点；0关闭 |
| `planning_teacher` | null | 可选 v6 checkpoint，只复制兼容编码器权重 |

`recurrent_enabled=false` 只表示不走 SB3 RecurrentPPO 配置；规划器有自己的持续 GRU。
PPO 的 `epochs_per_round`、entropy 和 clip 参数不用于规划学习，更新量由规划专用参数控制。
训练预算仍按真实 learner 决策计数；搜索、重新搜索和评估不计入真实交互预算。

## 模型与搜索语义

- `PlanningModel.encode` 使用已有全分辨率 CNN、实体 Transformer、历史和 belief 编码，
  加 GRU 更新真实历史状态。树状态保留全局记忆、32×18空间特征、己方手牌集合、下一张牌、
  标量和公开 belief。树内不重新调用原生环境，也不修改真实记忆。
- `dynamics(state, own_action, opponent_action, outcome_code)` 输出可继续推进的下一树状态，
  奖励、终局、执行成功、空间占用、下一张牌和部署几何预测。
- 动作使用固定卡牌身份和己方视角格子，避免跨换牌后手牌槽位含义变化。
  等待为0，其余为 `1 + (card_token-1)*576 + y*18+x`。原生提交坐标和编码共用格子中心。
- 时间、圣水回复、成功出牌后的消耗、移除已打卡牌及抽入已知下一张牌按规则更新。
  成功与否、后续未知抽牌、未来部署区域及战斗后果由模型预测；未实现完整精确卡序枚举。
- 训练后验读取真实公开后继表示，生成 straight-through 随机代码；在线只使用先验。
  balanced KL 约束后验与先验。代码代表公开结果分支，不能把训练后验当在线输入。
- 当前对手动作分布只读取当前公开状态，不读取我方同时提交的动作。动力学才读取联合动作。
  对手执行动作监督来自原生成功回执和公开事件，不向 learner 提供对手真实隐藏手牌。
- 搜索用有限结果样本近似对手/结果期望。每个动作的样本库达到上限后均匀复用样本，
  使有限预算能探索多层。不是逐层取最坏对手，也不是完整信息集 MCTS 或 Nash 求解器。
- 候选按卡牌分层，空间去重并包含合法探索点；限制落点后重新分配该卡牌的空间概率，
  保留卡牌总概率，防止等待因候选落点少而虚高。策略训练使用同一候选集合上的条件分布。
  这是受限候选搜索，不宣称对完整动作空间无偏。
- 整棵树保持根玩家收益视角；按 `r + gamma * continuation * value` 回传，不按树深翻转符号。
  终局停止展开；只有根节点动作被提交，下一次真实观测到来后重新规划。

所有未来状态、合法区域、对手行为与价值都是近似预测。当前方案通过公开结果树避免给
策略显式输入隐藏真值；它不包含完整对手手牌粒子滤波，也不保证 belief 已校准。

## 学习与持久化

`PlanningLearner` 用真实动作序列连续推进预测状态，不在每步用真实观测替换预测输入。
目标网络从真实观测历史编码后继表示，提供停止梯度的一致性目标和真实回报 bootstrap。
损失包括状态一致性、随机代码 KL、奖励、终局、成功执行、下一张牌、标量、公开 belief、
空间占用、部署区域、对手行为、搜索策略和价值。未直接复用 v6 逐实体辅助头；v6 实现保留。

样本按单场轨迹采集；结束前的真实后继在自动 reset 之前保存。样本不会跨终局。
长历史训练采用有限 burn-in，因此是截断历史近似。每轮更新之后，在线记忆从当前对局的
真实观测历史重建，避免把旧网络 hidden state 直接交给更新后的网络。

`final_model.zip` 和检查点是本路线的 Torch archive，不是 SB3 PPO ZIP，必须通过
`native_engine.planning_policy:NativePlanningAgent` 加载。普通推理不加载回放和优化器。
`.zip.state/` 保存目标网络、优化器、回放、RNG、下一个原生 seed、联赛及原始基线。
恢复时核对模型、训练状态、配置、联赛和基线哈希，从新原生对局继续；不拼接旧的未完轨迹。
中断时保留最近一次完整轮次的检查点；不把半次优化发布成完成检查点。

`status.json` 与 `metrics.jsonl` 记录真实决策、搜索转移数、达到的深度、成功出牌数和各损失。
当前没有为该路线生成 PPO TensorBoard/损失 SVG；固定评估仍生成原有赛程文件与图表。

## 冻结多步预测验证

在独立 seeds 上采集冻结策略轨迹，之后离线比较不同模型。采集不更新参数，拒绝与所给
config 的完整训练范围及固定评估 seed 重叠。实验自定义的其他保留范围也应由调用者避开。

```bash
conda run -n clash-royale python -m native_engine.planning_validation collect \
  --checkpoint artifacts/planning_runs/run01/final_model.zip \
  --config artifacts/planning_runs/run01/config.json \
  --output artifacts/planning_runs/heldout.pt \
  --decisions 1024 --battles 4 --seed 2900001 --search-budget 0

conda run -n clash-royale python -m native_engine.planning_validation evaluate \
  --checkpoint artifacts/planning_runs/run01/final_model.zip \
  --replay artifacts/planning_runs/heldout.pt \
  --output artifacts/planning_runs/multistep.json \
  --horizons 1 4 8 --samples 64 --outcomes 4
```

输出各 horizon 的空间误差、塔血误差、圣水误差、累计奖励误差、终局 Brier 与合法性 IoU。
模型使用先验采样并连续推进；评分沿相同的已记录双方动作进行。它不是反事实排序验证，
也不能据此证明搜索提高胜率。可以读取训练 sidecar 的 `training.pt` 做诊断，但该数据不算保留验证。

## 相同权重的搜索强度对比

```bash
conda run -n clash-royale python fixed_evaluation.py \
  --config artifacts/planning_runs/run01/config.json \
  --checkpoint artifacts/planning_runs/run01/final_model.zip \
  --output-dir artifacts/planning_runs/eval_no_search --search-budget 0

conda run -n clash-royale python fixed_evaluation.py \
  --config artifacts/planning_runs/run01/config.json \
  --checkpoint artifacts/planning_runs/run01/final_model.zip \
  --output-dir artifacts/planning_runs/eval_search --search-budget 64 --search-depth 6
```

覆盖只作用于被评估 learner；冻结对手保留自己的 checkpoint 配置。
`evaluation/planning_search.json` 记录实际搜索设置。逐局与汇总输出增加平均搜索转移数、
最大搜索深度和按 batch 槽位分摊的预测耗时。对比时固定权重、seeds、对手及角色，报告
胜率和耗时；深度上限不代表每个分支实际达到该深度。

## 验证边界

测试覆盖延迟收益下的多步决策、预算与终局、动作镜像、卡牌概率质量、资源规则、循环状态
重置、多步梯度、重新搜索、终局标签、联赛、保存/恢复、冻结数据采集及固定评估。

2026-10-12 最终版本已用匹配的 Android `libg.so` 运行原生短训练及续训，共192个 learner 决策，
22次提交出牌全部成功，搜索达到4层；另采集128个独立 seed 决策用于多步评分。
默认256维记忆、64通道空间网络也完成了16路 MPS 原生短训练，共64次决策、2次网络更新，
搜索达到5层，损失与梯度有限并成功保存。
同一短训权重在固定 seed 双角色赛程上分别关闭搜索和使用8次转移预算，各完成2场真实比赛，
均无超时与非法动作；开启搜索的赛程实际达到3层。样本量不足以支持胜率提升结论。
完整比赛验证同时发现并修复了单槽位 resident capture 重发当前 epoch 已交付事件的问题：
`Batch` 按原生 sequence 过滤已交付事件，并拒绝 cursor 回退，保留同 tick 不同 sequence 的合法事件。
这些是接口与数值验证，数据量不足以训练准确的战斗预测器或证明强度提升。
长训、完整比赛的多种子胜率与搜索预算收益需要另外运行，不应把短测指标当作收敛结果。
