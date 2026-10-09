# 皇室战争智能体模型架构与演进路线

## 1. 文档目的

本文记录当前皇室战争智能体的模型结构、CNN、Transformer、LSTM 三个编码器的
职责与配合方式，并定义后续训练、评估和模型演进路线。

当前实现基于：

- 16 路原生 `NativeBatchVecEnv`；
- Stable-Baselines3 PPO；
- Apple Silicon MPS 训练；
- `native_model_v2` observation schema；
- Cross-attention + Entity Transformer + LSTM 混合策略网络；
- 分层、带合法性掩码的动作分布。

相关实现：

- [`native_model.py`](../native_model.py)
- [`native_cross_modal.py`](../native_cross_modal.py)
- [`native_spatial_model.py`](../native_spatial_model.py)
- [`native_world_model.py`](../native_world_model.py)
- [`native_recurrent_model.py`](../native_recurrent_model.py)
- [`native_vec_env.py`](../native_vec_env.py)
- [`training_runtime.py`](../training_runtime.py)
- [`training_profiles/native_hybrid_ppo.json`](../training_profiles/native_hybrid_ppo.json)

## 2. 当前模型总体结构

当前实现已经从三路摘要直接拼接升级为 token 级跨模态推理，再由持续 LSTM 跨决策
维护状态：

```text
场上实体 ─ Entity Transformer ───────────────────────────┐
       ▲                  ▲                              │
       │ history→entity  │ hand→entity/history/belief   │
历史 token ─ Cross-Attention ─ Event LSTM ──────────────┤
belief token ─ Cross-Attention ─────────────────────────┤
手牌 token ─ Cross-Attention ───────────────────────────┤
战场网格 ─ CNN ─ 32×18 full-resolution spatial tokens ──┤
连续变化、圣水、时间、塔血 ─ MLP ───────────────────────┤
                                                        ▼
                                                Fusion MLP（256维）
                                                        │
                                      Persistent Actor/Critic LSTMs
                                                        │
                    ┌───────────────────────────────────┴──────────┐
                    ▼                                              ▼
      card query × 32×18 spatial tokens                    Critic 局面价值
                    │
       P(出哪张牌) × P(落点 | 牌, 场面, 历史, belief)
                    │
                    └──── shared latent + executed card/placement → auxiliary dynamics
```

三类编码器分别解决三个不同问题：

| 编码器 | 主要输入 | 主要职责 |
|---|---|---|
| Full-resolution CNN + spatial query | `16 × 32 × 18` 空间张量 | 保留每个位置，让不同卡牌生成独立落点热力图 |
| Transformer + cross-attention | 最多128个实体 token | 对象关系以及手牌、历史、belief 与实体的直接交互 |
| Event LSTM | 查询过当前实体的最近32次公开出牌事件 | 显式出牌顺序、来源归因、节奏和短期事件摘要 |
| Persistent LSTM | 连续决策状态 | 跨决策保留完整局面演化记忆 |

## 3. CNN：战场空间编码

### 3.1 输入

CNN 接收 `spatial` 观测，形状为：

```text
16 channels × 32 rows × 18 columns
```

空间通道描述：

- 双方单位占据位置；
- 单位生命比例；
- 兵种、建筑和投射物类型；
- 部署剩余时间；
- 攻击冷却；
- 是否已经锁定目标。

### 3.2 网络

当前空间编码器包含：

- 普通卷积；
- GroupNorm；
- SiLU；
- 两个不改变高宽的 Residual Blocks；
- `1 × 1` token projection；
- 二维坐标编码和逐位置 LayerNorm。

CNN 不再下采样并压缩成一个全局向量。v4 空间编码器全程保留原生
`32 × 18 = 576` 分辨率，使用卷积和 Residual Blocks 建立具有局部感受野的 feature map，
再为每个位置加入二维坐标编码。576个 spatial tokens 被直接送入动作头；不经过全局
池化后重建，也不依赖从 `8 × 5` 粗特征图插值得到落点表示。

### 3.3 作用

CNN 主要学习：

- 哪一路正在形成压力；
- 单位是否聚集；
- 桥头、塔下等局部区域的阵型；
- 防守单位和进攻单位的空间接触关系；
- 哪些区域适合作为部署位置。

CNN 对局部空间模式非常有效，但同一网格中的多个实体会发生信息聚合，因此需要
Transformer 保留对象级信息。

## 4. Transformer：实体关系编码

### 4.1 输入

每个公开实体被编码为一个24维 token，最多128个 token。字段包括：

- 敌我身份；
- 实体类型；
- 卡牌身份；
- 归一化坐标；
- 当前生命值；
- 部署剩余时间；
- 攻击冷却；
- 是否存在攻击目标；
- 目标相对位置；
- 是否为法术。

`entity_mask` 用于屏蔽补齐 token。

### 4.2 网络

当前实体编码器使用：

- 64维 token projection；
- 2层 Transformer Encoder；
- 每层4个 attention heads；
- 256维 feed-forward；
- 可学习 query 的 attention pooling。

实体 Transformer 保留所有有效实体 token，同时额外产生一个 attention pooling 摘要。
实体 token 不再只在池化后参与融合，而是作为历史、belief 和手牌 query 的 key/value。

### 4.3 作用

Transformer 主要学习：

- 哪个单位正在攻击哪个目标；
- 前排和后排单位是否构成组合；
- 哪个对象对防御塔威胁最大；
- 跨路或远距离单位之间的关系；
- 当前决策应重点关注哪些实体。

CNN 看的是局部地图模式，Transformer 看的是具体对象以及对象之间的全局关系，
两者具有互补性。

## 5. LSTM：显式事件摘要与持续时间记忆

### 5.1 输入

LSTM 接收最近32次成功的公开出牌事件，每个事件包含：

- 是否为有效事件；
- 出牌方；
- 卡牌身份；
- 部署坐标；
- 出牌时间；
- 距离当前时刻的时间；
- 卡牌费用。

### 5.2 Event LSTM

每个事件先投影为 token，并以历史 token 为 query、当前实体 token 为 key/value 做
cross-attention，再进入单层 LSTM。这样历史事件能够先判断“刚才那段出牌序列现在
对应了哪些仍在场的单位”，然后形成历史摘要。

### 5.3 Persistent Actor/Critic LSTM

融合后的256维状态会进入持续 LSTM。Actor 和 Critic 默认分别使用一层、256 hidden
size 的 LSTM。hidden/cell state 在每个决策之间持续传递，并由 RecurrentPPO 保存到
rollout buffer。

16路环境分别拥有独立状态。某一场对局结束时，`episode_start` 只重置该槽位，其他
仍在进行的战斗不会丢失记忆。训练使用 sequence minibatch、padding mask 和默认
32步 truncated BPTT。

### 5.4 连续变化输入

`native_model_v2` 新增20维 `temporal` 向量，记录相邻决策之间的：

- 双方塔血变化；
- 己方真实圣水与对手公开推算圣水变化；
- 双方单位生成和死亡；
- 双方战斗单位总血量变化；
- 双方出牌数和皇冠变化；
- 锁定目标数量变化；
- 实体数量变化和经过的 native ticks。

### 5.5 作用和边界

LSTM 主要学习：

- 对手最近使用了哪些牌；
- 双方最近在哪一路投入资源；
- 当前是进攻、防守还是转换阶段；
- 对手是否刚用掉关键卡；
- 当前战场是怎样逐步形成的。

Event LSTM 仍会重新编码最近32次公开出牌，用于提供明确、可检查的事件上下文；
Persistent LSTM 则负责跨决策保存更完整的隐状态。二者职责不同。

LSTM 本身也不会“记住未来”。未来回报目前由 PPO Critic 估计。显式预测未来需要
额外的 dynamics/world model 或辅助预测任务。

## 6. 其他上下文编码

三个主编码器以外，模型还使用：

### 6.1 手牌编码与查询

当前四张手牌和下一张牌分别形成 token。每张牌直接查询实体、历史摘要和 belief token，
因此不同卡牌可以关注不同威胁；前四个手牌 token 会继续送入动作头，下一张牌只参与
局面融合。

### 6.2 当前局面标量

12维标量包含：

- 己方圣水；
- 从公开事件重建的对手圣水；
- 双方皇冠数；
- 对局时间；
- 单倍、双倍、三倍圣水阶段；
- 玩家方向；
- 双方塔血比例；
- 历史缓冲区使用程度。

这些标量由 MLP 编码为32维。

### 6.3 对手卡序 belief

模型不会直接读取对手隐藏手牌。环境根据公开出牌序列枚举可能卡序，构造：

- 对手每张牌在手牌中的概率；
- 对手下一张牌的概率；
- 卡序推断置信度；
- 双方距离上次出牌的时间。

belief 被投影为 token，并查询当前实体与历史摘要。随后手牌 token 又能查询这个已经
结合当前场面的 belief token，使“对手可能如何反制”直接进入出牌和落点判断。

## 7. 跨模态融合、Actor 和 Critic

当前仍保留一个最终 Fusion MLP，但送入它的已不是互不交流的三份摘要，而是已经经过
cross-attention 的条件化表示：

```text
history ──query──> entities
belief  ──query──> entities + history
cards   ──query──> entities + history + belief
cards   ──query──> 32×18 spatial tokens
                         │
              条件化摘要 + scalar + temporal
                         ▼
                   256维融合状态
                         │
                Persistent LSTMs
                         │
                ┌────────┴──────────────┐
                ▼                       ▼
   card-conditioned spatial actor     Critic
```

- Actor 使用持续 LSTM 上下文调制每张手牌 token，再直接查询576个全分辨率 spatial
  token，生成
  `4 × 576` 个条件位置 logits；
- Critic 估计当前状态的长期累计回报；
- PPO 使用二者共同更新共享编码器和策略网络。
- 训练期间 world-model dynamics heads 从相同 Actor latent、实体 token 及实际执行的
  卡牌/落点预测相邻状态与短期未来；当前对手圣水、手牌和下一张牌推断只读取状态；
  推理时不需要这些预测，也不会读取未来标签。

## 8. 分层动作空间

策略不是直接把“卡牌+位置”当作一个无结构的大分类，而是分解为：

```text
P(action) = P(card slot) × P(tile | card slot)
```

第一阶段选择：

- 等待；
- 手牌槽位1；
- 手牌槽位2；
- 手牌槽位3；
- 手牌槽位4。

第二阶段在 `32 × 18` 网格上选择落点。每个位置 logit 来自对应手牌 query 与该位置
CNN token 的兼容度，因此不再由一个全局向量一次性线性生成所有位置。PPO 训练时为
效率并行计算四张 heatmap，但采样和 log-prob 的概率语义严格是
`P(card) × P(tile | card)`。`action_mask` 会在
采样前屏蔽：

- 圣水不足的卡牌；
- 不合法的部署区域；
- 当前状态下不可执行的动作。

这种设计可以降低无效探索，并保留“某张牌适合放在哪里”的条件关系。

## 9. 当前主要限制

### 9.1 对手分布有限

如果长期只对随机或简单策略训练，模型容易学习针对单一对手的投机策略，训练
reward 不能代表真实策略强度。

### 9.2 卡池规模有限

当前 observation schema 只支持固定8张卡，暂时无法证明对新卡组和完整卡池的
泛化能力。

### 9.3 持续记忆仍需正式训练验证

当前已经维护 per-environment hidden state、连续变化输入和 TBPTT，但尚需正式长训练
确认它相对固定窗口版本的实际收益，并观察长期记忆是否发生遗忘或状态漂移。

### 9.4 Cross-attention 的训练收益仍需正式验证

当前已经完成 token 级交互和卡牌条件空间动作头的工程闭环，但还没有在相同参数预算、
相同训练交互量和多个随机种子下证明它优于 late fusion。注意力权重也不能自动视为
因果解释，仍需反事实输入、消融和固定评估集验证。

### 9.5 计算和显存成本增加

每次决策需要维护最多128个实体 token、32个历史 token、5个卡牌 token 和576个
全分辨率 spatial tokens。正式长训练需同步观察 MPS 显存、吞吐率以及
attention 是否退化为近似均匀分布。

### 9.6 Reward 和信用分配

当前奖励主要来自塔血变化、皇冠、最终胜负和非法动作惩罚。长期决策可能需要等待
较长时间才能获得反馈。

### 9.7 模拟器偏差

模拟器中的最优策略不一定等于真实客户端环境中的最优策略。必须通过固定评估集、
行为检查和消融实验区分真实能力提升与模拟器偏差。

### 9.8 辅助目标是代理任务

公开实体 token 不包含 native entity ID；死亡和位移训练标签已改为在环境侧按原生 ID
精确关联相邻状态，再经独立 rollout 标签通道传递。伤害来源目前仍是五类公开结果
归因，而不是底层每一次伤害事件的精确 attacker ID。正式实验必须确认这些辅助任务
能提高策略强度，而不只是降低辅助 loss。

## 10. 演进路线

### 阶段一：建立可信基线和消融实验（已有前期实验，本阶段暂不重复）

目标：确认当前完整架构中的每个模块是否真正产生收益。项目已有前期实验支持
CNN + Transformer + LSTM 组合有效，因此当前不重新执行这一阶段；以下内容保留为
未来架构发生显著变化时的回归规范。

任务：

1. 建立固定种子和固定对手的独立评估集；
2. 训练并比较 CNN-only；
3. 训练并比较 Transformer-only；
4. 训练 CNN + Transformer；
5. 训练完整 CNN + Transformer + LSTM；
6. 测试移除 public belief 的影响；
7. 测试不同 history length；
8. 固定训练决策数、种子范围和模型参数预算。

验收指标：

- 固定评估集胜率；
- 平均皇冠差；
- 平均塔血差；
- 非法动作率；
- episode reward 仅作为辅助指标；
- 不同随机种子训练结果的均值和方差；
- MPS decisions/s 和训练稳定性；
- 参数量和推理延迟。

当前固定评估基础设施已经接入一键训练：round 0、固定间隔和最终轮使用完全相同的
seed 集合；每个 seed 都交换 p0/p1；对手固定为合法随机策略和训练开始时冻结的
round-0 策略。评估使用独立环境，不进入 PPO rollout，并输出逐局 JSONL、按对手与
aggregate 汇总的 CSV，以及胜率、得分率、皇冠差和塔血差曲线。CNN-only 等完整消融
实验仍按用户决定不在当前阶段重复。

完成标准：能够用重复实验说明 CNN、Transformer、LSTM 各自带来的独立收益。

### 阶段二：历史模型池和自博弈（基础接入已完成）

目标：减少对简单固定对手的过拟合，形成持续提升的对手分布。

任务：

1. 按固定间隔保存策略快照；
2. 建立历史模型池；
3. 混合随机、规则、历史模型和当前强模型；
4. 为对手池维护 Elo 或胜率矩阵；
5. 设置旧策略采样比例，避免灾难性遗忘；
6. 建立从简单对手到强对手的 curriculum；
7. 对镜像卡组、不同玩家方向分别评估。

当前实现约定：

- 默认对手比例为随机策略20%、当前策略镜像20%、历史快照60%；
- 每个 VecEnv 槽位在一局开始时抽取对手，整局保持不变；
- 默认每5个 PPO 轮次发布一个不可变快照；
- 历史池默认保留20个候选快照，最多4个活跃历史版本参与近期采样；
- 相同历史版本的多个槽位合并为一次批量模型推理；
- 每局记录 learner generation、对手版本、玩家方向、seed、结果和 Elo；
- `self_play/pool.json` 保存池状态与对局聚合；
- `self_play/matches.jsonl` 保存逐局结果；
- `self_play/ratings.csv` 保存 Elo 变化序列。

完成标准：新模型不只提高训练 reward，而且能稳定击败多个历史版本和基准策略。

### 阶段三：升级为真正的持续时间记忆（当前基础实现已完成）

目标：让模型跨决策保留状态，而不只重读固定事件窗口。

推荐优先实现 stateful LSTM：

1. 每个 VecEnv 槽位维护独立 hidden/cell state；
2. 某场战斗结束时，只清空对应槽位的 recurrent state；
3. rollout buffer 保存序列边界和初始 hidden state；
4. 使用 sequence minibatch 和 truncated BPTT；
5. 将塔血变化、实体生成/死亡、目标切换和圣水变化加入时间序列；
6. 与当前固定窗口 LSTM 做等预算比较。

当前基础实现：

- 使用官方 `sb3-contrib==2.9.0` RecurrentPPO rollout 和 sequence mask；
- Actor 与 Critic 分别维护一组持续 LSTM state；
- 16路环境的 hidden/cell state 按槽位独立保存；
- `episode_start` 只清空已经结束的槽位；
- 对手池为当前策略及每个历史模型维护独立的 per-slot recurrent state；
- `native_model_v2` 新增20维连续变化向量；
- 连续特征覆盖塔血、圣水、单位生成/死亡、战斗单位总血量、出牌、皇冠、目标状态和实体数量变化；
- recurrent buffer 每32个决策强制建立 TBPTT 边界，并从边界处保存的 hidden state 继续训练；
- padded timestep 通过 sequence mask 排除在 PPO loss 之外。

后续候选：

- Temporal Transformer：长上下文能力强，但计算量和显存成本更高；
- S4/Mamba 类模型：长序列效率较高，但需验证 MPS 算子兼容和训练稳定性。

完成标准：在需要记忆对手卡序、圣水投入和长时间攻防转换的测试集中显著优于固定
窗口模型。

### 阶段四：跨模态融合（基础实现已完成，v3 保留为消融基线）

目标：让手牌、实体、历史、belief 和候选落点直接相互推理，而不是先各自压缩再拼接。

建议结构：

```text
实体 ───────────────┐
历史 ───────────────┼─ cross-attention ─ contextual card token ─┐
belief ─────────────┘                                           │
CNN feature map ─ spatial tokens ─ placement queries ───────────┼─ 32×18 heatmap/card
Persistent LSTM context ─────────────────────────────────────────┘
```

v3 基础实现：

1. CNN 保留 feature map，并建立低分辨率 spatial tokens；
2. 576个候选落点使用上采样局部特征和坐标编码查询 spatial tokens；
3. 历史 token 查询当前实体，再进入 Event LSTM；
4. belief token 查询实体和历史；
5. 五个手牌 token 查询实体、历史和 belief，前四个送入动作头；
6. Persistent Actor LSTM 上下文调制每张手牌 query；
7. 每张牌与所有落点 token 做缩放点积，生成 `4 × 576` 个位置 logits；
8. 等待/手牌槽位和位置两层均保留原有合法性 mask；
9. `NativeCrossModalRecurrentPolicy` 和 v3 checkpoint 格式继续保留；
10. token 维度、attention heads 和实体层数可由 `./train.sh` 高级设置修改；
11. 自动化测试覆盖合法动作、跨模态输入影响、空间 token 差异和
    attention 梯度；M5 Pro MPS 已通过 rollout/backward、保存/重载、post-load
    masked inference 和有限参数检查。

完成标准：工程上要求 CPU 与 MPS rollout/backward、保存/加载和动作 mask 全部通过；
能力上要求在参数量和训练预算接近的情况下，落点质量、样本效率和固定评估胜率优于
late fusion 基线。后一个标准需要正式训练实验，不能由冒烟测试替代。

### 阶段五：空间保持型动作头（当前基础实现已完成，v4 默认）

目标：CNN 不再把战场压成128维全局向量，也不从粗粒度 feature map 插值恢复位置；
Actor 先选择卡牌，再由这张牌直接查询原生分辨率空间特征图。

```text
16×32×18 observation
        │
full-resolution residual CNN
        │
64×32×18 feature map + coordinate encoding
        │
contextual card query（card + cross-modal context + recurrent state）
        │
scaled query/key score + shared spatial bias
        │
该卡牌的 32×18 placement heatmap
        │
legality mask → P(tile | selected card)
```

当前基础实现：

1. 新增 `NativeSpatialHeatmapExtractor`，保持完整 `32×18` CNN feature map；
2. 每个位置经过共享 CNN，因此保留平移一致的局部阵型表示；
3. 二维坐标编码补充河道、桥、塔区和己方/敌方纵深等绝对位置；
4. 新增 `SpatialHeatmapActionHead`，四张手牌分别形成 query；
5. Persistent Actor LSTM 状态加入 card query，时间上下文能够改变落点偏好；
6. 卡牌 query 与576个空间 key 进行缩放点积，直接得到四张 `32×18` heatmap；
7. 分层分布先采样等待/手牌槽位，再从所选卡牌的 masked heatmap 采样位置；
8. v4 使用独立的 `NativeSpatialHeatmapRecurrentPolicy`，不会覆盖 v3 policy 类；
9. 默认一键训练 profile 已切换至 v4，交互设置支持 CNN feature channels；
10. 23项 CPU 测试验证空间扰动会改变 heatmap、不同卡牌产生不同 heatmap、梯度能
    回到 full-resolution CNN 和 card/spatial projections；
11. M5 Pro MPS 已通过 rollout/backward、保存/重载、合法动作、有限参数和
    `(2, 4, 32, 18)` heatmap 检查。

完成标准：工程闭环已经完成；策略强度仍需与 v3 cross-modal 和旧全局动作头进行
相同交互量、参数预算与随机种子的正式对比。

### 阶段六：世界模型和辅助学习（基础实现已完成，v6 动作条件版本默认）

目标：让共享编码器显式学习动力学和未来结果，提高稀疏奖励下的表示质量。

当前实现的辅助任务：

- 下一决策双方塔血变化：Smooth L1；
- 当前实体在下一决策是否死亡：逐实体 BCE；
- 存活实体下一决策的位置变化：逐实体 Smooth L1；
- 下一决策伤害来源：无伤害、敌方打己塔、己方打敌塔、敌方打己方单位、己方打敌方
  单位五分类；
- 对手当前公开圣水估计区间：五分类；
- 对手可能手牌：8张卡的多标签 BCE；
- 对手可能下一张牌：公开 belief 的软标签交叉熵；
- 若干决策后的局面价值：未来塔血差加皇冠差的 Smooth L1，默认4步、约2秒。

训练机制：

1. `AuxiliaryWorldModelHead` 读取 recurrent Actor latent；v6 动力学任务额外读取实际
   执行的卡牌身份和己方视角落点，当前对手信息任务只读取状态；
2. 实体任务读取 Entity Transformer tokens，并由状态与动作融合后的 latent 调制；
3. 环境按原生实体 ID 生成死亡/位移标签，`NativeWorldModelPPO` 在每个 padded sequence
   minibatch 内读取独立标签通道，并构造其他当前/下一状态配对；
4. episode 边界、TBPTT 边界和 padding 均从 target mask 中排除；
5. 辅助总损失按 `auxiliary_coef` 加入同一次 PPO backward，因此梯度会更新 CNN、
   Transformer、历史模块和持续 LSTM；
6. Actor/Critic、动作概率和推理 API 保持不变，辅助预测不替代 PPO；
7. `./train.sh` 支持世界模型 hidden size、总系数和 future horizon；
8. 每个任务分别记录 `train/aux_*_loss`，并生成 `auxiliary_losses.svg`；
9. v6 使用独立的 `NativeActionConditionedWorldModelRecurrentPolicy` 和
   `NativeActionConditionedWorldModelPPO`，v3/v4/v5 policy 与算法保留用于 checkpoint
   和消融；v5 继续使用无动作条件的原始参数结构，不自动转换为 v6；
10. 25项 CPU 测试已覆盖预测张量、联合反向、梯度、日志、图表、配置约束和旧阶段回归；
11. M5 Pro MPS 已通过联合 PPO/auxiliary backward、有限辅助 loss、保存/重载、合法
    动作和 post-load auxiliary prediction。

默认 loss 组合为：

```text
total_loss = PPO_loss + 0.1 × weighted_auxiliary_loss
```

各任务权重保存在训练 profile，可独立调整。辅助标签不会添加到 observation，因此不会
造成推理阶段的未来信息泄漏。

v6 的动作编码为11维：等待标志、八张卡的身份 one-hot、己方视角归一化 x/y。
环境使用成功原生回执和本次公开出牌事件共同确认执行结果；动作身份取自回执，
坐标取自对应提交落点，生成编码时尚未自动重置。未成功部署与等待均编码为
`[1, 0, ..., 0]`。`executed_card_xy_v1` 经 `info → rollout buffer → sequence minibatch`
单独传递，与状态 t、实体标签和 PPO 样本使用同一重排/padding；PPO 的 log-prob
仍使用采样动作。动作编码不会加入 observation 或 Actor/Critic 输入。

塔血、死亡、位移、伤害来源及短期价值预测读取状态与动作融合表示；对手当前圣水、
手牌和下一张牌推断保留纯状态输入。短期价值只以当前实际动作作为条件，后续己方
动作和对手响应来自 rollout 策略分布；它不是完整动作序列模型或确定性反事实模拟器。
预测仅用于辅助训练，尚未用于搜索或直接选牌。

完成标准：工程闭环已经完成；能力验收仍要求模型在相同交互数据量下获得更高的独立
评估胜率，并且辅助预测在验证集上具有可量化精度。

### 阶段七：扩展卡池和跨卡组泛化

目标：从固定8张卡扩展到多卡组乃至完整卡池。

任务：

1. 将卡牌静态属性加入 card embedding；
2. 区分费用、类型、射程、移动速度、攻击方式和目标类型；
3. 输入双方完整 deck context；
4. 使用共享卡牌编码器处理未见组合；
5. 建立 seen deck、new combination 和 held-out card 评估集；
6. 必要时升级 observation schema 版本，避免旧 checkpoint 被错误加载。

完成标准：模型能在未参与训练的卡组组合上保持可接受的策略强度。

## 11. 推荐实施优先级

阶段一已有前期有效性实验，因此当前实施顺序为：

1. **历史模型池和自博弈（基础接入已完成）**：继续通过正式训练调整对手分布；
2. **Stateful LSTM（基础实现已完成）**：通过正式长训练验证记忆收益；
3. **Cross-Attention（v3 基础接入已完成）**：保留为消融基线；
4. **空间保持型动作头（v4 基础接入已完成）**：通过正式训练与 v3、全局头做等预算
   比较；
5. **辅助预测和世界模型（v6 动作条件接入已完成）**：与 v5/v4 做等预算消融并调整
   各任务 loss 权重；
6. **扩展卡池与跨卡组泛化**；
7. 最后再系统比较更大的 Transformer、Temporal Transformer、Mamba 等架构。

固定评估基础设施已经完成，并作为后续阶段的统一验收工具；必要消融仍保留，但不
单独重复阶段一。

## 12. 实验规则

为了避免把模拟器偏差或训练随机性误判为模型进步，后续实验应遵循：

- 一次只改变一个主要变量；
- 固定训练交互量，而不是只固定运行时间；
- 每组关键实验至少使用多个随机种子；
- 训练对手和评估对手严格分离；
- 保存完整 config、checkpoint、训练日志和评估结果；
- 同时报告均值、方差和最差结果；
- 不以单次最高 reward 作为结论；
- 新 observation schema、动作空间或 reward 版本必须显式改版本号；
- 对吞吐率、显存和推理延迟做同步记录；
- 对重要升级保留旧模型作为回归基线。

## 13. 当前结论

CNN + Transformer + LSTM 的总体方向合理，因为它分别覆盖了：

- 空间结构；
- 对象级关系；
- 时间上下文。

当前已依次完成自博弈、真正的持续记忆、跨模态融合、空间保持型动作头、辅助世界
模型和独立固定评估赛程的基础工程接入。下一步最重要的不是继续盲目扩大模型，而是
实际运行相同训练预算，用固定评估赛程验证 v6 是否优于 v5/v4/v3，并确定哪些辅助任务
真正提高胜率。

## 14. 用户确认的优化范围（2026-10-09）

当前目标是提高固定八张卡组的镜像对战强度。以下范围优先于上面的通用演进顺序；
各项实施状态分别记录如下。

1. **部署规则修复（已实现并通过原生逐格验证）**：兵种受己方场地限制，摧毁对方公主塔后才扩大对应的部分部署
   区域；当前卡组的火球和万箭齐发可以在双方场地施放。优化讨论应核对掩码是否完整
   表达这些游戏规则，而不是取消规则限制。已补齐己方后场和边缘、存活塔占位、
   拆塔后的对应前场及桥格，并将当前两种法术开放到整个棋盘。双方的初始、左塔、
   右塔和两塔被拆状态共 4,608 个格点已与原生出牌事件及实际生成坐标逐格对照，
   无不一致。规则版本为 `standard_1v1_dynamic_v1`，详细说明见 [VEC_ENV.md](../VEC_ENV.md)。
2. **辅助标签精度（已实现）**：在训练标签生成中使用原生实体 ID，准确关联相邻
   状态的实体死亡和位移，避免最近位置匹配的歧义。标签行与当前观测的 ID 排序保持
   一致，存活查找使用完整下一原生状态，避免 128 个 token 截断造成假死亡；p1 位移
   使用旋转后的己方坐标。标签通过 `info → recurrent rollout buffer → sequence minibatch`
   独立传递，原生 ID 和标签均不进入 policy observation；episode、TBPTT、padding 的
   原有排除规则保持生效。标签版本为 `native_entity_id_v1`，网络参数及
   `native_model_v2` observation schema 不变，旧 v5 checkpoint 可加载并继续训练。
   回归测试覆盖同类单位交叉、原地替换、float32 精度范围外的 ID、实体截断、自动重置、
   标签重排/padding、损失边界和 checkpoint 重载续训。全套 42 项 CPU 测试通过；16 路
   MPS 模拟状态短训练、保存/重载、合法动作和续训通过，且已验证修改前代码生成的
   v5 checkpoint 能加载并续训。当前模拟器未启动，本次尚未执行原生实战及胜率验证。
3. **动作条件预测（已实现）**：将实际执行的卡牌身份与己方视角落点编码加入辅助
   世界模型，学习状态与动作对应的后果。使用原生成功回执及公开出牌事件确认执行；
   等待、非法动作和未成功执行均使用空动作。11维 `executed_card_xy_v1` 编码通过
   独立训练通道传递，并与 recurrent rollout 的重排、episode/TBPTT 边界及 padding
   保持对齐。塔血、实体死亡/位移、伤害来源和短期价值预测读取状态与动作，当前
   对手信息推断仍只读取状态。默认 profile 切换为独立 v6 policy/algorithm，旧 v5
   保留原始参数结构及加载/续训路径，`native_model_v2` observation 与分层动作空间
   不变。短期价值以当前动作作为条件，后续动作与对手响应仍由 rollout 策略决定。
   全套48项 CPU 回归通过，覆盖动作输入差异、纯状态对手推断、动作梯度、实际执行
   编码、换牌/终局重置、重排/padding、v6 保存重载续训及 v5 回归。16路 MPS 模拟状态
   使用默认网络完成短训练、有限 loss/梯度、保存重载、合法动作和续训，共256个训练
   决策；默认网络参数量为1,824,036。正式原生实战和等预算、多种子胜率验证仍属于
   第6项。
4. **探索强度与日志（未来待办，需进一步讨论）**：讨论选牌/落点熵、等待概率、
   卡牌使用率、有效探索系数和调度方案，再决定改动。
5. **记忆机制说明**：最近 32 次公开出牌由显式历史窗口保留；持续 Actor/Critic LSTM
   跨决策保存压缩状态，按学习到的门控保留或遗忘信息，并在对应对局结束后重置。
   32 步 TBPTT 是梯度传播窗口，不是推理记忆的硬截止时间；暂不因该数值直接扩大网络。
6. **训练与评估验证（流程已实现，正式能力验收待运行）**：新增
   `training_experiments.py` 实验矩阵，按完全相同的 learner 决策预算、PPO 更新数、
   自博弈比例和快照节奏，从头训练至少三个随机种子。默认比较完整 v6、同网络但
   `auxiliary_coef=0`、无动作条件的 v5 和无世界模型的 v4；另支持八个辅助任务各自
   权重置零的单项消融，不重复阶段一编码器消融，也不改第4项探索设置。
   所有运行共用同一组独立冻结 checkpoint，记录 algorithm、文件哈希并保存副本；
   每次运行自己的 round-0 策略只用于原有开发观察，不作为跨实验的统一强度基准。
   训练期间使用开发 seed，训练结束后重载最终模型，在另一套保留测试 seed 上按
   双角色执行固定赛程。评估轨迹不进入训练预算或辅助梯度；恢复评估前的训练 RNG。
   超时标记为未完成并使赛程无效，不再计作平局；汇总拒绝缺局、重复局、预算不等、
   赛程/配置改动及模型哈希不匹配。结果按对手、玩家方向和训练种子报告胜率、
   得分率、皇冠差、塔血差、非法动作率、参数量及吞吐，计算多种子均值、样本标准差、
   最差结果和同种子的成对差值；不自动据此宣布统计显著。
   `fixed_evaluation.py` 同时支持已保存 checkpoint 的独立重评。操作与产物说明见
   [TRAINING.md](../TRAINING.md)。工程测试覆盖四组×三个种子的模拟状态 CPU 短训练、
   共享冻结对手、最终模型重载、保留测试集、同参数关闭辅助梯度、完整性校验及
   全超时场景，全套54项 CPU 测试通过；四组×三个种子的紧凑网络 MPS 模拟状态流程
   亦通过，共192个训练决策，包含保存/重载和保留赛程评估。正式原生等预算多种子
   长训练尚未运行，不能宣称 v6 镜像强度已经提高。
   本流程验证辅助任务对策略强度的贡献，独立辅助预测精度验证集仍待另行构建。
7. **多卡组与卡池泛化（未来待办）**：当前阶段暂缓，先完成固定卡组镜像对战目标。
