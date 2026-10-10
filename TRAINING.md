# 一键训练与训练产物

## 启动

原生引擎和模型训练都从项目根目录启动：

```bash
./train.sh
```

`train.sh` 会在需要时启动已有的 `Pixel_10_Pro` AVD（无窗口 SwiftShader 模式），
等待 Android 启动，再由训练入口初始化原生引擎。若模拟器已运行则直接复用。
可用 `CLASH_TRAIN_AVD` 和 `CLASH_TRAIN_SERIAL` 指定其他已有 AVD/serial，
用 `ANDROID_SDK_ROOT` 指定 SDK 目录。训练结束或中断后如需关闭模拟器，运行：

```bash
./stop_emulator.sh
```

默认 recurrent profile 需要与 Stable-Baselines3 同版本的 `sb3-contrib`：

```bash
conda run -n clash-royale python -m pip install -r requirements-training.txt
```

脚本默认使用 `clash-royale` Conda 环境。需要更换环境时：

```bash
CLASH_TRAIN_CONDA_ENV=my-env ./train.sh
```

默认 `native_hybrid_ppo` profile 使用 v6 动作条件世界模型：动力学辅助预测读取实际
成功出牌的卡牌身份和己方视角落点，当前对手信息推断仍只读取状态。动作编码仅进入
辅助训练通道，不改变 policy observation、Actor/Critic 输入或动作掩码。
`summary.json` / `summary.md` 记录架构及 `executed_card_xy_v1` 编码版本。
旧 v5 的 `NativeWorldModelPPO` / `NativeWorldModelRecurrentPolicy` 类保留，可以按
原架构加载和续训旧 checkpoint；新 v6 使用独立类，不会自动将旧 checkpoint 升级。

一次“训练轮次”表示一次完整 rollout 加一次 PPO 更新。默认每轮有 8,192 个
learner decisions，由 16 路原生 `VecEnv` 并行收集。交互流程会询问：

- 训练配置和运行名称；
- 轮次、并行战斗数、decision ticks、随机种子；
- MPS 或 CPU；
- 原生引擎端口和检查点间隔；
- 是否调整 rollout、minibatch、epochs、学习率、Gamma、GAE、熵系数和 clip。
- 是否启用自博弈、历史快照间隔和对手池大小。
- 是否调整 TBPTT 序列长度、持续 LSTM hidden size 和层数。
- 是否调整 cross-attention token 维度、头数和实体 Transformer 层数。
- 是否调整全分辨率 CNN feature channels。
- 是否调整世界模型 hidden dimension、辅助总损失系数和未来预测步数。
- 是否启用独立固定评估、评估间隔和每个对手的固定 seed 数。

如果引擎端口不可用，默认会调用 `bootstrap_emulator.py` 初始化已安装匹配
APK 和 probe 的 ARM64 模拟器。冷启动后若 probe 尚未加载，它会使用模拟器内已有的
`/data/local/tmp/frida-server-*` 注入匹配的已安装 probe。脚本只启动已有 AVD，
不会创建 AVD 或安装 APK。

## 实时状态与产物

每次运行位于 `artifacts/training_runs/<timestamp>_<profile>/`：

`artifacts/training_runs/latest.json` 始终指向最近一次运行的目录、状态文件和配置
文件。Agent 可先读取这个固定入口，再读取对应的 `status.json` 和 `metrics.jsonl`。

| 文件 | 用途 |
|---|---|
| `config.json` | 本次训练的完整、可复现配置 |
| `status.json` | 原子更新的当前状态、轮次、进度和最近指标，适合 Agent 轮询 |
| `console.log` | 完整 stdout/stderr，写入同时仍显示在终端 |
| `metrics.jsonl` | 每轮及训练结束时的结构化指标 |
| `metrics.csv` | 可直接用表格或分析程序读取的指标 |
| `episodes.monitor.csv` | Stable-Baselines3 episode monitor |
| `sb3/progress.csv` | SB3 原始 logger 数据 |
| `sb3/events.out.tfevents.*` | TensorBoard 数据 |
| `checkpoints/*.zip` | 完成 PPO 更新后定期保存，最终轮额外保存 |
| `*.zip.state/` | 与模型哈希绑定的续训配置、RNG、联赛副本和运行状态 |
| `final_model.zip` | 最终模型 |
| `interrupted_model.zip` | 异常或手动中断时尽力保存的最后模型 |
| `summary.json` / `summary.md` | 耗时、轮次、参数量、模型哈希等最终信息 |
| `charts/*.svg` | loss、reward/episode、explained variance、KL/clip、学习率、训练吞吐率、MPS 显存及探索诊断曲线 |
| `self_play/pool.json` | 对手池、活跃快照、聚合胜负、Elo 和 matchup 矩阵 |
| `self_play/matches.jsonl` | 每场自博弈的双方版本、角色、seed 和结果 |
| `self_play/ratings.csv` | learner 与对手 Elo 的时间序列 |
| `self_play/snapshots/*.zip` | 可复现的不可变历史对手模型 |
| `evaluation/schedule.json` | 固定 seed、双角色、冻结对手和评估间隔 |
| `evaluation/matches.jsonl` | 每场独立评估的版本、seed、角色、结果、皇冠差和塔血差 |
| `evaluation/summary.csv` | 每轮按对手及 aggregate 汇总的胜率、得分率和局面指标 |
| `evaluation/latest.json` | 最近一轮固定评估，适合 Agent 读取 |
| `evaluation/status.json` | 当前是否正在评估及最近完成轮次 |
| `evaluation/baselines/initial_policy.zip` | 训练开始时冻结的 round-0 基准策略，续训保留原版本 |
| `evaluation/checkpoints/*.zip` | 默认模式下等待训练结束后评估的各轮模型 |
| `charts/fixed_evaluation.svg` | 固定赛程胜率、得分率、皇冠差和塔血差曲线 |

查看 TensorBoard：

```bash
conda run -n clash-royale tensorboard --logdir artifacts/training_runs
```

## 更换模型

训练入口使用 `training_profiles/*.json` 注册模型。复制默认配置并至少修改：

```json
{
  "name": "My Policy",
  "trainer": "training_runtime:run_sb3_training",
  "environment": "native_engine.native_vec_env:NativeBatchVecEnv",
  "algorithm": "stable_baselines3:PPO",
  "policy": "my_models.my_policy:MyPolicy"
}
```

模块必须能从项目根目录导入。只更换网络时保留 trainer、environment 和 algorithm；
更换 SB3 算法时修改 algorithm 并保证参数兼容；使用非 SB3 框架时实现
`trainer(config, run_dir)` 并在 profile 中替换 trainer。这样 `./train.sh`、交互配置、
运行目录和日志入口保持不变。

默认 profile 使用 `NativeActionConditionedWorldModelRecurrentPolicy`。它把手牌、实体、历史和
belief 通过 cross-attention 连接，同时保留 `32×18` CNN feature map；每张牌直接查询
576个空间 token 并生成自己的 placement heatmap。配置中的 `cross_modal_dim`、
`cross_modal_heads`、`cross_modal_entity_layers` 和 `spatial_feature_channels` 会被自动
传给特征提取器。第四阶段的 `NativeCrossModalRecurrentPolicy` 继续保留，供旧 v3
checkpoint 和消融实验使用。

第六阶段默认同时记录 `train/auxiliary_loss` 以及八个单项 loss，并额外生成
`charts/auxiliary_losses.svg`。辅助系数为零或所有任务权重为零时，训练跳过辅助预测和
损失计算，也不记录这些 loss；单项权重为零时跳过该项损失。预测头参数仍完整保留，
可供独立推理检查。这样避免无用计算和 `0 × NaN` 污染有效训练目标。
环境从复位前的完整相邻原生状态生成塔血、实体消失/
位移及伤害结果标签，独立存入 rollout buffer；终局和 TBPTT 末项也参加监督，padding
除外。伤害结果按 ID 对齐的非投射物 HP 损失计算，新增单位血量不再抵消已有伤害。
当前公开估计监督对手圣水区间与卡序 belief；未来局面指标默认使用4个决策后的
塔血与皇冠差，约为2秒。未来标签可跨 TBPTT/minibatch 切分，但不跨新局；精确落在
终局时使用 terminal observation。环境通过 `public_scene_value_v1` 后继标签保存每步实际
到达的场景值，包括 rollout 最后一步的非终局状态；只有 horizon 超出已采集范围或提前
终局时才排除。标签验证失败后会清空待写入值，避免复用上一步标签。

## 独立固定评估赛程

默认 profile 在 round 0、每 5 轮和最终轮保存评估快照，全部训练更新完成后依次
重载执行固定评估。快照及每轮 loss 都对应该轮完成后的参数，避免少算一次 PPO 更新。默认对手是合法随机策略和训练开始时冻结的
`initial_policy`；每个对手使用固定的 16 个 seed，每个 seed 分别让 learner 作为 p0、
p1 各打一局，因此每次共 64 局。learner 和冻结策略都使用 deterministic action。

一个 resident 原生服务只有一条 batch 数据通道。默认结束训练后释放通道一次，
评估不会重置进行中的训练对局、联赛分配或循环记忆。若有第二个独立 resident，可在
profile 设置 `evaluation_port`（直接入口为 `--evaluation-port`）实时评估；两个端口
必须连接不同 resident，仅转发到同一进程不能隔离。评估轨迹不进入 rollout 或梯度。
共享服务的默认模式下，训练过程中不会出现已完成的开发评估结果；训练结束后生成曲线。

训练中的自博弈胜率随对手池变化，只适合观察联赛过程；正式比较不同 checkpoint 时应
以 `evaluation/summary.csv` 中同一固定赛程的 `aggregate` 行为准。`score_rate` 将和局
按 0.5 计分，通常比只看 `win_rate` 更稳定；皇冠差和归一化塔血差用于区分大量和局。
不同训练运行的 `initial_policy` 各自初始化，不能作为跨运行的统一强度基准。
正式比较使用下面的共享冻结 checkpoint 赛程，并同时检查各对手和 p0/p1 的结果。

评估上限耗尽而原生对局尚未结束时，逐局结果为 `incomplete`，不计为平局，胜率和
得分率只统计已完成对局；任意一局超时都会使 `valid=false`，不能进入正式实验比较。
全部超时时两个比率均为 `null`。`summary.csv` 还记录完成局数、超时数、非法动作率和
每个对手的双角色分项。非法动作率按失败尝试数除以 learner 决策数计算。评估前后恢复
Python、NumPy、Torch CPU/MPS/CUDA 随机数状态，避免冻结对手的加载消耗训练 RNG。

非交互配置冒烟检查：

```bash
./train.sh --profile native_hybrid_ppo --defaults --dry-run
```

## 等预算、多种子和辅助消融

`training_experiments.py` 将实验分为生成计划、执行、汇总三个步骤。预算按每次运行的
learner decisions 计算，必须是 `rollout_decisions` 的整数倍；评估决策不进入训练预算。
各组保持相同决策间隔、rollout、PPO 更新数、训练对手采样比例和快照发布节奏，每次
运行从头初始化自己的自博弈池。不同组的策略和历史池会随训练自然分化。
正式计划拒绝 `resume_from`，汇总要求本次新增决策数等于完整预算，避免把旧模型的累计
步数算作本次实验交互。轮次、seed、并行数等整数配置拒绝小数或布尔值，不再静默截断。

默认比较四组，每组至少三个不同训练种子：

| 组名 | 改动 | 比较目的 |
|---|---|---|
| `full` | 默认 v6 | 动作条件辅助世界模型 |
| `no_aux` | v6 的 `auxiliary_coef=0` | 保留全部参数和同种子初始化，隔离辅助梯度的收益 |
| `state_only` | v5 policy/algorithm | 辅助任务只读取状态，隔离动作条件的收益 |
| `v4` | 空间动作头，无世界模型 | 无辅助头的架构基线，参数量单独报告 |

`--all-variants` 额外加入八个 `drop_<task>` 单任务消融，每次只将对应权重置零：
`tower_delta`、`entity_death`、`entity_position`、`damage_source`、`opponent_elixir`、
`opponent_hand`、`opponent_next_card`、`future_value`。也可用 `--variants full no_aux
drop_entity_death` 指定包含 `full` 的子集。所有组均使用第 1、2 项修复后的部署规则和
原生 ID 标签；本轮不重复 CNN/Transformer/LSTM 的阶段一消融。

先准备冻结对手 manifest，格式见
[`docs/evaluation_opponents.example.json`](docs/evaluation_opponents.example.json)。每个名字
对应 checkpoint 路径与其原始 algorithm 类；相对路径以 manifest 所在目录为基准，
可提供 `sha256` 锁定版本。示例中的旧 v5 checkpoint 路径需替换为实际可用的文件；正式
能力验收应加入多个独立训练、不同强度的历史策略。使用相同八张卡组和 observation
schema，评估模型不注册进本次训练对手池。

例如，为四组 × 三种子各安排 819,200 个训练决策：

```bash
conda run -n clash-royale python training_experiments.py plan \
  --opponents docs/evaluation_opponents.example.json \
  --total-decisions 819200 --seeds 0 1 2 \
  --suite-dir artifacts/experiments/mirror_v6

./train.sh --experiment-suite artifacts/experiments/mirror_v6 --dry-run
./train.sh --experiment-suite artifacts/experiments/mirror_v6
```

`plan` 只生成计划并复制冻结对手，不启动模拟器或训练。四组共 12 次运行、9,830,400
个训练决策；预算仅为示例，不保证收敛。添加 `--all-variants` 后为 36 次运行，先查看
`plan.json` 中的总预算。`train.sh` 复用现有模拟器启动和 bootstrap 流程，按顺序执行，
避免多进程争用原生服务的单条数据通道。引擎已就绪时也可直接运行：

```bash
conda run -n clash-royale python training_experiments.py run \
  --suite-dir artifacts/experiments/mirror_v6
```

开发评估沿用 profile 中的固定 seed（默认 900001 起，共 16 个），用于观察 round 0、
定期与最终轮。正式汇总使用最终 checkpoint 重载后的另一套保留测试 seed，默认
1900001 起，共 64 个；每个对手、每个 seed 交换 p0/p1。计划检查训练可能使用的连续
seed 范围、开发集与保留测试集不重叠。测试集在训练结束后运行，不用于挑选最高
reward 或最优 checkpoint；各实验复用相同对手副本、哈希、种子、双角色与决策上限。
更大预算需要更高的评估 seed 范围时，可用 `--evaluation-seed-base` 和
`--test-seed-base` 显式设置；`--evaluation-seed-count` / `--test-seed-count` 控制局数。
确定计划后不要根据保留测试结果反复调参；探索辅助权重使用开发结果，下一轮正式
验收另选保留 seed 范围。

产物位于 `artifacts/experiments/<suite>/`：

| 文件 | 用途 |
|---|---|
| `plan.json` | 各组完整配置、预算、种子、对手哈希和 Python/profile/标准对局配置哈希 |
| `opponents/*.zip` | 按内容哈希保存的共享冻结对手副本 |
| `runs/<variant>_seed_<seed>/` | 原有训练日志、配置、最终模型和开发评估 |
| `runs/.../post_training/` | 最终 checkpoint 重载后的保留测试配置、哈希与完整赛程结果 |
| `status.json` | 当前运行组、种子、完成/失败状态 |
| `per_seed.csv` | 按训练种子、对手和角色记录强度、参数量与端到端决策吞吐 |
| `comparison.json` | 均值、样本标准差、最差/最好值及同训练种子的 `full − variant` 差值 |
| `comparison.md` | 可读对照表及完整性状态 |

`training_elapsed_seconds` 和决策吞吐排除固定评估耗时，包含采样、PPO、日志与保存；
`elapsed_seconds` 保留包含评估的总耗时。续训吞吐使用本次新增决策数。两者均不能
解释为纯 PPO 或原生 stepping 速度。
正式汇总从逐局记录重新计算，拒绝预算不足、缺局、重复局、超时、配置/赛程改动及
checkpoint 哈希不匹配。中途已有完整结果的运行可跳过；中断训练保留已有产物，
重新从头训练需新建 suite，避免混入不等预算。Python/profile 改动后也应新建计划。
汇总还检查逐局 winner／胜负一致性、checkpoint 轮次、决策与失败计数、关键指标有限值
及有效训练耗时。保留测试因异常中断时，重跑 suite 会将该评估目录归档为
`evaluation_failed_NNN` 后重新执行相同赛程，不重训已完成的模型。已完成的评估不会自动
重跑，即使其超时而无效；损坏或无效的完整结果仍报错。
尚未完成整个矩阵时，汇总保留完整种子的组，但总状态为无效。

```bash
conda run -n clash-royale python training_experiments.py report \
  --suite-dir artifacts/experiments/mirror_v6
```

比较以训练种子为独立单位，不能把同一个 checkpoint 的大量对局当成独立训练重复。
优先查看各冻结策略及 p0/p1 的得分率、胜率、皇冠差、塔血差和非法动作率，再看
aggregate；报告多种子均值、标准差与最差结果，保留同种子的成对差值。当前汇总是
描述统计，不自动宣布统计显著。辅助 loss 降低不能代替镜像对战强度提升；当前流程
验证辅助任务的策略收益。独立辅助预测验证工具见下节；正式原生验证集与预测精度
结果仍需实际采集、评分。

## 单独重评已保存模型

可以使用保存的 `config.json` 和对应 algorithm 重评任意 checkpoint，无需训练。
输出目录必须全新，已有评估不会被覆盖；默认复用配置中的固定对手、种子和推理方式。

```bash
conda run -n clash-royale python fixed_evaluation.py \
  --config artifacts/experiments/mirror_v6/runs/full_seed_0/post_training/config.json \
  --checkpoint artifacts/experiments/mirror_v6/runs/full_seed_0/final_model.zip \
  --output-dir artifacts/reevaluation/mirror_v6_full_seed_0
```

原生引擎需已就绪。旧单次训练配置使用 `initial_policy` 时，从该训练目录的
`evaluation/baselines/initial_policy.zip` 加载；独立评估不建立 PPO rollout，不更新
参数。退出码 0 表示赛程完整，2 表示存在未完成对局。


## 直接入口与续训

直接入口默认使用同一个 v6 profile，`--output` 现在表示新运行目录：

```bash
conda run -n clash-royale python train_native.py --rounds 10 --output artifacts/native_training_v6
./train.sh --resume artifacts/native_training_v6/final_model.zip --rounds 20 --defaults
```

`--rounds 20` 是累计目标20轮；如果已完成10轮，只再训练10轮。也可用直接入口的
`--resume <checkpoint> --rounds 20 --output <新目录>`。必须一起保留 `.zip` 与同名
`.zip.state/`，检查点和最终模型都有对应状态；普通旧 `.zip` 不含这些运行状态，不能
冒充完整联赛续训。状态包含模型哈希、原配置、全局 RNG、下一 native seed、历史模型
副本、评级、比赛聚合和联赛 RNG；模型文件保留优化器及累计更新数。续训启动新对局，
不恢复 resident 内部战斗、正在进行的对手分配和对局循环状态，因此不保证逐步轨迹相同。
核心训练配置必须一致，可更改累计轮数、设备、端口和新运行目录。
固定评估启用时，续训必须保留原始 `initial_policy.zip`，缺失时拒绝续训，避免把恢复后的
策略当成初始基线。新保存的状态记录初始基线和历史对手快照的 SHA-256，恢复前核验；
历史快照注册后在加载和保存时也核验身份。旧状态若没有这些校验值，只能检查文件存在，
无法追溯验证其内容是否被替换。
历史快照 ID 不能使用 `random`、`current` 或路径形式；重复注册必须保持路径、轮次和
内容哈希一致，仍被对局使用的已淘汰 ID 也不能复用。恢复前拒绝重复或保留的快照 ID。
新状态采用 `native_training_resume_v2`，同时保存配置、全局 RNG 文件和联赛清单的
SHA-256；所有必要文件写入成功后才发布 `runtime.json`，恢复前先核验这些文件。
旧无 schema 状态保留兼容读取，但无法追溯核验这些额外文件的身份。

历史对手、固定评估和离线检查使用 `load_inference_model`，加载后释放训练 rollout
缓冲区和优化器状态；该对象仅用于推理。续训使用标准 `algorithm.load`，保留完整优化器。
每次运行记录 `source_sha256.json`，并在 `summary.json` 中保存启动时的源码哈希；
训练期间源码发生变化会使运行失效。正式实验汇总同时核验其与计划一致，哈希范围包括
Python、训练配置、依赖约束和原生 probe 源码。计划须使用修复后的源码重新生成。
历史实验数据和冻结对手文件已清理，当前工作区没有可执行的正式计划；准备可用的冻结
对手 checkpoint 后，按上面的 `training_experiments.py plan` 命令重新生成。
运行时也拒绝含训练产物的输出目录，且独占创建源码记录，防止重试或并发调用覆盖旧运行。
已准备的 `config.json`、`console.log`、`status.json` 可以保留。

离线固定评估使用显式 `evaluation_initial_policy_path`，或从 checkpoint 状态侧车、原运行
目录查找初始基线，并利用侧车中的哈希核验。找不到原始基线时明确报错，不会用待评估
模型替代初始策略。只评估 random 或其他冻结对手时不需要初始基线。
离线评估在加载前固定 checkpoint 身份，加载后与完成前再次核验，并在赛程运行前
写入 `checkpoint.json`。结果、摘要和图表全部写入后才发布 `status.json` 的完成标记；
任何收尾异常都记录为失败，保留测试可归档后重跑。正式汇总要求有效的完成标记，
不会仅凭已有逐局文件接纳失败结果；全部结果失效时清除上一次生成的 `per_seed.csv`。

中断后的 `interrupted_model.zip` 是尽力保存的权重，可能处于采样或优化中间；恢复时
优先使用最近一个完整的 `checkpoints/*.zip` 及其状态目录。最终轮在固定评估之前额外
保存，评估失败也保留完整的训练检查点。
`interrupted_model.zip` 仅在已确认完成 PPO 更新的边界才保存续训状态，步数恰好整除
rollout 不能单独证明更新完成。回调提前停止也不能把未完成更新的模型标记为完整训练。
该判断同时用于自博弈快照、最终固定评估和训练指标；中断摘要中的 `completed_rounds`
只计已经完成的 PPO 更新，`timesteps` 仍保留实际采样数。提前停止不执行待运行的延迟评估。
v4、v5、v6 共用 PPO 更新循环，只有有效样本数大于一时才归一化 advantage；直接 Python
调用产生的单样本 minibatch 不会因无偏标准差而污染参数。普通 v4 批次与上游更新保持一致。

learner 与对手动作在转换为 `int64` 前检查形状、整数性、有限值和表示范围；小数、布尔值、
字符串或溢出输入会明确报错，不会截断成另一条动作。整数形式的非法动作继续走环境原有的
非法动作处理。联赛、固定对手和直接传入的 SB3 对手在动作检查通过后才提交下一循环状态。
公开圣水估计按事件维护扣费后的值，历史时刻通过二分查找查询；出牌次数和最近出牌时刻
复用事件索引，构造观测不再重放完整账本。完整账本仍保留，重置时清空增量索引。

分层分布的 `log_prob` 对掩码外、小数、非有限或越界动作返回负无穷；等待仅接受
`(0, 0, 0)` 编码。随机动作只抽取一次 slot 和一次所选卡牌的条件落点；等待落点固定为零。
循环 minibatch 的全零填充掩码使用确定性等待分布，填充行仍由训练有效性掩码排除。
此改动保持网络维度和确定性推理规则，但改变随机数消耗，同一 seed 的随机轨迹可能变化。

## 探索诊断与策略回放

每轮 `metrics.jsonl/csv` 和 `charts/exploration.svg` 记录选牌熵、按选牌概率加权的
条件落点熵、等待概率/采样率及八张卡牌采样率。统计来自 learner 的采样 forward，
不会被当前策略镜像对手的推理覆盖；它们反映策略分布与采样，不等同于成功出牌率。
探索系数和调度保持 profile 设置，尚无实验依据自动修改。

```bash
conda run -n clash-royale python native_visualizer.py \
  --checkpoint artifacts/native_training_v6/final_model.zip \
  --steps 64 --output artifacts/replays/v6.html
```

需要已就绪的原生服务；工具记录 deterministic learner 对合法随机对手的轨迹，输出
独立 HTML 与 JSON。回放可逐帧选牌查看实际掩码后的条件落点概率、非法格、双方占据、
等待概率和所选动作；从 checkpoint 侧车或运行目录自动读取配置。
每帧记录真实的 `tick`、`next_tick` 和 `terminal`，终局短步长不会按完整决策间隔倒推。

## 独立辅助预测验证

先使用冻结 checkpoint 在独立 seeds 上采集公开轨迹，再对同一份数据评分：

```bash
conda run -n clash-royale python auxiliary_validation.py collect \
  --checkpoint artifacts/native_training_v6/final_model.zip \
  --dataset artifacts/validation/public_v3.npz --seed-base 2900001
conda run -n clash-royale python auxiliary_validation.py evaluate \
  --checkpoint artifacts/native_training_v6/final_model.zip \
  --dataset artifacts/validation/public_v3.npz --output artifacts/validation/v6_metrics.json
```

采集需原生服务；评分仅需 NPZ 和模型。验证 seeds 必须避开完整训练范围、开发 seeds
和实验计划的保留测试 seeds。数据记录观测/标签 schema、决策间隔、horizon、seed/角色、
采集策略权重哈希与 checkpoint；评分记录数据文件哈希，不更新参数。比较模型时固定
同一数据文件，各模型从观测序列重建自己的循环状态，不复用采集模型的 hidden state。
当前格式为 `native_auxiliary_validation_v2`，保存 slot、终局标记和后继场景值。评分前
核验数组形状、有限值、观测范围、动作编码、时间／slot 顺序、重置边界，以及未来标签
是否对应同局的指定 horizon。旧 v1 数据缺少这些校验字段，需要重新采集。
采集端核验环境返回的标签 schema 和数值范围后才写入数据文件，独占创建文件以防覆盖。
评分核验辅助头输出形状及有限值，拒绝 NaN／Inf 指标；下一张牌目标与训练采用相同的
概率归一化方式。
支持 v5/v6，v4 没有辅助头，不参与辅助预测评分。输出塔血/位移/短期指标 MAE、实体
消失 BCE/准确率、伤害结果准确率，以及公开圣水/卡序重构指标和有效标签数。

这些是公开动力学与规则信念重构验证，不包含真实隐藏手牌或真实 attacker ID；不能
据此宣称恢复隐藏信息。正式原生数据、等预算多种子长训及强度结果仍待运行。
