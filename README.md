# Safe USV Pursuit with 3DOF Body-Frame SAC

[English](#english) | [中文](#中文说明)

> **Bilingual maintenance policy / 双语同步维护约定**  
> Every technical change to this README must update the matching English and Chinese sections in the same change. Do not leave either language version stale.  
> 本 README 的任何技术性修改都必须在同一次修改中同步更新对应的英文与中文段落，不允许只更新其中一种语言。

## English

### Overview

This repository is a research codebase for safe single-USV pursuit in a bounded maritime environment. The USV must reach a target neighbourhood while avoiding static islands/reefs, moving vessels, map boundaries, and unsafe encounters. The policy is trained with Soft Actor-Critic (SAC) and executed through an underactuated 3DOF vessel model driven by continuous thrust and rudder commands.

The current reference experiment is [`SAC_3DOF_ENTITY_ENCODER_0826.txt`](SAC_3DOF_ENTITY_ENCODER_0826.txt). Its corresponding training run is `SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_0826`.

### Main Capabilities

- **3DOF USV dynamics:** surge, sway, and yaw dynamics; actuator-rate limits; rudder-angle and yaw-rate limits; action smoothing; current disturbance; 1 s environment steps with 0.1 s internal integration.
- **Body-frame observation:** target, obstacle positions, and relative dynamic-vessel velocity are expressed in the ownship body frame. The base state contains `u`, `v`, `r`, heading encoding, and actual rudder angle.
- **SAC safety learning:** replay transitions retain raw, constrained, and applied actions. When the COLREGs filter is active, the actor can use rule-imitation supervision during training.
- **Maritime safety layer:** COLREGs reports and an optional action filter protect the agent during assisted execution. Policy-only evaluation keeps the filter disabled to measure whether rule behaviour was internalized.
- **Moving entities:** dynamic vessels follow bounded, collision-aware and boundary-aware behaviour. The target can be static or configured with a constrained movement mode.
- **Evaluation tooling:** one-shot visual evaluation, custom multi-episode evaluation, COLREGs event auditing, CSV/JSON summaries, and a five-variant Proportional Navigation (PN) baseline.
- **Test-only visual rendering:** static circular collision geometry can be rendered as deep-toned oasis islands, sand islands, and reefs. Rendering is clipped to `0.98R` and never changes physics or collision checks.

### Requirements

The maintained local environment is `RLforUSV_L_T`. It must provide Python, PyTorch, NumPy, Matplotlib, Pillow, TensorBoard/TensorBoardX, and the project dependencies.

```powershell
conda activate RLforUSV_L_T
python -c "import torch, numpy, matplotlib, PIL, tensorboardX; print(torch.__version__)"
```

The default artifact root is `D:\USV\logs`. Set `USV_LOG_ROOT` before launching a command to use another drive or directory.

### Project Layout

```text
main.py                                  # Training entry point and checkpoint/resume flow
SAC_3DOF_ENTITY_ENCODER_0826.txt         # Current reference training configuration
SAC_3DOF_ENTITY_ENCODER_RULE_BUFFER_0829.txt # Optional dual-batch rule-imitation configuration
see_trained_once.py                      # One-shot SAC evaluation with PNG/GIF/CSV output
see_trained_custom.py                    # Custom scenes, COLREGs audit, and PN variants
see_trained_policy_only.py               # Policy-only evaluation helper
evaluation/test_terminal_report.py       # Shared bilingual terminal metrics report
CUSTOM_EVALUATION_PN_BASELINE_0823.txt   # Current PN five-variant test configuration
algorithms/sac/                          # MASAC networks, actor/critic, replay integration
dynamics/usv_3dof.py                     # Underactuated 3DOF vessel dynamics
safety/                                  # COLREGs reports and action filter
guidance/proportional_navigation.py      # PN candidate-action generator
multiagent/scenarios/usv_tracking_base.py# Scenario generation and rewards
utilities/                               # Paths, observation schema, and environment helpers
utilities/rule_replay_buffer.py           # Independent COLREGs teacher replay buffer
utilities/rule_metadata.py                # Stable encounter/filter metadata encodings
scripts/                                 # Focused validation scripts
```

### Training

Create a new experiment by copying the reference configuration. Change `TRAINING_RUN_NAME` before the first launch so existing logs and checkpoints are not overwritten.

```powershell
Copy-Item SAC_3DOF_ENTITY_ENCODER_0826.txt SAC_3DOF_ENTITY_ENCODER_MY_RUN.txt
# Edit SAC_3DOF_ENTITY_ENCODER_MY_RUN.txt:
#   TRAINING_RUN_NAME = SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_MY_RUN
#   PRE_TRAINED = false

& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python main.py SAC_3DOF_ENTITY_ENCODER_MY_RUN
```

### Optional Dual-Batch Rule Replay

The independent rule-replay path is opt-in. `MainReplayBuffer` still stores
every environment transition and remains the only source for the Critic,
SAC actor objective, entropy-temperature update, and PER priorities.
`RuleReplayBuffer` stores only `filter_active=True` transitions and supplies a
second batch to the Actor imitation term:

```text
L_actor = L_SAC(B_main) + RULE_IMITATION_WEIGHT * L_rule(B_rule)
```

The two losses are accumulated and back-propagated once. Rule samples never
enter the Critic target, Alpha update, or main-buffer priority calculation.
Until `RULE_BUFFER_MIN_SIZE` is reached, training follows the normal SAC
update and the rule term is zero. `RULE_SAMPLE_MODE = uniform` samples all
active rule records uniformly; `stratified` balances the available encounter
type codes where possible.

The configurable fields are `RULE_BUFFER_ENABLED`, `RULE_BUFFER_SIZE`,
`RULE_BUFFER_MIN_SIZE` (per-agent warm-up count), `RULE_ACTOR_BATCH_SIZE`,
`RULE_SAMPLE_MODE`, and `RULE_IMITATION_WEIGHT`. The historical misspelling
`RULE_IMMITATION_WEIGHT` is accepted as a fallback, but new files should use
the correctly spelled name. TensorBoard exposes pool fill/sampling statistics
under `rule_buffer/*` and actor-only batch/loss statistics under
`agent<i>/rule_update/*`.

The reference `SAC_3DOF_ENTITY_ENCODER_0826.txt` explicitly sets
`RULE_BUFFER_ENABLED = False`, so its existing continuation remains
main-buffer-only. A fresh enabled experiment is provided as
`SAC_3DOF_ENTITY_ENCODER_RULE_BUFFER_0829.txt`:

```powershell
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python main.py SAC_3DOF_ENTITY_ENCODER_RULE_BUFFER_0829
```

The rule pool is saved separately as `rule_buffer_last.file`,
`rule_buffer_best.file`, and `before_dir/rule_buffer_<episode>.file`. When an
enabled run resumes, that file is restored if present; enabling the module
from an older disabled checkpoint starts the rule pool empty while retaining
the SAC network/replay checkpoint. Rule metadata is appended to the fixed
`info` diagnostic array without changing its legacy field offsets.

For a true resume, keep the same `TRAINING_RUN_NAME`, set `PRE_TRAINED = true`, and keep the network contract unchanged. In particular, do not alter `OBSERVATION_LAYOUT`, `control_model`, `num_agents`, `num_landmarks`, `num_ob`, `num_static_ob_slots`, `ENTITY_ENCODER_ENABLED`, `ENTITY_ENCODER_LATENT_DIM`, or the actor/critic architecture. The resume state restores network weights, optimizer states, entropy state, replay buffer, metric windows, and RNG state where available.

```powershell
# After setting PRE_TRAINED = true in the same training configuration:
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python main.py SAC_3DOF_ENTITY_ENCODER_MY_RUN
```

Open TensorBoard for a run:

```powershell
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T tensorboard --logdir D:\USV\logs\SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_MY_RUN\log --host 127.0.0.1 --port 6006
```

### Evaluation

Run one deterministic visual evaluation of the current reference checkpoint:

```powershell
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_once.py SAC_3DOF_ENTITY_ENCODER_0826 --checkpoint last --episodes 1 --max-steps 200
```

Run the current five-variant PN comparison. Each variant receives the same episode seed and therefore the same initial scene, entity sizes, headings, and current realization.

```powershell
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --episodes 100 --max-steps 200
```

Useful temporary overrides:

```powershell
# Debug only: one seed, short rollout, no GIF.
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --episodes 1 --max-steps 10 --no-gif

# Evaluate only the unassisted SAC policy, which is the correct setting for rule internalization.
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --variants sac --episodes 100

# Hide only the drawn collision circle and use another large:medium:small visual allocation.
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --hide-collision-boundary --static-obstacle-ratio 2:3:5
```

The custom evaluator supports these variants:

| Variant | Candidate action | Agent COLREGs filter | Purpose |
| --- | --- | --- | --- |
| `sac` | SAC | Off | Learned policy without assistance |
| `pn_only` | PN | Off | Classical guidance baseline |
| `sac_pn` | SAC thrust plus SAC/PN rudder blend | Off | Learned-guidance hybrid |
| `sac_colregs` | SAC | On | SAC plus safety-layer upper bound |
| `sac_pn_colregs` | SAC/PN blend | On | Hybrid plus safety-layer upper bound |

`CUSTOM_EVALUATION_PN_BASELINE_0823.txt` is named after the original PN protocol date, but it now explicitly binds to the current `0826` training configuration and checkpoint directory. It keeps the 600 m, 2 moving-vessel, 3 static-obstacle stage-1 scene and the 200-step horizon used by the reference training contract.

### Three Evaluation Scripts and Fast Rule Check

The three evaluators now print the same bilingual terminal report after the rollout. They all use the current 3DOF environment, observation contract, checkpoint metadata validation, and deterministic policy inference (`noise=0`). Their differences are:

| Script | Agent COLREGs filter | Scene | Main purpose |
| --- | --- | --- | --- |
| `see_trained_once.py` | Keeps the configured training-time filter | Same environment configuration as the training configuration | Visualize the complete assisted system and its upper-bound safety behaviour |
| `see_trained_policy_only.py` | Always disabled for the trained agent | Same environment configuration as the training configuration | Measure whether the SAC policy itself has internalized COLREGs; this is the primary rule-learning check |
| `see_trained_custom.py` | Controlled by the selected variant | Read from a separate custom test TXT | Evaluate generalization and compare SAC, PN, hybrid, and assisted variants |

The dynamic obstacle vessels retain their configured scripted collision-avoidance behaviour in policy-only evaluation. Only the trained agent's action-set filter is disabled. The audit is shadow-only: it observes the encounter reports and the original policy action, but never changes the action, reward, or dynamics.

Examples:

```powershell
# Complete training-flow visualization, including the configured safety filter.
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_once.py SAC_3DOF_ENTITY_ENCODER_0826 --checkpoint last --episodes 1 --max-steps 200

# Pure trained policy; use this mode to judge rule internalization.
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_policy_only.py SAC_3DOF_ENTITY_ENCODER_0826 --checkpoint last --episodes 10 --max-steps 200 --no-save-gif

# Custom multi-variant evaluation; `sac` is the unassisted policy variant.
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --variants sac,sac_colregs --episodes 10 --max-steps 200
```

### Terminal Metrics and Internalization Definition

`evaluation/test_terminal_report.py` prints the following metrics for every one-shot/policy-only run and for every custom-evaluation variant:

| Metric | Meaning |
| --- | --- |
| `success_rate` | Fraction of episodes that reach the target neighbourhood and terminate successfully |
| `collision_rate` | Fraction of episodes terminated by collision with a vessel, obstacle, or target |
| `out_of_bounds_rate` | Fraction of episodes terminated after leaving the map boundary |
| `timeout_rate` | Fraction of episodes that reach the configured maximum step count |
| `mean_reward`, `reward_std` | Mean and standard deviation of episode return |
| `mean_steps` | Mean number of environment steps before termination |
| `mean_minimum_clearance_m` | Mean episode minimum net clearance after subtracting both entity radii; lower values mean less safety margin |
| `risk_event_count` | Number of continuous COLREGs risk episodes, counted per dynamic obstacle vessel rather than per time step |
| `policy_rule_follow_rate` | Fraction of risk events in which the original policy action satisfies the audit's rule-direction and reaction criteria |
| `safe_resolution_rate` | Fraction of risk events that end without collision/out-of-bounds/timeout and leave the configured clearance |
| `internalization_success_rate` | Fraction satisfying both policy rule-following and safe resolution: `compliant_and_safe_events / risk_event_count` |
| `filter_active_rate` | Fraction of environment-step actions changed by the agent COLREGs filter; a high value indicates dependence on assistance |

The report also prints rule-following, safe-resolution, and internalization rates for each detected encounter type. `N/A` is intentional when a denominator is zero. In particular, `risk_event_count = 0` means that this rollout provides no evidence about rule internalization; it must not be reported as 0% learned. A collision-free result alone proves safety for that test, not COLREGs internalization. For a convincing claim, compare `see_trained_policy_only.py` with `see_trained_once.py` or compare custom `sac` with `sac_colregs`, using multiple seeds and enough risk events.

### Evaluation Outputs

Unless `OUTPUT_ROOT` is set, custom evaluation writes to:

```text
D:\USV\logs\<TRAINING_RUN_NAME>\evaluation\<TEST_NAME>\<timestamp>\
```

Key outputs include:

- `effective_environment_config.txt`: final merged scene and training configuration.
- `run_manifest.json`: checkpoint, network contract, seeds, PN settings, visual settings, and summary metadata.
- `episode_summary.csv` and `variant_summary.csv`: task success, collisions, path efficiency, clearance, intervention, and PN diagnostics.
- `colregs_step_log.csv`, `colregs_event_log.csv`, and summary CSV files: event-level rule internalization and safe-resolution statistics.
- Per-episode PNG/GIF, trajectory CSV, and action/state CSV files.

COLREGs auditing counts an encounter as one continuous event rather than one event per step. The target vessel is excluded from this audit because it is the task endpoint; only dynamic obstacle vessels are audited. The audit observes actions and states but never changes dynamics, rewards, or the executed action.

### Configuration Rules

- Training configuration controls the **network contract** and the learned policy.
- Custom evaluation configuration controls the **test scene**, output, visualization, PN parameters, and audit thresholds.
- `PRESERVE_TRAINING_DYNAMICS = true` is recommended for fair comparison. It keeps the checkpoint's 3DOF time scale, actuator limits, and action smoothing, while explicit scene-scale safety distances still come from the test configuration.
- `NUM_DYNAMIC_OBSTACLES` and `NUM_STATIC_OBSTACLES` may change for generalization tests. The policy still observes only the training-time slot counts (`num_ob` and `num_static_ob_slots`).
- `STATIC_OBSTACLE_VISUAL_RATIO = 2:3:5` is visual only. With ten static obstacles it yields exactly 2 oasis islands, 3 sand islands, and 5 reefs. Sparse scenes preserve the intuitive size order: 1 -> large island, 2 -> large plus medium, 3 -> one of each.

### Validation

Before a long experiment, validate body-frame observation logic and run a short evaluator smoke test:

```powershell
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python scripts\check_body_frame_observation.py SAC_3DOF_ENTITY_ENCODER_0826
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --episodes 1 --max-steps 10 --no-gif
```

### License and Upstream Work

This repository retains the original upstream license in [`LICENSE`](LICENSE). It extends the upstream RLforUTracking project with the current USV 3DOF, safety, body-frame observation, and evaluation workflow. Preserve applicable upstream attribution when redistributing the code.

---

## 中文说明

### 项目概述

本仓库用于研究有界海洋环境下的单无人艇安全追踪。无人艇需要在避开静态岛屿/礁石、动态船舶、地图边界和危险会遇的前提下，抵达目标周边。策略采用 Soft Actor-Critic（SAC）训练，并通过以连续推力和舵角驱动的欠驱动 3DOF 船舶模型执行。

当前参考实验为 [`SAC_3DOF_ENTITY_ENCODER_0826.txt`](SAC_3DOF_ENTITY_ENCODER_0826.txt)，对应训练实验名为 `SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_0826`。

### 主要能力

- **3DOF 无人艇动力学：** 包含纵荡、横荡和艏摇动力学，执行器变化率限制、最大舵角和艏摇角速度限制、动作平滑、洋流扰动；环境步长为 1 s，内部积分步长为 0.1 s。
- **船体坐标系观测：** 目标、障碍物相对位置及动态船相对速度均在本船坐标系表达；基础状态包含 `u`、`v`、`r`、航向编码和实际舵角。
- **SAC 安全学习：** 经验池保存原始动作、约束动作和实际执行动作；COLREGs 过滤器介入时，actor 可在训练阶段获得规则模仿监督。
- **海事安全层：** COLREGs 报告和可选动作过滤器可在辅助执行阶段保护智能体；`policy_only` 评估关闭过滤器，用于判断策略是否真正内化规则行为。
- **动态实体：** 动态障碍船具有限速、限转向、避障和边界规避逻辑；目标船可设为静止或受约束的运动模式。
- **评估工具：** 支持单次可视化测试、自定义多回合测试、COLREGs 事件审计、CSV/JSON 汇总，以及五变体比例导引（PN）基线。
- **仅测试阶段的视觉渲染：** 圆形碰撞障碍物可显示为深色绿洲岛、沙岛和礁石；图案限制在 `0.98R` 内，不会改变物理动力学或碰撞判定。

### 环境要求

当前维护的本地环境为 `RLforUSV_L_T`，其中应包含 Python、PyTorch、NumPy、Matplotlib、Pillow、TensorBoard/TensorBoardX 及项目依赖。

```powershell
conda activate RLforUSV_L_T
python -c "import torch, numpy, matplotlib, PIL, tensorboardX; print(torch.__version__)"
```

默认日志与断点根目录为 `D:\USV\logs`。如需使用其他磁盘或目录，请在运行命令前设置 `USV_LOG_ROOT`。

### 项目结构

```text
main.py                                  # 训练入口、断点保存与续训流程
SAC_3DOF_ENTITY_ENCODER_0826.txt         # 当前参考训练配置
SAC_3DOF_ENTITY_ENCODER_RULE_BUFFER_0829.txt # 可选双批次规则模仿配置
see_trained_once.py                      # 单次 SAC 测试，输出 PNG/GIF/CSV
see_trained_custom.py                    # 自定义场景、COLREGs 审计和 PN 变体
see_trained_policy_only.py               # 仅策略测试工具
evaluation/test_terminal_report.py       # 统一的中英文终端指标报告
CUSTOM_EVALUATION_PN_BASELINE_0823.txt   # 当前 PN 五变体测试配置
algorithms/sac/                          # MASAC 网络、actor/critic 与经验池集成
dynamics/usv_3dof.py                     # 欠驱动 3DOF 船舶动力学
safety/                                  # COLREGs 规则报告与动作过滤器
guidance/proportional_navigation.py      # PN 候选动作生成器
multiagent/scenarios/usv_tracking_base.py# 场景生成与奖励函数
utilities/                               # 路径、观测模式与环境辅助工具
utilities/rule_replay_buffer.py           # 独立 COLREGs 规则教师经验池
utilities/rule_metadata.py                # 稳定的会遇/过滤模式编码
scripts/                                 # 面向单项功能的校验脚本
```

### 训练流程

新实验应从参考配置复制开始。首次运行前必须修改 `TRAINING_RUN_NAME`，避免覆盖已有日志和断点。

```powershell
Copy-Item SAC_3DOF_ENTITY_ENCODER_0826.txt SAC_3DOF_ENTITY_ENCODER_MY_RUN.txt
# 编辑 SAC_3DOF_ENTITY_ENCODER_MY_RUN.txt：
#   TRAINING_RUN_NAME = SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_MY_RUN
#   PRE_TRAINED = false

& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python main.py SAC_3DOF_ENTITY_ENCODER_MY_RUN
```

### 可选双批次规则经验池

独立规则经验池默认关闭，只有在配置中显式打开时才启用。`MainReplayBuffer`
仍然保存全部环境转移，并且仍是 Critic、SAC 主 Actor 损失、熵温度更新和
PER 优先级的唯一数据来源。`RuleReplayBuffer` 只保存
`filter_active=True` 的转移，为 Actor 提供第二个规则批次：

```text
L_actor = L_SAC(B_main) + RULE_IMITATION_WEIGHT * L_rule(B_rule)
```

两项损失相加后只进行一次反向传播。规则批次不会参与 Critic 目标、Alpha
更新或主经验池优先级计算。规则池达到 `RULE_BUFFER_MIN_SIZE` 之前，训练仍
按普通 SAC 更新，规则损失为零。`RULE_SAMPLE_MODE = uniform` 对规则样本
均匀采样；`stratified` 会在可用范围内平衡不同会遇类型。

可配置字段包括 `RULE_BUFFER_ENABLED`、`RULE_BUFFER_SIZE`、
`RULE_BUFFER_MIN_SIZE`（按 agent 计的预热样本数）、`RULE_ACTOR_BATCH_SIZE`、
`RULE_SAMPLE_MODE` 和 `RULE_IMITATION_WEIGHT`。历史配置中的拼写错误
`RULE_IMMITATION_WEIGHT` 仍作为回退项兼容，但新配置应使用正确拼写。
TensorBoard 会在 `rule_buffer/*` 下显示规则池容量/采样统计，在
`agent<i>/rule_update/*` 下显示仅 Actor 的规则批次和损失统计。

参考配置 [`SAC_3DOF_ENTITY_ENCODER_0826.txt`](SAC_3DOF_ENTITY_ENCODER_0826.txt)
已明确设置 `RULE_BUFFER_ENABLED = False`，因此原有续训仍保持主经验池语义。
新建的 [`SAC_3DOF_ENTITY_ENCODER_RULE_BUFFER_0829.txt`](SAC_3DOF_ENTITY_ENCODER_RULE_BUFFER_0829.txt)
用于开启该模块的全新实验：

```powershell
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python main.py SAC_3DOF_ENTITY_ENCODER_RULE_BUFFER_0829
```

规则池独立保存为 `rule_buffer_last.file`、`rule_buffer_best.file` 和
`before_dir/rule_buffer_<episode>.file`。启用模块的实验续训时，如果文件存在
就恢复；从旧的关闭状态断点首次打开模块时，规则池从空开始，但 SAC 网络和
主经验池仍可继续使用。规则元数据追加在固定 `info` 数组末尾，不改变旧字段
的位置。

真正的续训必须保持相同的 `TRAINING_RUN_NAME`，将 `PRE_TRAINED = true`，并保持网络契约不变。尤其不得更改 `OBSERVATION_LAYOUT`、`control_model`、`num_agents`、`num_landmarks`、`num_ob`、`num_static_ob_slots`、`ENTITY_ENCODER_ENABLED`、`ENTITY_ENCODER_LATENT_DIM` 或 actor/critic 架构。断点会在可用时恢复网络参数、优化器状态、熵参数、经验池、指标窗口和随机数状态。

```powershell
# 在相同训练配置中将 PRE_TRAINED 设置为 true 后：
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python main.py SAC_3DOF_ENTITY_ENCODER_MY_RUN
```

打开某个实验的 TensorBoard：

```powershell
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T tensorboard --logdir D:\USV\logs\SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_MY_RUN\log --host 127.0.0.1 --port 6006
```

### 测试流程

对当前参考断点执行一次确定性的可视化测试：

```powershell
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_once.py SAC_3DOF_ENTITY_ENCODER_0826 --checkpoint last --episodes 1 --max-steps 200
```

执行当前 PN 五变体公平对比。每个变体使用相同 episode 种子，因此初始场景、实体尺寸、航向和洋流实现完全一致。

```powershell
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --episodes 100 --max-steps 200
```

常用临时覆盖参数：

```powershell
# 仅调试：一个种子、短回合、不生成 GIF。
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --episodes 1 --max-steps 10 --no-gif

# 只评估未辅助的 SAC 策略；这是检验规则内化时应使用的设置。
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --variants sac --episodes 100

# 仅隐藏绘制出的碰撞圆，并使用另一组大:中:小静态障碍视觉比例。
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --hide-collision-boundary --static-obstacle-ratio 2:3:5
```

自定义测试支持以下五种变体：

| 变体 | 候选动作 | 智能体 COLREGs 过滤器 | 作用 |
| --- | --- | --- | --- |
| `sac` | SAC | 关闭 | 无辅助的已学习策略 |
| `pn_only` | PN | 关闭 | 经典导引基线 |
| `sac_pn` | SAC 推力 + SAC/PN 舵角融合 | 关闭 | 学习策略与导引混合 |
| `sac_colregs` | SAC | 开启 | SAC 加安全层的性能上界 |
| `sac_pn_colregs` | SAC/PN 融合 | 开启 | 混合导引加安全层的性能上界 |

`CUSTOM_EVALUATION_PN_BASELINE_0823.txt` 的文件名保留了 PN 协议最初建立的日期，但其内部已明确绑定到当前 `0826` 训练配置和断点目录。它保留了参考训练契约中的 600 m 地图、2 艘动态船、3 个静态障碍物和 200 步时域。

### 三种测试文件与快速判断规则内化

三个测试入口现在都会在回合结束后输出同一套中英文终端报告。它们使用当前的 3DOF 环境、观测契约、断点元数据校验和确定性策略推理（`noise=0`），区别如下：

| 文件 | 智能体 COLREGs 过滤器 | 测试场景 | 主要用途 |
| --- | --- | --- | --- |
| `see_trained_once.py` | 保留训练时配置的过滤器 | 与训练配置相同的环境配置 | 可视化完整辅助系统及其安全性能上界 |
| `see_trained_policy_only.py` | 始终关闭智能体过滤器 | 与训练配置相同的环境配置 | 测量 SAC 策略本身是否内化海事规则，是判断规则学习的主要入口 |
| `see_trained_custom.py` | 由所选变体控制 | 从独立测试 TXT 读取 | 测试泛化，并比较 SAC、PN、混合导引和辅助变体 |

在 `policy_only` 测试中，动态障碍船仍保持配置中的脚本化避碰行为，只有智能体自身的动作集过滤器被关闭。规则审计为 shadow-only，只读取会遇报告和策略原始动作，不会修改动作、奖励或动力学。

示例：

```powershell
# 完整训练流程可视化，保留配置中的安全过滤器。
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_once.py SAC_3DOF_ENTITY_ENCODER_0826 --checkpoint last --episodes 1 --max-steps 200

# 纯策略测试；判断规则内化时使用这个模式。
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_policy_only.py SAC_3DOF_ENTITY_ENCODER_0826 --checkpoint last --episodes 10 --max-steps 200 --no-save-gif

# 自定义多变体测试；`sac` 是无辅助策略变体。
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --variants sac,sac_colregs --episodes 10 --max-steps 200
```

### 终端指标与内化判定定义

`evaluation/test_terminal_report.py` 会为单次/纯策略测试以及自定义测试的每个变体输出以下指标：

| 指标 | 含义 |
| --- | --- |
| `success_rate` | 成功抵达目标周边并正常结束的回合比例 |
| `collision_rate` | 因船舶、障碍物或目标碰撞而结束的回合比例 |
| `out_of_bounds_rate` | 因越出地图边界而结束的回合比例 |
| `timeout_rate` | 达到最大步数而结束的回合比例 |
| `mean_reward`、`reward_std` | 回合累计奖励的均值和标准差 |
| `mean_steps` | 结束前平均环境步数 |
| `mean_minimum_clearance_m` | 回合最小净间距均值，已扣除双方实体半径；越小表示安全裕度越低 |
| `risk_event_count` | COLREGs 连续风险事件数，按每艘动态障碍船计数，而不是按时间步计数 |
| `policy_rule_follow_rate` | 原始策略动作满足审计规则方向和反应时限要求的风险事件比例 |
| `safe_resolution_rate` | 无碰撞、无出界、非超时且达到设定净距的风险事件比例 |
| `internalization_success_rate` | 同时满足策略遵规和安全化解的比例：`compliant_and_safe_events / risk_event_count` |
| `filter_active_rate` | 智能体 COLREGs 过滤器改变动作的环境步比例；越高表示对辅助越依赖 |

报告还会按检测到的对遇、交叉会遇、追越等类型分别输出规则跟随率、安全化解率和内化成功率。分母为 0 时显示 `N/A`，这是有意的：特别是 `risk_event_count = 0` 时，本次测试没有提供规则内化证据，不能把它解释为“0% 学会”。无碰撞只能证明该测试中的安全结果，不能单独证明已经内化 COLREGs。论文中应使用多随机种子和足够的风险事件，并比较 `see_trained_policy_only.py` 与 `see_trained_once.py`，或比较自定义测试中的 `sac` 与 `sac_colregs`。

### 测试输出

未设置 `OUTPUT_ROOT` 时，自定义测试输出路径为：

```text
D:\USV\logs\<TRAINING_RUN_NAME>\evaluation\<TEST_NAME>\<timestamp>\
```

主要输出文件包括：

- `effective_environment_config.txt`：最终生效的训练配置与场景配置合并结果。
- `run_manifest.json`：断点、网络契约、种子、PN 参数、视觉参数和汇总元数据。
- `episode_summary.csv` 与 `variant_summary.csv`：任务成功率、碰撞、路径效率、净空、规则介入和 PN 诊断信息。
- `colregs_step_log.csv`、`colregs_event_log.csv` 及汇总 CSV：事件级规则内化与安全化解统计。
- 每个 episode 的 PNG/GIF、轨迹 CSV、动作/状态 CSV。

COLREGs 审计将一次连续会遇视作一个事件，而不是每一步都视作独立事件。目标船是任务终点，故不参与该审计；只审计动态障碍船。审计仅观察动作和状态，不会修改动力学、奖励或最终执行动作。

### 配置规则

- 训练配置决定**网络契约**和所学策略。
- 自定义测试配置决定**测试场景**、输出、可视化、PN 参数和审计阈值。
- 为了进行公平对比，推荐保持 `PRESERVE_TRAINING_DYNAMICS = true`。此时测试将沿用断点训练配置中的 3DOF 时间尺度、执行器限制和动作平滑；与场景尺度相关的安全距离仍按测试配置显式读取。
- `NUM_DYNAMIC_OBSTACLES` 和 `NUM_STATIC_OBSTACLES` 可在泛化测试中修改，但策略每步仍只接收训练时槽数量 `num_ob` 和 `num_static_ob_slots` 所允许的最近实体。
- `STATIC_OBSTACLE_VISUAL_RATIO = 2:3:5` 只影响测试显示。10 个静态障碍物时严格生成 2 个绿洲岛、3 个沙岛和 5 个礁石；稀疏场景保持直观尺寸顺序：1 个为大岛，2 个为大岛加中岛，3 个则三类各一个。

### 验证

在开始长时间实验前，应先检查船体坐标系观测，再运行一个短测试：

```powershell
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python scripts\check_body_frame_observation.py SAC_3DOF_ENTITY_ENCODER_0826
& 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python see_trained_custom.py CUSTOM_EVALUATION_PN_BASELINE_0823.txt --episodes 1 --max-steps 10 --no-gif
```

### 许可证与上游项目

本仓库保留原项目的 [`LICENSE`](LICENSE)。当前项目在上游 RLforUTracking 基础上扩展了 USV 3DOF、海事安全、船体坐标系观测和评估流程。重新发布代码时应保留适用的上游署名。

### Optional DAgger teacher aggregation / 可选 DAgger 规则教师聚合

`DAGGER_ENABLED` is disabled by default. When enabled together with `RULE_BUFFER_ENABLED`, rule teacher metadata (`teacher_valid`, confidence, DCPA/TCPA placeholders, and aggregation iteration) is stored in the independent rule buffer and used only by the actor imitation path. The main replay buffer keeps the SAC transition semantics unchanged: the critic continues to learn `Q(s, raw_action)`, and teacher samples never enter critic targets, entropy tuning, or PER priorities. Existing 0904/0906 configurations explicitly set this option to `false`, so their checkpoints and evaluation flow remain compatible.

`DAGGER_ENABLED` 默认关闭。与 `RULE_BUFFER_ENABLED` 同时启用时，独立规则池会保存规则教师元数据（`teacher_valid`、置信度、DCPA/TCPA 占位字段和聚合轮次），这些信息只服务于 Actor 模仿路径。主经验池仍保持 SAC 语义，Critic 继续学习 `Q(s, raw_action)`；教师样本不会进入 Critic 目标、熵温度更新或 PER 优先级。0904/0906 配置已显式设置为 `false`，因此旧 checkpoint 和测试流程保持兼容。

In filter-coupled DAgger, the existing COLREGs filter is the online teacher:
it computes `rule_action` for every policy-visited state, uses the action only
when a rule decision is active, and stores the policy action, teacher action,
encounter metadata, and DCPA/TCPA in the independent rule buffer. Intervention
records use full confidence; non-intervention records use
`DAGGER_NON_INTERVENTION_CONFIDENCE` and remain low-weight teacher samples.

在过滤器耦合 DAgger 中，现有 COLREGs 过滤器就是在线教师：它对策略访问的
每个状态计算 `rule_action`，只有规则介入时才覆盖执行动作，同时把策略动作、
教师动作、会遇元数据和 DCPA/TCPA 写入独立规则池。介入样本使用完整置信度，
非介入样本使用 `DAGGER_NON_INTERVENTION_CONFIDENCE` 作为低权重教师样本。

### Optional hysteresis state machine and reward guidance / 可选滞回状态机与奖励引导

When `COLREGS_PERSISTENT_STATE_ENABLED=true`, the filter stores an explicit
`RISK_DETECTED`, `GIVE_WAY`, or `STAND_ON` phase. Configurable enter/exit DCPA
and TCPA thresholds provide hysteresis; defaults are conservative and the
feature remains disabled in the 0904/0906 configurations.

启用 `COLREGS_PERSISTENT_STATE_ENABLED=true` 后，过滤器会显式保存
`RISK_DETECTED`、`GIVE_WAY` 或 `STAND_ON` 状态，并使用可配置的进入/退出
DCPA、TCPA 阈值形成滞回。0904/0906 中该开关仍为 `false`。

`RULE_REWARD_ENABLED` is also disabled by default. When enabled, optional
action-alignment, CPA/TCPA risk, and give-way obligation terms are folded into
the existing intervention reward component, preserving the fixed info schema.
Enable these terms only for ablation experiments and compare tracking return,
collision rate, rule compliance, and filter-off performance separately.

`RULE_REWARD_ENABLED` 默认关闭。启用后，动作一致性、CPA/TCPA 风险和让路
义务项会加入现有 intervention 奖励项，不改变固定 info 宽度。该功能应仅用于
消融实验，并分别比较追踪回报、碰撞率、规则合规率和 filter-off 表现。
