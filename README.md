# RLforUSV_COLREGS

面向**无人艇（USV）海事规则内化**的强化学习研究代码库：在欠驱动 3DOF 动力学与 COLREGs 约束下，训练策略在**撤除安全层后**仍能履行避碰义务，并提供可复现、可证伪的内化判据。

---

## 1. 研究定位

| 项 | 内容 |
| --- | --- |
| 任务 | 目标追踪 / 抵近（后续扩展围捕与多艇协同） |
| 规则 | COLREGs（对遇 / 交叉让路 / 交叉直航 / 追越），Rule 8「及早、明显」 |
| 核心问题 | **安全过滤器的存在使得「内化」与「被约束」在行为上不可区分** |
| 解决路径 | ① 设计会遇 + 事件级判据；② 覆盖感知的主动规则教师；③ 角色分层下的义务再判定 |

## 2. 关键实测指标（当前版本）

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 教师发声密度 `rule_filter_active_ratio` | **3.62%** | 0911 全量训练日志（n=23,456）——教师极少发声 |
| 会遇密度（原做法） | **0.0000 / 回合**，≥1 次会遇的 reset **0.00%** | 200 resets 实测 |
| 会遇密度（会遇生成器） | **3.175 / 回合**，≥1 次会遇的 reset **100.00%** | 120 resets 实测（P0-1/P0-2 验收） |
| 观测维度 | 34 → **53** | `15 + 7·N_d + 5·N_s`（N_d=4, N_s=2） |
| 规则模仿权重 | 0.05 | 0911 配置 |
| CTRIP | `ICM=False`, `CTIP_active=False` | 0911 规则实验**未启用** |

## 3. 目录结构

```
main.py                         训练入口（配置解析 / 主循环 / 日志 / 检查点）
see_trained_custom.py           评估入口（多变体 + COLREGs 审计）
see_trained_policy_only.py      纯策略评估（撤除安全层）
see_trained_once.py             单回合评估

multiagent/
  core.py                       World / Entity / Agent / Landmark / Obstacle；会遇评估
  scenarios/usv_tracking_base.py 场景与任务：实体、观测、奖励、终止、机动更新
algorithms/sac/                 MASAC / SACAgent / EntityEncoderNetwork / ICM
safety/
  colregs.py                    COLREGs 引擎（会遇分类 / 义务 / DCPA-TCPA）
  colregs_action_filter.py      动作集合投影（硬约束）
utilities/
  buffer.py                     PER（SummTree）
  rule_replay_buffer.py         规则经验池（teacher_valid / teacher_confidence / dagger_iteration）
  encounter_wrapper.py          会遇生成包装器（CPA 倒推，P0-1）
  teacher_query.py              覆盖感知教师查询（判据 / 置信度 / 加权损失）
  scenario_config.py            物理尺度配置（含会遇生成参数）
evaluation/
  colregs_event_audit.py        事件级审计
  test_terminal_report.py       终端报告与 Wilson 区间
dynamics/usv_3dof.py            欠驱动 3DODF 模型
guidance/                       比例导引（评估基线）
tracking/target_pf.py           range-only 定位（PF / LS）
scripts/                        自检与验收脚本（见第 6 节）
```

## 4. 环境与学习契约

### 4.1 动作链

```
raw → constrained → rule → smoothed(EMA) → actuator
```

教师 = 纯 `rule_action`；`agent_colregs_action_filter_enabled` 控制安全层开关。

### 4.2 观测

- 契约：`3dof_body_frame_v2_action_state`
- 维度：`dim = 15 + 7·N_d + 5·N_s`
- 动态槽位按**风险**（TCPA → DCPA → 距离）排序，而非按距离

### 4.3 奖励与终止

| 项 | 说明 |
| --- | --- |
| Reward v1 | 捕获 + 保持稀疏奖励（用估计位置） |
| **Reward v2** | task-driven pursuit（**当前生效入口**） |
| 终止原因 | `out_of_bounds` / `collision` / `target_collision` / `success` / `timeout` / `hold_success`（扩展，默认关闭） |
| 阶段 | `agent.phase ∈ {transit, approach, capture}` |
| 角色 | `vessel_role ∈ {ownship, target, dynamic_obstacle, static_obstacle, ...}` |

### 4.4 规则内化管线

```
COLREGsEngine（会遇分类 / 义务 / DCPA-TCPA）
  → COLREGsActionSetFilter（硬投影，filter_active）
  → 持久义务状态机（迟滞 + 最大保持）
  → RuleReplayBuffer（teacher_valid / teacher_confidence）
  → L_actor = L_SAC(B_main) + w · L_rule(B_rule)（按置信度加权）
  → COLREGsEventAudit → 终端报告（Wilson CI）
```

## 5. 快速开始

```bash
# 训练（会遇生成版）
python main.py SAC_3DOF_ENTITY_ENCODER_0911_DENSE.txt

# 训练（基线对照）
python main.py SAC_3DOF_ENTITY_ENCODER_0911_BASELINE.txt

# 评估：多变体 + COLREGs 审计
python see_trained_custom.py CUSTOM_EVALUATION_TEMPLATE.txt --variants sac,sac_colregs

# 评估：撤除安全层（纯策略）
python see_trained_policy_only.py SAC_3DOF_ENTITY_ENCODER_0911_DENSE.txt
```

环境依赖：Python 3.7 / PyTorch 1.13（CUDA 11.7）/ numpy / matplotlib / tensorboardX。

## 6. 验收脚本

| 脚本 | 用途 | 期望输出 |
| --- | --- | --- |
| `scripts/measure_encounter_density.py` | 会遇密度测量 | 宽会遇/回合、≥1 次会遇 reset 占比 |
| `scripts/diagnose_encounter_lateral.py` | 横向偏移 / 速度诊断 | 分布统计 |
| `scripts/analyze_designed_encounter.py` | A0 设计会遇判据 | 首次合规步 / 合规比 / 反向比 |
| `scripts/sweep_designed_thresholds.py` | 阈值稳健性扫描 | 25 组阈值下的判定集合 |
| `scripts/run_a0_eval.py` | 一条命令＝评估 + A0 指标 | 终端报告 + `designed_encounter.json` |
| `scripts/check_risk_slot_ordering.py` | 观测槽位风险排序 | PASS |
| `scripts/check_body_frame_observation.py` | 观测契约 | PASS（dim=53） |
| `scripts/check_role_phase_termination.py` | 阶段 / 终止接口 | PASS |
| `scripts/check_done_state_isolation.py` | per-agent 终止隔离与防腐门 | PASS |

## 7. 配置速查（0911 / DENSE）

| 键 | 0911 | DENSE |
| --- | --- | --- |
| `encounter_generation_mode` | 0（关闭） | **1（CPA 倒推生成）** |
| `encounter_tcpa_min/max_s` | — | 60 / 120 |
| `encounter_dcpa_min/max_m` | — | 10 / 60 |
| `encounter_accept_tcpa_max_s` / `dcpa_max_m` | — | 120 / 60 |
| `num_ob` / `num_static_ob_slots` | 2 / 1 | 4 / 2 |
| `DAGGER_ENABLED` | false | false |
| `PRE_TRAINED` | true | **false** |
| `colregs_risk_time_horizon` / `min_cpa_distance` | 60 s / 30 m | 60 s / 30 m（待 P0-3 对齐） |

## 8. 实验状态

| 阶段 | 项 | 状态 |
| --- | --- | --- |
| **P0-8** | 基线配置与实验纪律 | ✅ |
| **P0-1/P0-2** | 会遇生成 + 接受判据 | ✅ 已验收（0.0000 → 3.175/回合） |
| P0-7 | 观测槽位 2→4 生效复测 | ✅ 代码与验收脚本 |
| P0-3 | 尺度对齐（激活 1–2.5 km / 进入 250 m） | ⬜ 待做（目标：教师密度 >15%） |
| P0-4 | M13 判据接口化（A0 进评估器） | ⬜ 脚本已有，待接入报告 |
| P0-5 | Imazu 22 案例评测协议 | ⬜ 待做 |
| P0-6 | AIS 真实会遇分布校准 | ⬜ 待做 |
| P0-9 | 课程接口（默认关闭） | ⬜ 待做 |
| P1 | CTRIP 开关消融 / 规则软奖励 / Rule 8 奖励 / 三臂对照 | ⬜ 实验稳定后 |
| P2 | CMDP / 意图条件化义务 / 特权蒸馏 / 围捕与角色分层 | ⬜ 后续 |

## 9. 版本留存约定

**每次对新增内容验收之前，必须先提交一版**，保证任何改动都可无痛回退：

```bash
git add -A
git commit -m "<阶段>：<改动>（<验收指标>）"
git push
```

大文件（`*.pt` / `*.file` / `smoke_artifacts/` / 评估输出 / 图片）不入库，见 `.gitignore`。

## 10. 参考实现与致谢

本项目的机制设计参考了以下开源实现（用于对标行业基线，非代码复制）：

| 参考 | 借鉴点 |
| --- | --- |
| [ntnu-itk-autonomous-ship-lab/rlmpc](https://github.com/ntnu-itk-autonomous-ship-lab/rlmpc) | 规则处理器参数、规则软奖励族（含 Rule 8 明显动作） |
| [ntnu-itk-autonomous-ship-lab/colav-simulator](https://github.com/ntnu-itk-autonomous-ship-lab/colav-simulator) | 会遇参数化生成、坏场景筛除 |
| [yaspang/RL_CORALL_extension](https://github.com/yaspang/RL_CORALL_extension) | CPA 倒推会遇生成、Imazu 课程、意图仲裁结构 |
| [imasmitja/DRL4AUV](https://github.com/imasmitja/DRL4AUV) | 多智能体粒子环境与追踪任务骨架 |

## 11. 免责声明

本代码库为学术研究用途。COLREGs 的工程化实现（会遇分类阈值、义务判定）是**近似模型**，不等同于法律文本解释，亦不得直接用于实船决策。
