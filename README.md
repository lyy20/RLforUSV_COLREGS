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

## 2. 关键实测指标（V6 环境层，2026-10-10）

| 指标 | 数值 | 说明 / 小结论 |
| --- | --- | --- |
| **会遇数/回合** | **4.00**（对照 2.38） | CPA 反推生成比随机放置 **+68%** |
| **教师发声率** | **14.06–15.03%**（对照 10.84–11.36%） | 三点复测一致，**+30%** |
| 会遇形成率 | 100%（两臂均 100%） | **不区分**，不作为声明指标 |
| 对遇类型占比 | **25%**（8/32） | 与 AIS 实测 29% 接近 ✅ |
| **本船物理能力** | **5.89 m/s**（旧参数 1.40） | 配置上限 6.0 已可实际达到（旧版只是「上限」，非能力） |
| 观测维度 | 53（`15 + 7·N_d + 5·N_s`，N_d=4, N_s=4） | 动态槽 2→4、静态槽 1→4 |
| TB 标量数 | **29**（原 122） | 记录精简 −76%，另**新增 4 个损失标量**（原被 NoOp 日志吞掉） |
| 训练成本 | **2.97 s/回合**（对照 1.49） | 会遇生成 + 验证门带来 **≈2×** 墙钟 |
| 历史基线（旧环境） | 教师发声 3.62%；会遇 0.0000/回合 | **仅供对照**，不适用于 V6 |

## 3. 最新修改（V6 环境层，2026-10-10）

### 3.1 物理能力修复（关键）

| 问题 | 修复 |
| --- | --- |
| `usv_max_surge_speed=6.0` 只是**上限**，实际只能跑 **1.40 m/s**（世界层硬编码阻尼 0.25 + 3DOF 阻力过大） | 世界阻尼接入配置（0.25→**0.02**）；3DOF 阻力接入配置（18/6→**5/1.5**）；推力 80 N；**已存在 agent 的模型热替换**（`PROPAGATE_USV3DOF`） |
| 实测结果 | 满推力剖面 1 s:1.48 → 5 s:5.42 → 10 s:**5.85** → 稳态 **5.89 m/s** ✅ |

### 3.2 会遇生成与验证

| 项 | 内容 |
| --- | --- |
| 生成方式 | CPA 反推（TCPA 60–120 s、DCPA 10–60 m）；类型权重 HO **3.0** / CR 2.4 / OT 0.6 |
| **验证门** | reset 后短滚 **40 步** + CPA 外推，容差 `max(设计×2+50 m, 300 m)`，最多 **6** 次重放；**限内接受率 87%** |
| 目标运动方式 | **7 种**：静止 / 直线匀速 / 航点 / 逃逸 / 随机机动 / 蛇形 / 圆周绕行（权重 0.8/1.2/1.2/1.2/1.2/1.0/0.8） |
| 初始朝向 | **随机**（标准差实测 92–115°，均匀理论 104°） |
| 直航语义 | 会遇目标船**保持航向航速**（Rule 17 直航义务），不再自主避让 |

### 3.3 观测与任务

| 项 | 值 |
| --- | --- |
| 动态槽 / 静态槽 | **4 / 4**（网络契约变更，需重训） |
| `observation_velocity_scale` | 0.005 → **0.012**（速度提到 6 m/s 后，对遇最大闭合 9.4 m/s 会饱和） |
| 地图 / 回合 | **3 km** / **600 步**（闭合 3.7 m/s，p90 情形 544 s + 50 余量） |
| 目标速度 | 1.2–2.2 m/s（下调，避免任务过难） |
| 本船 | 20 m 级；**0 速度起步 + 随机朝向** |

### 3.4 参数治理

| 项 | 内容 |
| --- | --- |
| 冲突登记表 | `configs/parameter_conflict_registry.json`（12 条：含语义/尺度/真冲突分类与处置） |
| 基线方案 | 以 **colav-simulator** 为基线（键级可映射），其余来源只做单点校正 |
| 责任声明 | 参数改动前必须过登记表；冲突不得静默混用 |

## 4. 验收实验记录（附小结论）

### 4.1 会遇形成的 A/B（同环境、同占位策略）

| 指标 | GEN_ON | GEN_OFF | 小结论 |
| --- | --- | --- | --- |
| 会遇数/回合 | **4.00** | 2.38 | **+68%**，生成器有效 |
| 会遇形成率 | 100% | 100% | **不区分**，不可作声明 |
| 对遇占比 | **25%** | 6% | 权重上调后接近 AIS 分布 |

### 4.2 训练侧 A/B（400 回合，三点复测）

| 指标 | GEN_ON | GEN_OFF | 小结论 |
| --- | --- | --- | --- |
| 教师发声率 | 15.03 → 14.06 → **14.55%** | 11.36 → 10.84 → **11.04%** | **+30%**，方向稳定 |
| 平均回报 | −104.6 → **−107.0** | −148 → **−155** | 生成臂更好 |
| 最小距离中位 | **291 m** | 370 → 387 m | 会遇更近 |
| 训练墙钟 | **19:47 / 400 回合** | **9:56 / 400 回合** | 代价 **≈2×** |

> ⚠️ **边界**：以上为**训练期诊断量**（单 seed、~400 回合），可用于支撑**环境设计**，**不得**作为策略质量结论。

### 4.3 设计 CPA 的可行性核查（结论性）

| 现象 | 数值 | 结论 |
| --- | --- | --- |
| 设计 DCPA vs 实际最小距离 | 37 m vs 330 m（**8.7×**） | **CPA 不可指定** |
| 连**静止目标** | 42 m vs 407 m（**9.66×**） | 偏差来自**本船航迹**（追击 ⇒ 曲线），非他船 |
| 对比：标准会遇测试（Imazu/colav） | 双方脚本化**近直线** ⇒ 可指定 | 差异在**任务设定**，不在能力 |

> 因此本项目采用 **B 语义**：设计规定**初始几何**（类型/方位/距离/初速）+ 验收门「会遇确实形成」；
> **实际 CPA 作为测量量报告**，不承诺具体数值。

### 4.4 验证成本（提速测试）

| 设置 | 限内接受率 | reset 墙钟 | 结论 |
| --- | --- | --- | --- |
| **40 步 / 6 次**（保留） | **87%** | 0.912 s | 保证够强 |
| 20 步 / 3 次（提速档） | 43% | 0.409 s | 快 2.2×，但丢掉近一半保证 ⇒ **不划算** |

### 4.5 记录精简验证

| 项 | 优化前 | 优化后 | 结论 |
| --- | --- | --- | --- |
| 标量总数 | 122 | **29** | −76%；`performance/*` 72 个全关 |
| 每步热写标签 | 2457 次/150 回合 | **0** | 消除 I/O 热点 |
| 损失标量 | **0**（被吞） | **4** | 训练健康可见 |

## 5. 正式实验矩阵

### 5.1 臂定义（单变量原则）

| 臂 | 名称 | 与父臂差异 | 目的 |
| --- | --- | --- | --- |
| **A0** | 纯 SAC | `agent_colregs_action_filter_enabled=false` + `RULE_BUFFER_ENABLED=False` | 内化上界对照 |
| **A1** | 仅安全层 | filter=true，规则缓冲关 | 借来的合规基线 |
| **A2** | 安全层 + 规则模仿（主臂） | `RULE_IMITATION_WEIGHT=0.05` | 主方法 |
| **A3** | 模仿权重消融 | 仅 `RULE_IMITATION_WEIGHT=0.20` | 权重敏感性 |
| **A4** | 模仿 + DAgger（可选） | 仅 `DAGGER_ENABLED=true` | 教师聚合消融 |
| **E0** | 环境对照（可选） | 仅 `encounter_generation_mode=0` | 环境层贡献 |

### 5.2 矩阵与成本（实测 2.97 s/回合；**2 进程并发为上限**）

| 层级 | 臂 | seed | 回合 | run 数 | 2 并发墙钟 |
| --- | --- | --- | --- | --- | --- |
| 第一阶段（筛选） | A0/A1/A2/A3 | 1,2,3 | 100k | 12 | **≈20.6 天** |
| 第二阶段（正式） | 一阶段前 2 名 | 1,2,3 | 500k | 6 | **≈51.7 天** |
| 可选 | A4 / E0 | 1,2,3 | 100k | 6 | ≈10.3 天 |

**配置命名**：`V6_<臂>_S<seed>.txt`，基于 `SAC_3DOF_ENTITY_ENCODER_0911_V6.txt`（冻结）。

### 5.3 训练后评估协议（判据冻结）

| 变体 | 用途 |
| --- | --- |
| `sac`（关安全层） | **内化判据**：策略独立合规能力 |
| `sac_colregs`（开安全层） | 系统级性能 |

命令：`python see_trained_custom.py <cfg> --variants sac --episodes 80 --no-gif`，输出 **M13 六项联合报告**。
必须标注：checkpoint 回合数、seed 数、事件数、CI。

### 5.4 验收判据

| 阶段 | 判据 |
| --- | --- |
| 第一阶段（100k） | ①教师发声率 ≥10% ②loss 有限 ③出界 ≤5% ④A2 相对 A1 正向 |
| 第二阶段（500k） | ①3 seed 曲线进平台 ②**关安全层**内化率 A2>A1（带 CI）③反向动作比 A2<A1 |

### 5.5 执行纪律

1. **同时最多 2 进程**，错峰 ~40 s 启动（3 进程会 CUDA 报错）；
2. **退出码不可靠**（多次出现「跑满但 exit=1」）⇒ 以**产物**为准（TB 标量、回合计数）；
3. 每 run 结束归档 `docs/results/<臂>_<seed>.json` + 一行 commit；
4. 留 ≥8 GB 内存；阶段间**冻结配置**，禁止边跑边改。

## 6. 目录结构

```
main.py                         训练入口（配置解析 / 主循环 / 日志 / 检查点）
see_trained_custom.py           评估入口（多变体 + COLREGs 审计）
see_trained_policy_only.py      纯策略评估（撤除安全层）
multiagent/                     World / 场景 / 观测 / 奖励 / 终止
algorithms/sac/                 MASAC / SACAgent / EntityEncoder / ICM
safety/                         COLREGs 引擎 + 动作集合投影
utilities/                      buffer / rule_replay_buffer / encounter_wrapper / vessel_identities / scenario_config
evaluation/                     colregs_event_audit / test_terminal_report / joint_report
configs/                        vessel_identities.json / parameter_conflict_registry.json
docs/plan/                      规划与方法学文档（见下）
docs/plan/服务器选型与Linux迁移.md  服务器比价 + 成本模型 + Linux 迁移清单（租服务器看这份）
requirements-linux.txt          Linux 训练依赖（版本由服务器端 pip freeze 锁定）
scripts/                        自检与验收脚本
scripts/server_bootstrap.sh     服务器一键部署 + 冒烟 + 环境快照
scripts/bench_server.sh         服务器标定：并发进程数 × workers 的真实吞吐与峰值显存
```

## 7. 环境与学习契约

### 7.1 动作链

```
raw → constrained → rule → smoothed(EMA α=0.35) → actuator
```

教师 = 纯 `rule_action`；`agent_colregs_action_filter_enabled` 控制安全层开关（训练期可关）。

> 注：α=0.35 与真实船体航向时间常数 `T_chi=3 s`（α≈0.283）一致，属**物理正确**，不是「稀释教师」。

### 7.2 观测

- 契约：`3dof_body_frame_v2_action_state`；维度 `15 + 7·N_d + 5·N_s`（V6：N_d=4, N_s=4 ⇒ 63）
- 动态槽按**风险**（TCPA → DCPA → 距离）排序

### 7.3 规则内化管线

```
COLREGsEngine（会遇分类 / 义务 / DCPA-TCPA）
  → COLREGsActionSetFilter（硬投影，filter_active）
  → 持久义务状态机（迟滞 + 最大保持）
  → RuleReplayBuffer（teacher_valid / teacher_confidence）
  → L_actor = L_SAC(B_main) + w · L_rule(B_rule)（置信度加权）
  → COLREGsEventAudit → 终端报告（Wilson CI）
```

## 8. 快速开始

```bash
# 训练（V6 冻结环境）
python main.py SAC_3DOF_ENTITY_ENCODER_0911_V6.txt

# 评估：关安全层（内化判据）
python see_trained_custom.py curriculum/A0_ATTRIB_ARM_A.txt --variants sac --episodes 80 --no-gif
```

环境依赖：Python 3.7 / PyTorch 1.13（CUDA 11.7）/ numpy / matplotlib / tensorboardX。

## 9. 验收脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/measure_encounter_density.py` | 会遇密度测量 |
| `scripts/measure_encounter_formation.py` | **会遇形成 A/B**（M1/M2/M3） |
| `scripts/check_encounter_realized.py` | **设计 vs 实际 CPA**（支持 zero/cruise/pursuit 占位） |
| `scripts/measure_verify_cost.py` | **验证门接受率与成本** |
| `scripts/check_param_realism.py` | **参数现实性门**（CPA/L、探测/L、Froude、窗内航程/L） |
| `scripts/check_config_discipline.py` | **配置纪律门**（PRE_TRAINED vs checkpoint） |
| `scripts/check_imazu_cases.py` | Imazu 22 案例引擎复现（4/4） |
| `scripts/calibrate_ais_distribution.py` | AIS 实测会遇分布校准 |
| `scripts/run_a0_eval.py` | 评估 + A0 指标 + **M13 六项联合报告** |
| `scripts/summarize_arms.py` | 多臂汇总 |

## 10. 规划与方法学文档（docs/plan）

| 文档 | 内容 |
| --- | --- |
| 环境层声明与冻结.md | **可声明 / 不可声明清单 + 冻结值** |
| 正式实验矩阵.md | 实验矩阵全文 |
| 参数层方法学-冲突清单与基线选择.md | 8 项冲突登记 + 基线选择（colav） |
| 参数改动清单与任务层方向.md | 170 键逐键对比（35 改 / 135 未改） |
| 环境参数全量清册与现实化方案.md | 62 项环境参数 + 现实锚点 |
| 船舶身份参数集方案.md | 5 类真实船型（含出处）+ 双边界推导 |
| 任务层参数分析.md / 任务层参数评审表.md | 任务层清册、冲突登记、修改方向 |
| 回滚审视-预设CPA是否可行.md | **CPA 不可指定的完整论证** |
| V6实现与验收.md | V6 功能与验收记录 |
| 实验有效性边界声明.md | **策略级结论的有效性边界（必读）** |

## 11. 实验状态

| 阶段 | 项 | 状态 |
| --- | --- | --- |
| P0-1/2 | 会遇生成 + 接受判据 | ✅ 4.00/回合（+68%） |
| P0-3 | 教师覆盖率（尺度对齐） | ✅ **+30%**（三点复测） |
| P0-4 | M13 判据接口化 | ✅ `evaluation/joint_report.py` |
| P0-5 | Imazu 22 案例 | ✅ 引擎复现 4/4 |
| P0-6 | AIS 真实分布校准 | ✅ 196 会遇 → CPA/L 4.5–9.9 |
| P0-7 | 观测槽位 | ✅ 4 / 4（契约变更，需重训） |
| P0-8 | 配置纪律门 | ✅ `check_config_discipline.py` |
| 环境层 | 物理能力 / 7 运动方式 / 随机朝向 / 验证门 | ✅ 已冻结 |
| 正式实验 | A0–A4 矩阵 | ⏳ **待执行（用户自行开跑）** |

## 12. 版本留存约定

**每次验收前必须先提交一版**，保证可无痛回退：

```bash
git add -A
git commit -m "<阶段>：<改动>（<验收指标>）"
git push
```

大文件（`*.pt` / `*.file` / 评估输出 / 图片）不入库，见 `.gitignore`。

## 13. 参考实现与致谢

| 参考 | 借鉴点 |
| --- | --- |
| [rlmpc](https://github.com/ntnu-itk-autonomous-ship-lab/rlmpc) | 规则处理器参数、规则软奖励族 |
| [colav-simulator](https://github.com/ntnu-itk-autonomous-ship-lab/colav-simulator) | **会遇参数化基线 + 4 个真实船型（含质量/惯量/推力）** |
| [RL_CORALL_extension](https://github.com/yaspang/RL_CORALL_extension) | CPA 倒推会遇生成、Imazu 课程 |
| [DRL4AUV](https://github.com/imasmitja/DRL4AUV) | 多智能体粒子环境与追踪任务骨架 |

## 14. 免责声明

本代码库为学术研究用途。COLREGs 的工程化实现（会遇分类阈值、义务判定）是**近似模型**，不等同于法律文本解释，亦不得直接用于实船决策。