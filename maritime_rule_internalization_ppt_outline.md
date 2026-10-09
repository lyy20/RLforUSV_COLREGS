# 海事规则内化：流程与关键模块

> 资料范围：本项目源码、配置文件、README 与 `D:\USV\logs\SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_0911\evaluation\CUSTOM_BODY_FRAME_GENERALIZATION_DEMO\20260913_182700\` 中的配对评估结果。
>
> 说明：未发现单独上传的论文正文。以下内容严格依据项目材料整理；项目材料未给出的信息标注为“原文未明确说明”。

---

## 第 1 页｜封面

**标题**：海事规则内化流程与关键模块

**副标题**：面向 3DOF 无人艇安全追踪的 COLREGs 规则学习框架

**页脚信息**：SAC + COLREGs 动作过滤器 + 双批次规则模仿

**视觉**：深蓝海面、船体坐标系轨迹、青蓝规则流线；不加入项目材料之外的机构或作者信息。

**证据锚点**：`README.md`“Overview / 项目概述”。

---

## 第 2 页｜研究背景与任务定义

**核心结论**：项目研究的是有界海洋环境中的单 USV 安全追踪。

**内容**：

- 任务：到达目标邻域，同时规避静态岛礁、动态船舶、地图边界与危险会遇。
- 控制：连续推力与舵角驱动的欠驱动 3DOF 船舶模型。
- 学习主体：Soft Actor-Critic（SAC）策略。
- 安全层：COLREGs 报告与可选动作过滤器。
- 规则内化判据：关闭智能体过滤器后，策略自身仍能遵规并安全化解风险事件。

**证据锚点**：`README.md`“Main Capabilities / 海事安全层 / 三种测试文件与快速判断规则内化”。

---

## 第 3 页｜规则内化要解决的问题

**问题链**：

1. 规则层能给出会遇义务，但策略原始动作可能不遵规。
2. 过滤器可以在执行时修正动作，但过滤器介入本身不等于策略已经学会规则。
3. 训练阶段需要把规则教师信号送入 Actor，同时保持 SAC Critic、熵温度和主经验池优先级语义稳定。
4. 测试阶段需要区分“安全层保护”与“策略自身内化”。

**关键区分**：

- `sac_colregs`：SAC + 过滤器，表示带安全层的性能上界。
- `sac` / `policy_only`：过滤器关闭，主要用于检验策略是否内化规则。

**证据锚点**：`README.md`“Optional Dual-Batch Rule Replay / 三种测试文件与快速判断规则内化”。

---

## 第 4 页｜海事规则内化总流程

**流程图文字**：

`观测与历史` → `COLREGs 引擎计算相对运动` → `会遇分类与义务` → `动作过滤器生成 rule_action` → `平滑器得到 applied_action` → `环境执行与审计` → `规则记录进入独立经验池` → `Actor 双批次更新` → `policy_only 评估`

**数据边界**：

- 主经验池保存全部环境转移，继续服务 SAC 主训练链。
- 规则经验池只服务 Actor 规则模仿路径。
- 评估审计仅读取状态与动作，不修改动力学、奖励或执行动作。

**证据锚点**：`main.py` 动作链读取函数、规则记录写入逻辑；`evaluation/colregs_event_audit.py` 模块说明；`README.md` 双批次规则经验池说明。

---

## 第 5 页｜规则过滤器：构成方式

**模块**：`safety/colregs_action_filter.py` 中的 `COLREGsActionSetFilter`。

**输入**：

- `world`、`agent`、策略原始动作 `raw_action`。
- 环境开关：过滤器启用、3DOF 控制、COLREGs 启用、船舶身份。

**内部步骤**：

- 将动作裁剪到推力/舵角边界，得到 `constrained_action`。
- 选择最关键的碰撞风险报告。
- 根据会遇类型和让路/直航角色选择规则动作集合。
- 将约束动作投影到规则动作集合中的最近动作，或保持航向航速。

**输出结构**：`raw_action`、`constrained_action`、`rule_action`、`applied_action`、`mode`、`reason`、会遇类型、目标、DCPA/TCPA。

**证据锚点**：`safety/colregs_action_filter.py` 的 `ActionFilterDecision`、`filter_action`、`_give_way_action_set`、`_decision`。

---

## 第 6 页｜规则过滤器：具体作用

**规则动作分支**：

- 对遇：选择向右转向的动作集合。
- 交叉会遇让路：选择避让、避免船头穿越、从船尾通过等方向的动作集合。
- 追越让路：选择保持间距并采取早期实质行动的动作集合。
- 直航船：通常保持航向与航速；紧急距离内使用紧急动作集合。
- 未定义风险：使用保守的风险动作集合。

**作用定位**：

- 执行时提供规则安全上界。
- 输出纯规则 `rule_action`，与后续 EMA 平滑后的 `applied_action` 分离。
- 记录过滤器是否介入，为规则教师样本筛选和依赖度指标提供信号。

**证据锚点**：`safety/colregs_action_filter.py`；`README.md` 对 `filter_active_rate` 的定义。

---

## 第 7 页｜规则状态器：滞回与义务锁存

**实现位置**：`COLREGsActionSetFilter._apply_persistent_rule_state`。

**显式状态**：`RISK_DETECTED`、`GIVE_WAY`、`STAND_ON`。

**构成方式**：

- 进入条件：DCPA/TCPA 进入阈值，或达到紧急距离。
- 锁存内容：目标、会遇类型、让路/直航角色、最近报告。
- 持续更新：风险仍存在时更新实时几何量，但保持初始义务。
- 退出条件：DCPA 与 TCPA 同时离开退出窗口，并连续确认若干步。
- 保护机制：达到最大持有步数时释放状态。

**具体作用**：降低阈值附近反复分类造成的规则动作抖动，并避免刚越过阈值就立即恢复追踪。

**默认状态**：0909/0911 配置启用；0904/0906 配置关闭；这是不同实验配置的事实，不代表所有实验均启用。

**证据锚点**：`safety/colregs_action_filter.py`；`SAC_3DOF_ENTITY_ENCODER_0909.txt`；`SAC_3DOF_ENTITY_ENCODER_0911.txt`；`README.md`“Optional hysteresis state machine”。

---

## 第 8 页｜规则教师数据：动作链与记录字段

**动作链**：

`raw_action` → `constrained_action` → `rule_action` → `smoothed_action` → `applied_action`

**规则经验记录**：

- 状态上下文：`history_obs`、`history_action`、`observation`。
- 动作上下文：原始、约束、纯规则、最终执行动作。
- 规则元数据：会遇类型编码、过滤模式编码、`filter_active`。
- 几何与追踪：DCPA、TCPA、episode、环境步。
- 可选教师元数据：`teacher_valid`、`teacher_confidence`、`dagger_iteration`。

**关键边界**：规则教师动作是纯 `rule_action`；平滑后的执行动作只用于执行链恢复与审计。

**证据锚点**：`main.py` 规则记录构造；`utilities/rule_replay_buffer.py` 的 `push` 字段；`README.md` DAgger 说明。

---

## 第 9 页｜双批次经验池：构成方式

**主经验池 `MainReplayBuffer`**：

- 保存每个环境转移。
- 仍是 Critic、SAC Actor 主目标、熵温度更新和 PER 优先级的唯一来源。

**规则经验池 `RuleReplayBuffer`**：

- 独立容量与独立保存文件。
- 常规模式只收集 `filter_active=True` 的转移。
- 通过 `uniform` 或 `stratified` 采样产生规则 Actor 批次。
- 规则池达到 `RULE_BUFFER_MIN_SIZE` 前，规则项跳过，训练退化为普通 SAC。

**默认配置示例**：0911 使用容量 100000、按 agent 预热 1000、规则 Actor 批次 16、均匀采样、模仿权重 0.05。

**证据锚点**：`utilities/rule_replay_buffer.py`；`main.py` 更新循环；`SAC_3DOF_ENTITY_ENCODER_0911.txt`。

---

## 第 10 页｜模仿学习损失：构成与作用

**总 Actor 目标**：

```text
L_actor = L_SAC(B_main) + RULE_IMITATION_WEIGHT × L_rule(B_rule)
```

**规则项构成**：

- Actor 在规则批次状态上输出均值动作，经 `tanh` 得到归一化动作。
- 与纯规则教师动作计算逐维平方误差。
- 用规则修正幅度形成基础权重，并乘以 `teacher_valid × teacher_confidence`。
- 对加权误差归一化后得到 `L_rule`。

**反向传播边界**：

- 两项损失相加后只反向传播一次。
- 规则批次不进入 Critic 目标、Alpha 更新或 PER 优先级。

**证据锚点**：`algorithms/sac/masac.py` Actor 更新段；`README.md` 双批次公式。

---

## 第 11 页｜训练时序与开关逻辑

**训练时序**：

1. 环境执行并收集主转移。
2. 过滤器介入时，提取规则动作链与会遇元数据。
3. 规则记录写入独立池。
4. 主池满足训练条件后，每个 agent 采样主批次。
5. 规则池满足预热阈值后再采样规则批次。
6. 调用 `maddpg.update(..., rule_samples=...)`，完成 Critic、Actor 与可选 Alpha 更新。
7. 保存网络、主池与规则池；规则池单独恢复。

**兼容性**：规则池是可选路径；关闭时保留历史主经验池语义。旧 checkpoint 开启规则池时，规则池从空开始，SAC 网络与主池可继续使用。

**证据锚点**：`main.py` 规则池初始化、采样、保存与恢复；`README.md` 续训说明。

---

## 第 12 页｜评估：如何判断“内化”

**事件级指标**：

- `policy_rule_follow_rate`：原始策略动作满足规则方向与反应时限的风险事件比例。
- `safe_resolution_rate`：事件无碰撞、无出界、非超时且达到配置净距的比例。
- `internalization_success_rate`：同时满足遵规与安全化解的事件比例。
- `filter_active_rate`：过滤器改变动作的环境步比例，越高表示越依赖辅助。

**审计口径**：

- 连续会遇按每艘动态障碍船计为一个事件，不按时间步重复计数。
- 目标船不纳入规则审计。
- `risk_event_count = 0` 时显示 `N/A`，不能解释为 0% 学会。

**证据锚点**：`README.md`“终端指标与内化判定定义”；`evaluation/colregs_event_audit.py`。

---

## 第 13 页｜项目评估记录：过滤器开关对照

**数据来源**：`20260913_182700/filter_on_off_pair_summary.csv`，100 个配对 episode，0911 训练目录下的自定义泛化评估。

**观察值**：

| 指标 | policy-only | assisted | 说明 |
|---|---:|---:|---|
| 成功率 | 56% | 78% | assisted 提高 22 个百分点 |
| 碰撞率 | 30% | 0% | 该评估中安全层显著减少碰撞 |
| 平均最小净距 | 17.75 m | 24.44 m | assisted 平均更大 |
| 平均回合回报 | -85.91 | -23.35 | assisted 平均更高 |
| 平均风险事件数 | 1.54 | 1.92 | 两种设置事件暴露不同 |
| 内化成功率（有事件样本） | 4.00% | 8.87% | 仅对非空事件样本求均值 |

**结论边界**：

- 该文件证明 100 回合配对评估中的观察差异，不证明规则模仿项单独造成差异。
- assisted 的安全结果包含过滤器作用，不能替代 policy-only 内化判断。
- 项目材料未提供跨随机种子、消融实验或统计显著性结论，原文未明确说明。

**证据锚点**：上述 CSV；`README.md` 对过滤器开关与内化指标的解释。

---

## 第 14 页｜方法贡献与已知边界

**从项目实现可归纳的贡献**：

- 把规则评估、执行过滤、规则教师数据与策略评估拆成可追踪的数据链。
- 用持久化规则状态处理风险阈值附近的会遇义务连续性。
- 通过独立规则经验池把规则模仿限定在 Actor，保持 SAC 主训练语义稳定。
- 用事件级审计区分“过滤器保护”与“策略内化”。

**边界与未明确项**：

- 规则动作集合是项目内的离散候选集合；其与真实船舶操纵性能的充分性未在材料中证明。
- 0909/0911 的规则奖励与 DAgger 开关均关闭，相关增益未在本项目材料中给出。
- 未发现完整论文实验章节、基线显著性检验或跨任务泛化统计，原文未明确说明。

**证据锚点**：`README.md`、`safety/`、`algorithms/sac/masac.py`、0911 配置。

---

## 第 15 页｜结论与展望

**结论**：

- 海事规则内化流程由“规则判断—动作过滤—教师记录—双批次 Actor 更新—过滤器关闭评估”组成。
- 规则过滤器负责执行时安全修正；规则状态器负责义务连续性；模仿损失负责把规则动作约束传给 Actor；双批次经验池负责隔离主 SAC 数据与规则教师数据。
- 0911 的 100 回合配对日志显示 assisted 相比 policy-only 有更高成功率、更低碰撞率，但这属于安全层辅助结果，不能单独证明策略已经内化规则。

**展望**：

- 按项目已有指标补充多随机种子、足够风险事件与置信区间。
- 对规则经验池、模仿权重、持久化状态与规则奖励做独立消融。
- 比较 policy-only 与 assisted，并报告过滤器介入率与事件级内化成功率。

**结束语**：谢谢。

**证据锚点**：项目 README、源码与 0911 配对评估记录。
