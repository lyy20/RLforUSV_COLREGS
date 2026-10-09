import csv
import matplotlib.pyplot as plt
import numpy as np




# ---------------------- 1. 模拟数据（替换为真实数据） ----------------------
# 训练步数，假设最大2.0M，步长0.01M
training_episodes = np.arange(0, 1.001, 0.001)
num_steps = len(training_episodes)
# method = ['','PER_','ICM_','CTIP_']
method = ['','PER_','CTIP_']
true_method = ['','PER_','CTRIP_']
# 模拟三种算法（SAC、PER - SAC、ICM - SAC、CTIP - SAC ）的Reward曲线
# np.random.seed(0)
# 创建十次实验的数据
num_experiments = 3

SEED = [ '2','4','7' ]

num_time = 1000
doc_name = '\SAC7.26_'
num = 0
reward = []
reward_mean = []
reward_max = []
reward_min = []
for i in range(len(method)):
    reward.append(np.zeros((num_experiments, num_steps)))
# 为每种算法生成十次实验的数据
for index_method in range(len(method)):
    for i in range(num_experiments):
        # print(r'D:\USV\data_doc\SAC7.6'+ method[index_method] + str(i + 1) + r'\Reward.csv')
        with open(r'D:\USV\data_doc'+ doc_name + method[index_method] + SEED[i] + r'\Reward.csv', 'r', encoding='utf-8') as file:
            # 创建CSV读取器
            reader = csv.reader(file)
            # 跳过表头
            header = next(reader)
            # 逐行读取数据并添加到列表中
            for j , row in enumerate(reader):
                if j <= num_time:
                    value = float(row[-1])
                    reward[index_method][i][j] = reward[index_method][i][j] + value

    reward_mean.append(np.mean(reward[index_method], axis=0))
    reward_max.append(np.max(reward[index_method], axis=0))
    reward_min.append(np.min(reward[index_method], axis=0))

# 模拟State Coverage曲线
StateCoverage = []
StateCoverage_mean = []
StateCoverage_max = []
StateCoverage_min = []
for i in range(len(method)):
    StateCoverage.append(np.zeros((num_experiments, num_steps)))
# 为每种算法生成十次实验的数据
for index_method in range(len(method)):
    for i in range(num_experiments):
        # print(r'D:\USV\data_doc\SAC7.6'+ method[index_method] + str(i + 1) + r'\Error.csv')
        with open(r'D:\USV\data_doc'+ doc_name + method[index_method] + SEED[i] + r'\StateCoverage.csv', 'r', encoding='utf-8') as file:
            # 创建CSV读取器
            reader = csv.reader(file)
            # 跳过表头
            header = next(reader)
            # 逐行读取数据并添加到列表中
            for j , row in enumerate(reader):
                if j <= num_time:
                    value = float(row[-1])
                    StateCoverage[index_method][i][j] = StateCoverage[index_method][i][j] + value 

    StateCoverage_mean.append(np.mean(StateCoverage[index_method], axis=0))
    StateCoverage_max.append(np.max(StateCoverage[index_method], axis=0))
    StateCoverage_min.append(np.min(StateCoverage[index_method], axis=0))


# 模拟Target error曲线
error = []
error_mean = []
error_max = []
error_min = []
for i in range(len(method)):
    error.append(np.zeros((num_experiments, num_steps)))
# 为每种算法生成十次实验的数据
for index_method in range(len(method)):
    for i in range(num_experiments):
        # print(r'D:\USV\data_doc\SAC7.6'+ method[index_method] + str(i + 1) + r'\Error.csv')
        with open(r'D:\USV\data_doc'+ doc_name + method[index_method] + SEED[i] + r'\Error.csv', 'r', encoding='utf-8') as file:
            # 创建CSV读取器
            reader = csv.reader(file)
            # 跳过表头
            header = next(reader)
            # 逐行读取数据并添加到列表中
            for j , row in enumerate(reader):
                if j <= num_time:
                    value = float(row[-1])
                    error[index_method][i][j] = error[index_method][i][j] + value 

    error_mean.append(np.mean(error[index_method], axis=0))
    error_max.append(np.max(error[index_method], axis=0))
    error_min.append(np.min(error[index_method], axis=0))

# 模拟Agent out曲线
AgentOut = []
AgentOut_mean = []
AgentOut_max = []
AgentOut_min = []
for i in range(len(method)):
    AgentOut.append(np.zeros((num_experiments, num_steps)))
# 为每种算法生成十次实验的数据
for index_method in range(len(method)):
    for i in range(num_experiments):
        # print(r'D:\USV\data_doc\SAC7.6'+ method[index_method] + str(i + 1) + r'\Error.csv')
        with open(r'D:\USV\data_doc'+ doc_name + method[index_method] + SEED[i] + r'\AgentOut.csv', 'r', encoding='utf-8') as file:
            # 创建CSV读取器
            reader = csv.reader(file)
            # 跳过表头
            header = next(reader)
            # 逐行读取数据并添加到列表中
            for j , row in enumerate(reader):
                if j <= num_time:
                    value = float(row[-1])
                    AgentOut[index_method][i][j] = AgentOut[index_method][i][j] + value

    AgentOut_mean.append(np.mean(AgentOut[index_method], axis=0))
    AgentOut_max.append(np.max(AgentOut[index_method], axis=0))
    AgentOut_min.append(np.min(AgentOut[index_method], axis=0))


# 模拟LandmarkCollisions曲线
LandmarkC = []
LandmarkC_mean = []
LandmarkC_max = []
LandmarkC_min = []
for i in range(len(method)):
    LandmarkC.append(np.zeros((num_experiments, num_steps)))
# 为每种算法生成十次实验的数据
for index_method in range(len(method)):
    for i in range(num_experiments):
        # print(r'D:\USV\data_doc\SAC7.6'+ method[index_method] + str(i + 1) + r'\Error.csv')
        with open(r'D:\USV\data_doc'+ doc_name + method[index_method] + SEED[i] + r'\LandmarkC.csv', 'r', encoding='utf-8') as file:
            # 创建CSV读取器
            reader = csv.reader(file)
            # 跳过表头
            header = next(reader)
            # 逐行读取数据并添加到列表中
            for j , row in enumerate(reader):
                if j <= num_time:
                    value = float(row[-1])
                    LandmarkC[index_method][i][j] = LandmarkC[index_method][i][j] + value

    LandmarkC_mean.append(np.mean(LandmarkC[index_method], axis=0))
    LandmarkC_max.append(np.max(LandmarkC[index_method], axis=0))
    LandmarkC_min.append(np.min(LandmarkC[index_method], axis=0))


# 模拟ObstacleCollisions曲线
ObstacleC = []
ObstacleC_mean = []
ObstacleC_max = []
ObstacleC_min = []
for i in range(len(method)):
    ObstacleC.append(np.zeros((num_experiments, num_steps)))
# 为每种算法生成十次实验的数据
for index_method in range(len(method)):
    for i in range(num_experiments):
        # print(r'D:\USV\data_doc\SAC7.6'+ method[index_method] + str(i + 1) + r'\Error.csv')
        with open(r'D:\USV\data_doc'+ doc_name + method[index_method] + SEED[i] + r'\ObstacleC.csv', 'r', encoding='utf-8') as file:
            # 创建CSV读取器
            reader = csv.reader(file)
            # 跳过表头
            header = next(reader)
            # 逐行读取数据并添加到列表中
            for j , row in enumerate(reader):
                if j <= num_time:
                    value = float(row[-1])
                    ObstacleC[index_method][i][j] = ObstacleC[index_method][i][j] + value 

    ObstacleC_mean.append(np.mean(ObstacleC[index_method], axis=0))
    ObstacleC_max.append(np.max(ObstacleC[index_method], axis=0))
    ObstacleC_min.append(np.min(ObstacleC[index_method], axis=0))


# 竖线位置（示例，对应图中a、b、c、d、e位置，按需调整）
vline_positions = [0.2, 0.4, 0.6, 0.8, 1.0]  
vline_labels = ['a', 'b', 'c', 'd', 'e']

# ---------------------- 2. 创建多子图布局 ----------------------
# 调整各子图的高度比例
# height_ratios = [1, 1, 1, 1, 1, 1]  # 可以根据需要调整这些值
height_ratios = [1, 1, 1, 1, 1]  # 可以根据需要调整这些值
fig, axes = plt.subplots(5, 1, figsize=(14, 9), sharex=True, gridspec_kw={"left":0.06,"right":0.98,"top":0.98,"bottom":0.06,"wspace": 0.2, "hspace": 0.1,'height_ratios': height_ratios} )  # 水平和垂直间距)

# fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
# 设置整体标题
# fig.suptitle('Algorithm Performance Comparison', fontsize=16,fontweight = 'bold', y=0.95)

reward_threshold = 320  # Reward的目标阈值
error_threshold = 0.003  # Target error的可接受阈值
agent_out_threshold = 10  # Agent out的上限阈值
landmarkC_threshold = 10  # Target Collisions的上限阈值
obstacleC_threshold = 10  # Obstacle Collisions的上限阈值

color = ["#168581","#ffbe5b","#0050b6","#5888b8","#ab0b37","#00372f"]
# ---------------------- 3. 绘制Reward子图（修改部分） ----------------------
ax_reward = axes[0]

ax_reward.axhline(
    y=reward_threshold,
    color=color[4],  # 阈值线颜色
    linestyle='-.',  # 线型（虚线/点划线等）
    linewidth=1.5,
    alpha = 0.8
)

# 绘制区间（最大值到最小值），使用填充区域
alpha_value = 0.2  # 填充区域的透明度
for index_method in range(len(method)):
    ax_reward.fill_between(training_episodes, reward_min[index_method], reward_max[index_method], color=color[index_method], alpha=alpha_value)

# 绘制平均曲线，使用更高的线宽
linewidth_value = 2.0  # 线宽
for index_method in range(len(method)):
    ax_reward.plot(training_episodes, reward_mean[index_method], label= true_method[index_method] + r'SAC', color=color[index_method], linewidth=linewidth_value)

ax_reward.set_ylabel('Reward', fontsize=10)
ax_reward.legend(loc='upper left',fontsize = 12)
# 设置x轴起始位置为0，使图形从y轴开始


ax_reward.set_ylim(bottom=-50, top=350)
ax_reward.set_xlim(left=0,right = 1.0)
# # 添加竖线及标注
for pos, label in zip(vline_positions, vline_labels):
    ax_reward.axvline(x=pos, color=color[3], linestyle='--')
    ax_reward.text(pos - 0.02, ax_reward.get_ylim()[1] * 0.02, label, fontsize=12,color=color[5])
# ---------------------- 4. 绘制State Coverage子图 ----------------------
# ax_state_coverage = axes[1]
# for index_method in range(len(method)):
#     ax_state_coverage.plot(training_episodes, StateCoverage_mean[index_method], label= method[index_method] + r'SAC', color=color[index_method])
# ax_state_coverage.set_ylabel('State Coverage')
# # 添加竖线（因sharex，位置和标注逻辑同Reward子图，可复用循环）
# # for pos, label in zip(vline_positions, vline_labels):
# #     ax_state_coverage.axvline(x=pos, color='red', linestyle='--')
# # 设置x轴起始位置为0，使图形从y轴开始
# # ax_state_coverage.set_ylim(bottom=0, top=0.03)
# ax_state_coverage.set_xlim(left=0)
# ---------------------- 4. 绘制Target error子图 ----------------------
ax_target_error = axes[1]

ax_target_error.axhline(
    y=error_threshold,
    color=color[4],
    linestyle='-.',
    linewidth=1.5,
    alpha = 0.8
)


for index_method in range(len(method)):
    ax_target_error.plot(training_episodes, error_mean[index_method], label= method[index_method] + r'SAC', color=color[index_method])
ax_target_error.set_ylabel('Target error', fontsize=10)
# 添加竖线（因sharex，位置和标注逻辑同Reward子图，可复用循环）
for pos, label in zip(vline_positions, vline_labels):
    ax_target_error.axvline(x=pos, color=color[3], linestyle='--')
# 设置x轴起始位置为0，使图形从y轴开始
ax_target_error.set_ylim(bottom=0, top=0.03)
ax_target_error.set_xlim(left=0)
# ---------------------- 5. 绘制Agent out子图 ----------------------
ax_agent_out = axes[2]

ax_agent_out.axhline(
    y=agent_out_threshold,
    color=color[4],
    linestyle='-.',
    linewidth=1.5,
    alpha = 0.8
)


for index_method in range(len(method)):
    ax_agent_out.plot(training_episodes, AgentOut_mean[index_method], label= method[index_method] + r'SAC', color=color[index_method])
ax_agent_out.set_ylabel('Agent out', fontsize=10)
# 添加竖线
for pos, _ in zip(vline_positions, vline_labels):
    ax_agent_out.axvline(x=pos, color=color[3], linestyle='--')
# 设置x轴起始位置为0，使图形从y轴开始
ax_agent_out.set_ylim(bottom=0, top=320)
ax_agent_out.set_xlim(left=0)
# ---------------------- 6. 绘制LandmarkCollision子图 ----------------------


T_color = ["#0050b6","#ffbe5b","#168581","#EC6E66","#5888b8","#00372f"]

ax_LandmarkCollision = axes[3]

ax_LandmarkCollision.axhline(
    y=landmarkC_threshold,
    color=color[4],
    linestyle='-.',
    linewidth=1.5,
    alpha = 0.8
)

for index_method in range(len(method)):
    ax_LandmarkCollision.plot(training_episodes, LandmarkC_mean[index_method], label= method[index_method] + r'SAC', color=T_color[index_method])
ax_LandmarkCollision.set_ylabel('Target Collisions', fontsize=10)
# 添加竖线
for pos, _ in zip(vline_positions, vline_labels):
    ax_LandmarkCollision.axvline(x=pos, color=color[3], linestyle='--')
# 设置x轴起始位置为0，使图形从y轴开始
ax_LandmarkCollision.set_ylim(bottom=0, top=120)
ax_LandmarkCollision.set_xlim(left=0)
# ---------------------- 6. 绘制ObstacleCollision子图 ----------------------
ax_ObstacleCollision = axes[4]

ax_ObstacleCollision.axhline(
    y=obstacleC_threshold,
    color=color[4],
    linestyle='-.',
    linewidth=1.5,
    alpha = 0.8
)

for index_method in range(len(method)):
    ax_ObstacleCollision.plot(training_episodes, ObstacleC_mean[index_method], label= method[index_method] + r'SAC', color=color[index_method])
ax_ObstacleCollision.set_ylabel('Obstacle Collisions', fontsize=10)
ax_ObstacleCollision.set_xlabel('Training episodes (M)')
# 添加竖线
for pos, _ in zip(vline_positions, vline_labels):
    ax_ObstacleCollision.axvline(x=pos, color=color[3], linestyle='--')
# 设置x轴起始位置为0，使图形从y轴开始
ax_ObstacleCollision.set_ylim(bottom=0, top=320)
ax_ObstacleCollision.set_xlim(left=0)
# 调整子图间距
# plt.tight_layout()
# 显示图形
plt.show()
