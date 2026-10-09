import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 模拟数据（替换为真实数据） ----------------------
# 训练步数，假设最大2.0M，步长0.01M
training_episodes = np.arange(0, 2.01, 0.01)  
num_steps = len(training_episodes)

# 模拟三种算法（SAC、LSTM - SAC、H - LSTM - SAC）的Reward曲线
np.random.seed(0)
# 创建十次实验的数据
num_experiments = 10
reward_sac_all = np.zeros((num_experiments, num_steps))
reward_lstm_sac_all = np.zeros((num_experiments, num_steps))
reward_h_lstm_sac_all = np.zeros((num_experiments, num_steps))

# 为每种算法生成十次实验的数据
for i in range(num_experiments):
    # 添加一些随机性以模拟不同实验结果
    noise = np.random.normal(0, 5, num_steps)
    reward_sac_all[i] = 50 + 50 * np.random.rand(num_steps).cumsum() / num_steps * 2 + noise
    
    noise = np.random.normal(0, 5, num_steps)
    reward_lstm_sac_all[i] = 30 + 70 * np.random.rand(num_steps).cumsum() / num_steps * 2 + noise
    
    noise = np.random.normal(0, 5, num_steps)
    reward_h_lstm_sac_all[i] = 10 + 90 * np.random.rand(num_steps).cumsum() / num_steps * 2 + noise

# 计算每种算法的平均曲线
reward_sac_mean = np.mean(reward_sac_all, axis=0)
reward_lstm_sac_mean = np.mean(reward_lstm_sac_all, axis=0)
reward_h_lstm_sac_mean = np.mean(reward_h_lstm_sac_all, axis=0)

# 计算每种算法的最大值和最小值
reward_sac_max = np.max(reward_sac_all, axis=0)
reward_sac_min = np.min(reward_sac_all, axis=0)
reward_lstm_sac_max = np.max(reward_lstm_sac_all, axis=0)
reward_lstm_sac_min = np.min(reward_lstm_sac_all, axis=0)
reward_h_lstm_sac_max = np.max(reward_h_lstm_sac_all, axis=0)
reward_h_lstm_sac_min = np.min(reward_h_lstm_sac_all, axis=0)

# 模拟Target error曲线
target_error_sac = 0.4 - 0.2 * np.random.rand(num_steps).cumsum() / num_steps
target_error_lstm_sac = 0.35 - 0.15 * np.random.rand(num_steps).cumsum() / num_steps
target_error_h_lstm_sac = 0.3 - 0.1 * np.random.rand(num_steps).cumsum() / num_steps

# 模拟Agent out曲线
agent_out_sac = 7 - 7 * np.random.rand(num_steps).cumsum() / num_steps
agent_out_lstm_sac = 6 - 6 * np.random.rand(num_steps).cumsum() / num_steps
agent_out_h_lstm_sac = 5 - 5 * np.random.rand(num_steps).cumsum() / num_steps

# 模拟Collisions曲线
collisions_sac = 2 + 8 * np.random.rand(num_steps).cumsum() / num_steps
collisions_lstm_sac = 3 + 7 * np.random.rand(num_steps).cumsum() / num_steps
collisions_h_lstm_sac = 4 + 6 * np.random.rand(num_steps).cumsum() / num_steps

# 竖线位置（示例，对应图中a、b、c、d、e位置，按需调整）
vline_positions = [0.1, 0.4, 0.6, 1.2, 1.9]  
vline_labels = ['a', 'b', 'c', 'd', 'e']

# ---------------------- 2. 创建多子图布局 ----------------------
fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
# 设置整体标题
fig.suptitle('Algorithm Performance Comparison', fontsize=16, y=0.95)

# ---------------------- 3. 绘制Reward子图（修改部分） ----------------------
ax_reward = axes[0]

# 绘制区间（最大值到最小值），使用填充区域
alpha_value = 0.3  # 填充区域的透明度
ax_reward.fill_between(training_episodes, reward_sac_min, reward_sac_max, color='green', alpha=alpha_value)
ax_reward.fill_between(training_episodes, reward_lstm_sac_min, reward_lstm_sac_max, color='orange', alpha=alpha_value)
ax_reward.fill_between(training_episodes, reward_h_lstm_sac_min, reward_h_lstm_sac_max, color='blue', alpha=alpha_value)

# 绘制平均曲线，使用更高的线宽
linewidth_value = 2.0  # 线宽
ax_reward.plot(training_episodes, reward_sac_mean, label='SAC (Mean)', color='green', linewidth=linewidth_value)
ax_reward.plot(training_episodes, reward_lstm_sac_mean, label='LSTM - SAC (Mean)', color='orange', linewidth=linewidth_value)
ax_reward.plot(training_episodes, reward_h_lstm_sac_mean, label='H - LSTM - SAC (Mean)', color='blue', linewidth=linewidth_value)

ax_reward.set_ylabel('Reward')
ax_reward.legend()
# 设置x轴起始位置为0，使图形从y轴开始
ax_reward.set_xlim(left=0)
# 添加竖线及标注
for pos, label in zip(vline_positions, vline_labels):
    ax_reward.axvline(x=pos, color='red', linestyle='--')
    ax_reward.text(pos + 0.01, ax_reward.get_ylim()[1] * 0.9, label, fontsize=12, backgroundcolor='white')

# ---------------------- 4. 绘制Target error子图 ----------------------
ax_target_error = axes[1]
ax_target_error.plot(training_episodes, target_error_sac, label='SAC', color='green')
ax_target_error.plot(training_episodes, target_error_lstm_sac, label='LSTM - SAC', color='orange')
ax_target_error.plot(training_episodes, target_error_h_lstm_sac, label='H - LSTM - SAC', color='blue')
ax_target_error.set_ylabel('Target error')
# 添加竖线（因sharex，位置和标注逻辑同Reward子图，可复用循环）
for pos, label in zip(vline_positions, vline_labels):
    ax_target_error.axvline(x=pos, color='red', linestyle='--')
# 设置x轴起始位置为0，使图形从y轴开始
ax_target_error.set_xlim(left=0)
# ---------------------- 5. 绘制Agent out子图 ----------------------
ax_agent_out = axes[2]
ax_agent_out.plot(training_episodes, agent_out_sac, label='SAC', color='green')
ax_agent_out.plot(training_episodes, agent_out_lstm_sac, label='LSTM - SAC', color='orange')
ax_agent_out.plot(training_episodes, agent_out_h_lstm_sac, label='H - LSTM - SAC', color='blue')
ax_agent_out.set_ylabel('Agent out (%)')
# 添加竖线
for pos, _ in zip(vline_positions, vline_labels):
    ax_agent_out.axvline(x=pos, color='red', linestyle='--')
# 设置x轴起始位置为0，使图形从y轴开始
ax_agent_out.set_xlim(left=0)
# ---------------------- 6. 绘制Collisions子图 ----------------------
ax_collisions = axes[3]
ax_collisions.plot(training_episodes, collisions_sac, label='SAC', color='green')
ax_collisions.plot(training_episodes, collisions_lstm_sac, label='LSTM - SAC', color='orange')
ax_collisions.plot(training_episodes, collisions_h_lstm_sac, label='H - LSTM - SAC', color='blue')
ax_collisions.set_ylabel('Collisions (%)')
ax_collisions.set_xlabel('Training episodes (M)')
# 添加竖线
for pos, _ in zip(vline_positions, vline_labels):
    ax_collisions.axvline(x=pos, color='red', linestyle='--')
# 设置x轴起始位置为0，使图形从y轴开始
ax_collisions.set_xlim(left=0)
# ---------------------- 6. 绘制Collisions子图 ----------------------
ax_Obstacles_collisions = axes[4]
ax_Obstacles_collisions.plot(training_episodes, collisions_sac, label='SAC', color='green')
ax_Obstacles_collisions.plot(training_episodes, collisions_lstm_sac, label='LSTM - SAC', color='orange')
ax_Obstacles_collisions.plot(training_episodes, collisions_h_lstm_sac, label='H - LSTM - SAC', color='blue')
ax_Obstacles_collisions.set_ylabel('Collisions (%)')
ax_Obstacles_collisions.set_xlabel('Training episodes (M)')
# 添加竖线
for pos, _ in zip(vline_positions, vline_labels):
    ax_Obstacles_collisions.axvline(x=pos, color='red', linestyle='--')
# 设置x轴起始位置为0，使图形从y轴开始
ax_Obstacles_collisions.set_xlim(left=0)
# 调整子图间距
plt.tight_layout()
# 显示图形
plt.show()
