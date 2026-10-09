import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# 设置更通用的字体选项，避免Arial依赖
# plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Sans", "sans-serif"]

# 定义目标运动轨迹（匀速水平直线）
def target_trajectory(t, start_pos=(0, 0), velocity=(1, 0)):
    """Generate target trajectory moving horizontally at constant speed"""
    x = start_pos[0] + velocity[0] * t
    y = start_pos[1] + velocity[1] * t
    return x, y

# 生成目标预测位置（随训练阶段变化，随机性减小）
def predicted_target(t, stage=0, start_pos=(0, 0), velocity=(1, 0)):
    """Generate predicted target positions, more accurate with higher stages"""
    x_true, y_true = target_trajectory(t, start_pos, velocity)
    
    # Prediction error decreases with training stage
    error_factor = 0.5 / (stage + 1)  # 减小随机性，误差因子降低到0.5
    
    # Add random error
    x_pred = x_true + np.random.normal(0, error_factor)
    y_pred = y_true + np.random.normal(0, error_factor)
    
    return x_pred, y_pred

# 生成智能体位置（随训练阶段变化，基于上一位置）
def agent_position(t, prev_pos, stage=0, start_pos=(0, 0), velocity=(1, 0)):
    """Generate agent positions based on previous position, closer to target with higher stages"""
    x_target, y_target = target_trajectory(t, start_pos, velocity)
    
    # 智能体向目标移动的速度随训练阶段提高
    move_factor = 0.3 + 0.5 * (stage / 5)  # 初始为0.3，最高为0.8
    
    # 计算智能体应移动的方向和距离
    dx = x_target - prev_pos[0]
    dy = y_target - prev_pos[1]
    
    # 添加一些随机性，使运动更自然
    noise_factor = 0.3 - 0.25 * (stage / 5)  # 初始为0.3，最高为0.05（减小随机性）
    dx += np.random.normal(0, noise_factor)
    dy += np.random.normal(0, noise_factor)
    
    # 计算新位置（基于上一位置）
    x_agent = prev_pos[0] + dx * move_factor
    y_agent = prev_pos[1] + dy * move_factor
    
    return x_agent, y_agent

# 生成数据
t_values = np.linspace(0, 20, 100)
start_pos = (0, 0)
velocity = (1, 0)  # 水平匀速直线运动的速度向量

# 预计算目标轨迹
target_positions = [target_trajectory(t, start_pos, velocity) for t in t_values]
target_x, target_y = zip(*target_positions)

# 创建组合图
def create_combined_plot(num_stages=5):
    """Create a single plot with multiple stages arranged horizontally"""
    fig, axes = plt.subplots(1, num_stages, figsize=(15*num_stages, 4))
    
    # 如果只有一个阶段，确保axes是可索引的
    if num_stages == 1:
        axes = [axes]
    
    # 计算每个阶段的数据
    for stage in range(num_stages):
        # 初始化智能体位置（从起点开始）
        agent_positions = [start_pos]
        
        # 计算智能体位置（基于上一位置）
        for i in range(1, len(t_values)):
            prev_pos = agent_positions[-1]
            new_pos = agent_position(t_values[i], prev_pos, stage, start_pos, velocity)
            agent_positions.append(new_pos)
        
        # 计算预测位置
        predicted_positions = [predicted_target(t, stage, start_pos, velocity) for t in t_values]
        
        pred_x, pred_y = zip(*predicted_positions)
        agent_x, agent_y = zip(*agent_positions)
        
        # 在当前子图上绘制
        ax = axes[stage]
        
        # 设置黑色边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)  # 增加线宽以提高可见性
        
        # Plot target trajectory (thick black line)
        ax.plot(target_x, target_y, 'k-', linewidth=4, label='Target Trajectory' if stage == 0 else "")
        
        # Plot predicted target points (yellow dots, larger size)
        ax.scatter(pred_x, pred_y, c='yellow', marker='o', s=100, alpha=0.7, label='Predicted Target' if stage == 0 else "")
        
        # Plot agent positions (blue dots, larger size)
        ax.scatter(agent_x, agent_y, c='blue', marker='o', s=100, alpha=0.7, label='Agent Position' if stage == 0 else "")
        
        # Add title
        ax.set_title(f'Training Stage {stage+1}')
        
        # Remove tick labels and grid but keep the frame
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set axis limits
        ax.set_xlim(min(target_x)-1, max(target_x)+1)
        ax.set_ylim(min(target_y)-2, max(target_y)+2)
    
    # 只在第一张图显示图例
    if num_stages > 0:
        axes[0].legend(loc='upper right')
    
    plt.tight_layout()
    return fig

# 创建组合图
combined_fig = create_combined_plot(num_stages=5)
plt.show()