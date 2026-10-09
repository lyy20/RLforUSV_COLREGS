import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import math
from matplotlib import transforms  # 新增导入

# 模拟数据：每个USV的位置(x,y)、运动方向角度（弧度制，0为x轴正方向，逆时针为正）
usv_data = [
    {"x": 10, "y": 70, "angle": np.radians(30)},
    {"x": 60, "y": 90, "angle": np.radians(0)},
    {"x": 85, "y": 50, "angle": np.radians(90)},
    {"x": 20, "y": 30, "angle": np.radians(180)},
    {"x": 60, "y": 10, "angle": np.radians(270)},
]

# 定义绘制船只形状的函数（修改第26行）
def draw_ship(ax, x, y, angle, ship_length=8, ship_width=3):
    # 基础变换：平移+旋转
    base_transform = (
        transforms.Affine2D()
        .translate(x, y)
        .rotate(angle)
        + ax.transData
    )
    
    # 1. 船体（主体）
    hull_points = np.array([
        [-ship_length/2, -ship_width/2.5],  # 船尾左侧
        [ship_length/2 * 0.5, -ship_width/2],  # 船头左侧
        [ship_length/2, 0],  # 船头尖端
        [ship_length/2 * 0.5, ship_width/2],  # 船头右侧
        [-ship_length/2, ship_width/2.5],  # 船尾右侧
        [-ship_length/2, -ship_width/2.5]  # 闭合
    ])
    hull = Polygon(hull_points, closed=True, facecolor='#8B4513', edgecolor='black', linewidth=1.5, transform=base_transform)
    ax.add_patch(hull)
    
    # 2. 甲板
    deck_points = np.array([
        [-ship_length/2 * 0.8, -ship_width/4],
        [ship_length/2 * 0.4, -ship_width/3],
        [ship_length/2 * 0.4, ship_width/3],
        [-ship_length/2 * 0.8, ship_width/4],
        [-ship_length/2 * 0.8, -ship_width/4]
    ])
    deck = Polygon(deck_points, closed=True, facecolor='#A0522D', edgecolor='black', linewidth=1, transform=base_transform)
    ax.add_patch(deck)
    
    # 3. 驾驶舱
    bridge_points = np.array([
        [-ship_length/4, -ship_width/8],
        [ship_length/8, -ship_width/8],
        [ship_length/8, ship_width/8],
        [-ship_length/4, ship_width/8],
        [-ship_length/4, -ship_width/8]
    ])
    bridge = Polygon(bridge_points, closed=True, facecolor='#4682B4', edgecolor='black', transform=base_transform)
    ax.add_patch(bridge)
    
    # 6. 船头标识（可选）
    bow_marker = plt.Circle(
        (ship_length/2 * 0.7, 0), ship_width/12, 
        facecolor='yellow', edgecolor='black', transform=base_transform
    )
    ax.add_patch(bow_marker)
    
    # 7. 尾迹效果（模拟船只移动）
    if angle != 0:  # 避免角度为0时出现NaN
        wake_angle = angle + np.pi  # 尾迹方向与船头相反
        wake_length = ship_length * 0.8
        wake_width = ship_width * 0.6
        
        # 左尾迹
        left_wake = Polygon(
            np.array([
                [x, y],
                [x + np.cos(wake_angle - 0.1) * wake_length, y + np.sin(wake_angle - 0.1) * wake_length],
                [x + np.cos(wake_angle - 0.2) * wake_length * 0.7, y + np.sin(wake_angle - 0.2) * wake_length * 0.7],
                [x, y]
            ]),
            closed=True, facecolor='lightblue', alpha=0.3, edgecolor=None
        )
        ax.add_patch(left_wake)
        
        # 右尾迹
        right_wake = Polygon(
            np.array([
                [x, y],
                [x + np.cos(wake_angle + 0.1) * wake_length, y + np.sin(wake_angle + 0.1) * wake_length],
                [x + np.cos(wake_angle + 0.2) * wake_length * 0.7, y + np.sin(wake_angle + 0.2) * wake_length * 0.7],
                [x, y]
            ]),
            closed=True, facecolor='lightblue', alpha=0.3, edgecolor=None
        )
        ax.add_patch(right_wake)
# 初始化绘图
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')

# 绘制障碍物、目标点等（模拟原有逻辑，可替换为真实数据）
obstacles = [
    (50, 70, 10), (30, 50, 5), (40, 30, 8), (70, 40, 12), (50, 50, 3)
]
for x, y, r in obstacles:
    circle = plt.Circle((x, y), r, color='gray')
    ax.add_artist(circle)

targets = [
    (40, 90, 'purple'), (80, 70, 'red'), (10, 50, 'blue'), 
    (40, 10, 'orange'), (80, 30, 'green')
]
for x, y, color in targets:
    ax.scatter(x, y, color=color, marker='*', s=100)

# 绘制所有USV（调用自定义函数）
for usv in usv_data:
    draw_ship(ax, usv["x"], usv["y"], usv["angle"])

plt.grid(True)
plt.show()