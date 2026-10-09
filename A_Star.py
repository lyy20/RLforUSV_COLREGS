import heapq
import math

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

circular_dis = 0.2
# 定义节点类
class Node:
    def __init__(self, x, y, g=float('inf'), h=float('inf'), parent=None, angle=None):
        self.x = x
        self.y = y
        self.g = g  # 从起点到当前节点的实际代价
        self.h = h  # 从当前节点到目标节点的估计代价
        self.f = g + h  # 总代价
        self.parent = parent
        self.angle = angle  # 当前节点的方向

    def __lt__(self, other):
        return self.f < other.f

# 计算启发式函数（欧几里得距离）
def heuristic(node, goal):
    return math.sqrt((node.x - goal[0]) ** 2 + (node.y - goal[1]) ** 2)

# 判断点是否在圆形障碍物内
def is_in_obstacle(point, obstacles, safety_distance):
    x, y, agent_radius = point
    for obstacle in obstacles:
        obs_x, obs_y, radius = obstacle
        effective_radius = radius + safety_distance + agent_radius
        distance = math.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
        if distance <= effective_radius:
            return True
    return False

# 计算当前需要朝向的角度
def calculate_direction_angle(current, next_point):
    dx = next_point[0] - current[0]
    dy = next_point[1] - current[1]
    angle = math.atan2(dy, dx)
    return angle

# 检查偏转角度是否超过60°
def is_angle_valid(current_angle, next_angle):
    angle_diff = abs(next_angle - current_angle)
    if angle_diff > math.pi:
        angle_diff = 2 * math.pi - angle_diff
    return angle_diff <= math.pi / 3

# 可视化地图和路线
def visualize_map(obstacles, agent, target, path, safety_distance = 0.003):
    plt.figure(figsize=(8, 8))

    # 绘制圆形障碍物及安全区域
    for obstacle in obstacles:
        x, y, radius = obstacle
        circle = Circle((x, y), radius, color='gray')
        plt.gca().add_patch(circle)
        safe_circle = Circle((x, y), radius + safety_distance, color='gray', alpha=0.2)
        plt.gca().add_patch(safe_circle)

    # 绘制智能体
    plt.scatter(agent[0], agent[1], color='blue', s=100, label='Agent')

    # 绘制目标
    plt.scatter(target[0], target[1], color='green', s=100, label='Target')

    # 绘制路径
    if path:
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        plt.plot(path_x, path_y, color='red', linestyle='-', linewidth=2, label='Path')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path Planning Visualization')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
# 优化后的检查是否在开放列表中的代码
def is_node_in_open_list(open_list, new_node, step_size):
    for node in open_list:
        if math.sqrt((node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2) < step_size:
            return True
    return False
# A*算法，连续空间搜索
def a_star_continuous(start, goal, obstacles, start_angle, step_size=0.1, safety_distance=0.05, close_threshold=0.05, max_iterations=10000):
    open_list = []
    closed_set = set()
    node_dict = {}  # 用于存储每个位置的最小 g 值节点

    start_node = Node(start[0], start[1], g=0, h=heuristic(Node(start[0], start[1]), goal), parent=None, angle=start_angle)
    heapq.heappush(open_list, start_node)
    node_dict[(start_node.x, start_node.y)] = start_node

    iteration = 0
    while open_list and iteration < max_iterations:
        current_node = heapq.heappop(open_list)

        # 当与目标点的距离小于阈值时，停止寻路
        if heuristic(current_node, goal) < close_threshold:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y, start[2]))
                current_node = current_node.parent
            return path[::-1]

        closed_set.add((current_node.x, current_node.y))

        # 考虑更多方向
        num_directions = 36
        directions = []
        goal_angle = calculate_direction_angle((current_node.x, current_node.y), goal)
        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            new_x = current_node.x + step_size * math.cos(angle)
            new_y = current_node.y + step_size * math.sin(angle)

            # 避免重复访问相近节点
            if (new_x, new_y) in closed_set:
                continue

            new_angle = calculate_direction_angle((current_node.x, current_node.y), (new_x, new_y))
            if current_node.angle is not None and not is_angle_valid(current_node.angle, new_angle):
                continue

            if not is_in_obstacle((new_x, new_y, start[2]), obstacles, safety_distance):
                angle_diff_to_goal = abs(new_angle - goal_angle)
                if angle_diff_to_goal > math.pi:
                    angle_diff_to_goal = 2 * math.pi - angle_diff_to_goal
                directions.append((angle_diff_to_goal, angle, new_x, new_y, new_angle))

        # 按与目标方向的夹角排序
        directions.sort(key=lambda x: x[0])

        for _, angle, new_x, new_y, new_angle in directions:
            new_g = current_node.g + step_size
            new_node = Node(new_x, new_y, g=new_g, h=heuristic(Node(new_x, new_y), goal), parent=current_node, angle=new_angle)

            # 检查新节点的 g 值是否比已存在的相同位置节点的 g 值更小
            if (new_x, new_y) in node_dict:
                existing_node = node_dict[(new_x, new_y)]
                if new_g >= existing_node.g:
                    continue
                else:
                    # 更新节点信息
                    node_dict[(new_x, new_y)] = new_node
            else:
                node_dict[(new_x, new_y)] = new_node

            if not is_node_in_open_list(open_list, new_node, step_size):
                heapq.heappush(open_list, new_node)

        iteration += 1

    return None


# 绕目标进行圆周运动，直接返回角度
def circular_motion(current, goal, obstacles, step_size=0.1, safety_distance=0.05, target_distance=0.2):
    # 计算智能体与目标的距离
    dx = current[0] - goal[0]
    dy = current[1] - goal[1]
    distance = math.sqrt(dx ** 2 + dy ** 2)

    # 计算智能体到目标的角度
    angle_to_goal = math.atan2(dy, dx)

    # 计算逆时针切线的方向
    tangent_angle = angle_to_goal + math.pi / 2
    # 根据当前距离与目标距离的差值微调角度
    distance_error = distance - target_distance
    # 这里使用一个简单的比例系数来调整角度，可根据实际情况调整
    angle_adjustment = 3 * distance_error  #这个系数跟步长dt有相关性，暂定为正比1：10
    # print(distance_error,angle_adjustment,tangent_angle,angle_adjustment + tangent_angle)
    tangent_angle += angle_adjustment
    # 确保角度在 [0, 2*pi) 范围内
    tangent_angle = tangent_angle % (2 * math.pi)

    # 根据切线方向生成一个试探点
    new_x = current[0] + step_size * math.cos(tangent_angle)
    new_y = current[1] + step_size * math.sin(tangent_angle)

    # 检查试探点是否在障碍物内
    if not is_in_obstacle((new_x, new_y, current[2]), obstacles, safety_distance):
        return tangent_angle
    return None

# 限制角度在 -1 到 1 之间
def limit_angle(angle):
    return max(-math.pi / 3, min(math.pi / 3, angle))

# 主函数
def A_Star(num_agents, num_obstacles, num_landmarks, agents, obstacles, landmarks, angle, agent_index, landmark_index):
    # Info结构（  智能体数量，障碍物数量，目标数量，智能体基础信息，障碍物基础信息，目标基础信息（位置，大小），智能体朝向角集合 ）
    # 如果存在多个智能体情况时，除了当前的智能体外，其余智能体均作为障碍物，除了选定的目标外，其余目标也作为障碍物，
    agent = agents[agent_index]
    target = landmarks[landmark_index]
    obstacles_all = []
    for i in range(num_agents):
        if i != agent_index:
            obstacles_all.append(agents[i])
    for i in range(num_obstacles):
        obstacles_all.append(obstacles[i])
    for i in range(num_landmarks):
        if i != landmark_index:
            obstacles_all.append(landmarks[i])

    start_angle = angle[agent_index]
    # print("\n智能体当前方向",start_angle)
    # 运行A*算法导航至目标附近
    path = a_star_continuous(agent, target, obstacles_all, start_angle)
    # 可视化地图和路线
    # print(path)
    # visualize_map(obstacles, agent, target, path)

    global circular_dis

    if path:
        current_position = path[0]
        if heuristic(Node(*current_position[:2]), target) < circular_dis:
            circular_dis = 0.3
            # 开始绕圈运动
            tangent_angle = circular_motion(current_position, target, obstacles_all)
            if tangent_angle is not None:
                angle_diff = tangent_angle - angle[agent_index]
                if angle_diff < -5.:
                    angle_diff += 2 * math.pi
                # print(f"\n绕圈智能体当前需要转动的角度: {limit_angle(angle_diff)} 弧度",tangent_angle , angle[agent_index] ,tangent_angle - angle[agent_index])
                return limit_angle(angle_diff)
        else:
            circular_dis = 0.2
            if len(path) > 1:
                current_position = path[0]
                next_position = path[1]
                # print("\n",current_position,next_position)
                angleT0 = calculate_direction_angle(current_position, next_position)
                # print(f"\n智能体当前需要朝向的角度: {angleT0} 弧度")
                # print(f"\n智能体当前需要转动的角度: {angleT0 - angle[agent_index]} 弧度")
                return angleT0 - angle[agent_index]

    # 未找到路径或已到达目标
    return 0.