import numpy as np
from typing import Dict, Tuple, List, Optional, Set
import sys

try:
    from bitarray import bitarray as _bitarray
except ImportError:
    _bitarray = None


class SparseVisitedStates:
    """Sparse fallback for very large binary state-coverage maps."""

    def __init__(self, total_bins: int):
        self.total_bins = int(total_bins)
        self._visited = set()

    def setall(self, value: int) -> None:
        if value:
            raise ValueError("SparseVisitedStates only supports setall(0).")
        self._visited.clear()

    def __setitem__(self, index: int, value: int) -> None:
        index = int(index)
        if index < 0 or index >= self.total_bins:
            raise IndexError("state coverage index out of range")
        if value:
            self._visited.add(index)
        else:
            self._visited.discard(index)

    def __getitem__(self, index: int) -> int:
        return int(int(index) in self._visited)

    def __len__(self) -> int:
        return self.total_bins

    def count(self, value: int = 1) -> int:
        visited_count = len(self._visited)
        return visited_count if value else self.total_bins - visited_count


def _make_visited_states(total_bins: int):
    # Current bin settings can exceed billions of states; sparse storage keeps
    # smoke tests and single-USV experiments from allocating hundreds of MB.
    max_dense_bits = 50_000_000
    if _bitarray is not None and total_bins <= max_dense_bits:
        visited_states = _bitarray(total_bins)
        visited_states.setall(0)
        return visited_states
    return SparseVisitedStates(total_bins)


class StateCoverageTracker:
    def __init__(self, bin_params: Dict, lsh_epsilon: float = 0.1, weight_dict: Optional[Dict] = None,
                 valid_ranges: Optional[Dict] = None, prune_invalid: bool = True, num_obstacles: int = 1):
        """
        初始化状态覆盖率追踪器

        参数:
            bin_params: 各维度的分箱参数，格式为{维度名称: (最小值, 最大值, 箱数)}
            lsh_epsilon: 局部敏感哈希的容差值
            weight_dict: 各维度的权重，格式为{维度名称: 权重值}
            valid_ranges: 各维度的有效取值范围，格式为{维度名称: (最小值, 最大值)}
            prune_invalid: 是否剪枝无效状态
            num_obstacles: 要考虑的最近障碍物数量
        """
        self.bins = bin_params
        self.epsilon = lsh_epsilon
        self.weights = weight_dict or {dim: 1.0 for dim in bin_params}
        self.dim_order = list(bin_params.keys())
        self.num_obstacles = num_obstacles

        # 预计算分箱边界
        self.bin_edges = {
            dim: np.linspace(min_val, max_val, num_bins + 1)
            for dim, (min_val, max_val, num_bins) in bin_params.items()
        }

        # 有效取值范围（用于状态空间剪枝）
        self.valid_ranges = valid_ranges or {}
        self.prune_invalid = prune_invalid

        # 预计算总箱数
        self.total_bins = self._compute_total_bins()

        # 使用 bitarray 存储访问状态，初始化为全 0
        self.visited_states = _make_visited_states(self.total_bins)
        # 获取 bitarray 对象的内存占用（字节）
        # memory_usage = sys.getsizeof(self.visited_states)
        # print(f"bitarray 占用的内存: {memory_usage} 字节")

        # 初始化随机投影矩阵
        self.projection_matrix = self._init_projection_matrix()

    def _bin(self, value: float, dim: str) -> int:
        """将连续值映射到对应的箱索引"""
        min_val, max_val, _ = self.bins[dim]

        # 检查是否超出分箱范围
        if value < min_val:
            # print(f"警告: {dim}值{value}低于最小值{min_val}，将被截断")
            value = min_val
        if value > max_val:
            # print(f"警告: {dim}值{value}高于最大值{max_val}，将被截断")
            value = max_val

        # 计算箱索引
        if value < self.bin_edges[dim][0]:
            return 0
        if value >= self.bin_edges[dim][-1]:
            return self.bins[dim][2] - 1
        return np.digitize(value, self.bin_edges[dim]) - 1

    def _polar_transform(self, cartesian_coords: Tuple[float, float]) -> Tuple[float, float]:
        """将笛卡尔坐标转换为极坐标"""
        x, y = cartesian_coords
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x) % (2 * np.pi)  # 归一化到[0, 2π)
        return (r, theta)

    def _init_projection_matrix(self):
        """初始化随机投影矩阵"""
        num_dimensions = len(self.dim_order)
        num_projections = 1000  # 可以根据需要调整投影数量
        return np.random.randn(num_projections, num_dimensions)

    def _tolerant_hash(self, discrete_state: Tuple, epsilon: float) -> int:
        """生成带容差的哈希键，使用随机投影LSH算法，并将其映射到一个整数索引"""
        state_vector = np.array(discrete_state)
        projected_vector = np.dot(self.projection_matrix, state_vector)
        hash_vector = np.sign(projected_vector)
        # 将哈希向量转换为一个整数索引
        index = hash(tuple(hash_vector)) % self.total_bins
        return index

    def _is_valid_state(self, state: Dict) -> bool:
        """检查状态是否在有效范围内"""
        for dim, (min_val, max_val) in self.valid_ranges.items():
            if dim in state:
                if isinstance(state[dim], tuple) or isinstance(state[dim], list):
                    # 处理坐标等多维值
                    for v in state[dim]:
                        if v < min_val or v > max_val:
                            return False
                else:
                    # 处理标量值
                    if state[dim] < min_val or state[dim] > max_val:
                        return False
        return True

    def discretize_state(self, state: Dict) -> Tuple:
        """将连续状态转换为离散表示"""
        # 提取各维度的值
        q_t = state['q_t']  # 目标相对位置

        # 处理障碍物信息
        obstacles = state.get('obstacles', [])
        # 提取N个最近障碍物的距离和方位角
        sorted_obstacles = sorted(obstacles, key=lambda obs: obs['distance'])
        obstacle_dists = [obs['distance'] for obs in sorted_obstacles[:self.num_obstacles]]
        obstacle_angles = [obs['angle'] for obs in sorted_obstacles[:self.num_obstacles]]
        obstacle_sizes = [obs['size'] for obs in sorted_obstacles[:self.num_obstacles]]
        # 补齐不足N个障碍物的情况
        while len(obstacle_dists) < self.num_obstacles:
            obstacle_dists.append(0.0)
            obstacle_angles.append(0.0)
            obstacle_sizes.append(0.0)

        # 离散化各维度

        q_polar = self._polar_transform(q_t)
        q_r_bin = self._bin(q_polar[0], 'q_r')
        q_theta_bin = self._bin(q_polar[1], 'q_theta')

        # 障碍物特征离散化
        obstacle_dist_bins = [self._bin(dist, f'obstacle_dist_{i}') for i, dist in enumerate(obstacle_dists)]
        obstacle_angle_bins = [self._bin(angle, f'obstacle_angle_{i}') for i, angle in enumerate(obstacle_angles)]
        obstacle_size_bins = [self._bin(size, f'obstacle_size_{i}') for i, size in enumerate(obstacle_sizes)]

        return (q_r_bin, q_theta_bin,
                *obstacle_dist_bins, *obstacle_angle_bins, *obstacle_size_bins)

    def update_coverage(self, state: Dict) -> None:
        """更新状态覆盖率"""
        # 检查状态有效性
        if self.prune_invalid and not self._is_valid_state(state):
            return

        discrete_state = self.discretize_state(state)
        hash_index = self._tolerant_hash(discrete_state, self.epsilon)
        self.visited_states[hash_index] = 1  # 将访问状态对应的位设置为 1

    def calculate_coverage(self) -> Dict:
        """计算覆盖率"""
        covered_bins = self.visited_states.count(1)  # 统计访问状态中值为 1 的位的数量

        # 计算实际可到达的状态空间大小
        if self.prune_invalid:
            # 使用预定义的有效范围调整总状态数
            adjusted_total_bins = self._compute_adjusted_total_bins()
        else:
            adjusted_total_bins = self.total_bins

        raw_coverage = covered_bins / adjusted_total_bins if adjusted_total_bins > 0 else 0

        # 计算各维度的覆盖率（简化实现）
        dimension_coverages = {dim: 0.0 for dim in self.dim_order}
        # 实际应用中需要更复杂的逻辑来计算各维度的覆盖率

        # 计算加权覆盖率
        weighted_coverage = sum(
            self.weights[dim] * dimension_coverages.get(dim, 0.0)
            for dim in self.weights
        ) / sum(self.weights.values())

        return {
            'raw_coverage': raw_coverage,
            'weighted_coverage': weighted_coverage,
            'visited_states_count': covered_bins,
            'adjusted_total_possible_states': adjusted_total_bins,
            'original_total_possible_states': self.total_bins
        }

    def _compute_total_bins(self) -> int:
        """计算总状态箱数"""
        total = 1
        for dim in self.bins:
            total *= self.bins[dim][2]
        return total

    def _compute_adjusted_total_bins(self) -> int:
        """计算考虑有效范围后的调整状态箱数"""
        if not self.valid_ranges:
            return self.total_bins

        # 计算每个维度的有效箱数比例
        valid_ratio = 1.0
        for dim, (valid_min, valid_max) in self.valid_ranges.items():
            if dim in self.bins:
                bin_min, bin_max, num_bins = self.bins[dim]
                # 计算有效范围占总范围的比例
                if bin_max > bin_min:
                    ratio = min(valid_max, bin_max) - max(valid_min, bin_min)
                    ratio = ratio / (bin_max - bin_min)
                    ratio = max(0.0, min(1.0, ratio))  # 确保比例在[0,1]之间
                    valid_ratio *= ratio

        # 根据有效比例调整总箱数
        return max(1, int(self.total_bins * valid_ratio))

    def _compute_dimension_coverages(self) -> List[float]:
        """计算各维度的覆盖率（简化实现）"""
        # 实际应用中需要更复杂的逻辑来计算各维度的覆盖率
        # 这里仅返回原始覆盖率作为示例
        return [len(self.visited_states) / self.total_bins] * len(self.dim_order)


# 测试代码
def run_tests():
    # 定义分箱参数
    num_obstacles = 3
    bin_params = {
        'q_r': (0.0, 2.0, 200),  # 目标距离：0-100m，分为10箱
        'q_theta': (0.0, 2 * np.pi, 24),  # 目标方位角：0-2π，分为8箱
    }
    for i in range(num_obstacles):
        bin_params[f'obstacle_dist_{i}'] = (0.0, 0.5, 10)  # 障碍物距离：0-50m，分为5箱
        bin_params[f'obstacle_angle_{i}'] = (0.0, 2 * np.pi, 8)  # 障碍物方位角：0-2π，分为4箱
        bin_params[f'obstacle_size_{i}'] = (0.0, 0.15, 10)  # 障碍物大小：0.05-0.15，分为4箱

    # 定义权重
    weight_dict = {
        'q_r': 0.3,
        'q_theta': 0.3,
    }
    for i in range(num_obstacles):
        weight_dict[f'obstacle_dist_{i}'] = 0.1
        weight_dict[f'obstacle_angle_{i}'] = 0.1
        weight_dict[f'obstacle_size_{i}'] = 0.1

    # 定义有效取值范围（关键改进）
    valid_ranges = {
        'q_r': (0.0, 80.0),  # 目标最大可见距离为80m
    }
    for i in range(num_obstacles):
        valid_ranges[f'obstacle_dist_{i}'] = (0.0, 40.0)  # 障碍物检测范围为40m

    # 创建覆盖率追踪器（启用状态空间剪枝）
    tracker = StateCoverageTracker(bin_params, lsh_epsilon=0.01, weight_dict=weight_dict,
                                   valid_ranges=valid_ranges, prune_invalid=True, num_obstacles=num_obstacles)

    # 增加更多示例状态
    test_states = [
        # 正常状态
        {
            'q_t': (30.0, 40.0),
            'obstacles': [
                {'distance': 10.0, 'angle': np.pi / 4, 'size': 0.10},
                {'distance': 20.0, 'angle': np.pi / 2, 'size': 0.10}
            ]
        },
        # 高速状态（接近有效范围上限）
        {
            'q_t': (60.0, 60.0),
            'obstacles': [
                {'distance': 30.0, 'angle': np.pi / 3, 'size': 0.10},
                {'distance': 35.0, 'angle': np.pi / 6, 'size': 0.10}
            ]
        },
        # 超出有效范围的状态（会被剪枝）
        {
            'q_t': (90.0, 0.0),  # 超出有效范围(80.0)
            'obstacles': [
                {'distance': 45.0, 'angle': 0.0, 'size': 0.10},  # 超出有效范围(40.0)
                {'distance': 42.0, 'angle': np.pi / 2, 'size': 0.10}  # 超出有效范围(40.0)
            ]
        },
        # 另一个超出有效范围的状态
        {
            'q_t': (100.0, 0.0),  # 超出有效范围(80.0)
            'obstacles': []
        }
    ]

    # 更新覆盖率
    for i, state in enumerate(test_states):
        print(f"\n添加测试状态 {i + 1}:")
        tracker.update_coverage(state)

    # 计算并打印覆盖率
    coverage = tracker.calculate_coverage()
    print("\n状态覆盖率统计:")
    print(f"已访问状态数: {coverage['visited_states_count']}")
    print(f"原始总可能状态数: {coverage['original_total_possible_states']}")
    print(f"调整后总可能状态数: {coverage['adjusted_total_possible_states']}")
    print(f"原始覆盖率: {coverage['raw_coverage']:.2%}")
    print(f"加权覆盖率: {coverage['weighted_coverage']:.2%}")

    # 测试：检查相似状态是否被识别为同一状态
    similar_state = {
        'q_t': (30.2, 40.2),
        'obstacles': [
            {'distance': 10.3, 'angle': np.pi / 4 + 0.02, 'size': 0.10},
            {'distance': 20.2, 'angle': np.pi / 2 + 0.02, 'size': 0.10}
        ]
    }

    initial_count = coverage['visited_states_count']
    tracker.update_coverage(similar_state)
    new_coverage = tracker.calculate_coverage()

    if new_coverage['visited_states_count'] == initial_count:
        print("\n测试通过：相似状态被正确识别为同一状态")
    else:
        print("\n测试失败：相似状态被错误识别为新状态")


if __name__ == "__main__":
    run_tests()
