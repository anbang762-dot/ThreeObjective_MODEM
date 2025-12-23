import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文
plt.rcParams['axes.unicode_minus'] = False

# ==================== 全局参数 ====================
START_POINT = (10, 10)  # 起点
END_POINT = (90, 90)  # 终点
BOUNDARY = (100, 100)  # 区域边界
K = 5  # 控制点数量

# 电池模型参数
BATTERY_CAPACITY = 10000  # 电池总容量（mAh）
ENERGY_PER_METER = 10  # 每米飞行能耗（mAh）
ENERGY_PER_RISK = 0.5  # 每单位风险能耗（mAh）

# ==================== 重新设置障碍物和风险区域 ====================
# 障碍物 (x_min, x_max, y_min, y_max)
OBSTACLES = [
    # 中心区域的大障碍物
    (30, 70, 40, 60),
    # 左上角障碍物
    (10, 25, 70, 85),
    # 右上角障碍物
    (75, 85, 15, 25),
    # 左下角障碍物
    (15, 30, 10, 20),
    # 右下角障碍物
    (65, 80, 70, 85),
    # 中间的横向障碍物
    (20, 50, 25, 35),
    # 中间的纵向障碍物
    (45, 55, 20, 80),
]


def create_risk_map(size=100):
    """创建风险场 - 使用多个高风险区域"""
    risk_map = np.ones((size, size))  # 背景风险值为1

    # 设置不同类型的风险区域
    # 1. 高风险区域 (风险值=10)
    risk_map[15:35, 60:80] = 10  # 区域1 - 右下角高风险
    risk_map[60:80, 15:35] = 10  # 区域2 - 左上角高风险
    risk_map[40:60, 40:60] = 10  # 区域3 - 中心高风险

    # 2. 中等风险区域 (风险值=5)
    risk_map[70:90, 70:90] = 5  # 区域4 - 右上角中等风险
    risk_map[10:30, 10:30] = 5  # 区域5 - 左下角中等风险

    # 3. 创建一些线性高风险带 (模拟雷达扫描区域)
    # 水平高风险带
    risk_map[45:55, :] = 8
    # 垂直高风险带
    risk_map[:, 45:55] = 8

    # 4. 创建一些随机高风险点
    np.random.seed(42)  # 设置随机种子以便重现
    for _ in range(20):
        x = np.random.randint(10, 90)
        y = np.random.randint(10, 90)
        size = np.random.randint(3, 8)
        risk_map[y - size:y + size, x - size:x + size] = np.random.choice([7, 8, 9])

    return risk_map


# ==================== 路径处理函数 ====================
def decode_path(control_points):
    """将控制点解码为完整路径"""
    points = []
    for i in range(0, len(control_points), 2):
        points.append((control_points[i], control_points[i + 1]))
    return [START_POINT] + points + [END_POINT]


def compute_distance(control_points):
    """计算路径长度 - f1"""
    path = decode_path(control_points)
    total_length = 0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        total_length += np.linalg.norm(p2 - p1)
    return total_length


def compute_risk(control_points, risk_map):
    """计算风险暴露 - f2"""
    path = decode_path(control_points)
    total_risk = 0

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]

        # 计算线段长度
        segment_length = np.linalg.norm(np.array(p2) - np.array(p1))

        # 根据线段长度决定采样点数
        num_samples = max(10, int(segment_length))

        for j in range(num_samples):
            t = j / max(num_samples - 1, 1)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])

            # 边界检查
            x_idx = min(max(0, int(x)), risk_map.shape[1] - 1)
            y_idx = min(max(0, int(y)), risk_map.shape[0] - 1)
            total_risk += risk_map[y_idx, x_idx] * (segment_length / num_samples)

    return total_risk


def compute_battery_consumption(control_points, risk_map):
    """计算电池损耗 - f3 (新增的第三个目标)"""
    # 基础能耗：与路径长度成正比
    distance = compute_distance(control_points)
    base_energy = distance * ENERGY_PER_METER

    # 风险相关能耗：高风险区域需要更多能量（如开启避障系统、加速通过等）
    risk = compute_risk(control_points, risk_map)
    risk_energy = risk * ENERGY_PER_RISK

    # 总能耗
    total_energy = base_energy + risk_energy

    # 返回电池损耗百分比（占电池总容量的百分比）
    battery_depletion = (total_energy / BATTERY_CAPACITY) * 100

    return battery_depletion


def is_intersect(p1, p2, obstacle):
    """检查线段是否与矩形障碍物相交"""
    x_min, x_max, y_min, y_max = obstacle

    # 快速检查：如果两个点都在矩形同一侧，则不可能相交
    if (p1[0] < x_min and p2[0] < x_min) or (p1[0] > x_max and p2[0] > x_max) or \
            (p1[1] < y_min and p2[1] < y_min) or (p1[1] > y_max and p2[1] > y_max):
        return False

    # 检查端点是否在矩形内
    if (x_min <= p1[0] <= x_max and y_min <= p1[1] <= y_max) or \
            (x_min <= p2[0] <= x_max and y_min <= p2[1] <= y_max):
        return True

    # 检查线段与矩形四条边的交点
    edges = [
        [(x_min, y_min), (x_max, y_min)],  # 下边
        [(x_min, y_max), (x_max, y_max)],  # 上边
        [(x_min, y_min), (x_min, y_max)],  # 左边
        [(x_max, y_min), (x_max, y_max)]  # 右边
    ]

    def line_intersection(line1, line2):
        """计算两条线段的交点"""
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            return True
        return None

    segment = [p1, p2]
    for edge in edges:
        if line_intersection(segment, edge) is not None:
            return True

    return False


def is_feasible(control_points):
    """检查路径是否可行"""
    path = decode_path(control_points)

    # 检查边界
    for point in path:
        if not (0 <= point[0] <= BOUNDARY[0] and 0 <= point[1] <= BOUNDARY[1]):
            return False

    # 检查障碍物
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        for obstacle in OBSTACLES:
            if is_intersect(p1, p2, obstacle):
                return False

    return True


def add_penalty(fitness, penalty=1000):
    """为不可行解添加罚函数"""
    return (fitness[0] + penalty, fitness[1] + penalty, fitness[2] + penalty)


# ==================== 三目标优化算法基础函数 ====================
def dominates(a, b):
    """判断解a是否支配解b（三个目标）"""
    return (a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]) and \
        (a[0] < b[0] or a[1] < b[1] or a[2] < b[2])


def crowding_distance(population_fitness):
    """计算拥挤距离（三个目标）"""
    n = len(population_fitness)
    if n <= 3:
        return [np.inf] * n

    distances = np.zeros(n)

    # 对每个目标进行计算
    for obj in range(3):
        # 获取该目标的所有值并排序
        obj_values = np.array([f[obj] for f in population_fitness])
        order = np.argsort(obj_values)

        # 设置边界距离为无穷大
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        # 计算范围
        f_min = obj_values[order[0]]
        f_max = obj_values[order[-1]]

        # 避免除以零
        if f_max - f_min < 1e-10:
            continue

        # 计算中间点的拥挤距离
        for i in range(1, n - 1):
            idx = order[i]
            prev_idx = order[i - 1]
            next_idx = order[i + 1]

            # 检查是否为有限数值
            if np.isfinite(obj_values[next_idx] - obj_values[prev_idx]):
                distances[idx] += (obj_values[next_idx] - obj_values[prev_idx]) / (f_max - f_min)

    return distances


def non_dominated_sort(population, fitness_values):
    """非支配排序（三个目标）"""
    n = len(population)
    fronts = []
    domination_counts = [0] * n
    dominated_solutions = [[] for _ in range(n)]

    # 第一轮：计算支配关系
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(fitness_values[i], fitness_values[j]):
                dominated_solutions[i].append(j)
            elif dominates(fitness_values[j], fitness_values[i]):
                domination_counts[i] += 1

    # 构建第一前沿（非支配解）
    first_front = [i for i in range(n) if domination_counts[i] == 0]
    if first_front:
        fronts.append(first_front)

    # 构建后续前沿
    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)

        if next_front:
            fronts.append(next_front)
        current_front += 1

    return fronts


def repair_solution(control_points):
    """修复不可行解"""
    repaired = control_points.copy()

    # 修复边界
    for i in range(len(repaired)):
        if i % 2 == 0:  # x坐标
            if repaired[i] < 0:
                repaired[i] = np.random.uniform(0, 10)
            elif repaired[i] > 100:
                repaired[i] = np.random.uniform(90, 100)
        else:  # y坐标
            if repaired[i] < 0:
                repaired[i] = np.random.uniform(0, 10)
            elif repaired[i] > 100:
                repaired[i] = np.random.uniform(90, 100)

    return repaired


# ==================== 原始MODE算法实现 ====================
class MODE_Original:
    """原始MODE算法实现"""

    def __init__(self, pop_size=100, K=5, F=0.5, CR=0.9, max_gen=200, archive_size=100):
        self.pop_size = pop_size
        self.K = K  # 控制点数量
        self.dim = K * 2  # 决策变量维度
        self.F = F  # 缩放因子
        self.CR = CR  # 交叉概率
        self.max_gen = max_gen
        self.archive_size = archive_size  # 外部档案大小
        self.risk_map = create_risk_map()

        # 初始化种群
        self.population = np.random.uniform(0, 100, (pop_size, self.dim))

        # 评估初始种群
        self.fitness = []
        for i in range(pop_size):
            self.population[i] = repair_solution(self.population[i])

            if is_feasible(self.population[i]):
                f1 = compute_distance(self.population[i])
                f2 = compute_risk(self.population[i], self.risk_map)
                f3 = compute_battery_consumption(self.population[i], self.risk_map)
                self.fitness.append((f1, f2, f3))
            else:
                # 不可行解添加大罚项
                f1 = compute_distance(self.population[i]) + 1000
                f2 = compute_risk(self.population[i], self.risk_map) + 1000
                f3 = compute_battery_consumption(self.population[i], self.risk_map) + 1000
                self.fitness.append((f1, f2, f3))

        self.fitness = np.array(self.fitness)

        # 用非支配解初始化外部档案
        self.archive = []
        self.update_archive()

    def update_archive(self):
        """更新外部档案"""
        # 合并当前档案和种群中的解
        all_solutions = []
        all_fitness = []

        # 添加档案中的解
        for item in self.archive:
            all_solutions.append(item['solution'])
            all_fitness.append(item['fitness'])

        # 添加当前种群中的解
        for i in range(len(self.population)):
            all_solutions.append(self.population[i])
            all_fitness.append(self.fitness[i])

        # 去重
        unique_solutions = []
        unique_fitness = []
        seen = set()
        for i in range(len(all_solutions)):
            sol_tuple = tuple(np.round(all_solutions[i], 4))
            if sol_tuple not in seen:
                seen.add(sol_tuple)
                unique_solutions.append(all_solutions[i])
                unique_fitness.append(all_fitness[i])

        # 找到非支配解
        if len(unique_solutions) > 0:
            fronts = non_dominated_sort(unique_solutions, unique_fitness)

            # 第一前沿就是非支配解
            if fronts and len(fronts[0]) > 0:
                # 清空档案，只保留非支配解
                self.archive = []
                for idx in fronts[0]:
                    self.archive.append({
                        'solution': unique_solutions[idx].copy(),
                        'fitness': unique_fitness[idx]
                    })

                # 如果档案过大，基于拥挤度移除解
                if len(self.archive) > self.archive_size:
                    archive_fitness = [item['fitness'] for item in self.archive]
                    cd = crowding_distance(archive_fitness)

                    # 按拥挤度排序，移除拥挤度小的解
                    sorted_indices = np.argsort(cd)  # 从小到大排序
                    self.archive = [self.archive[i] for i in sorted_indices[-self.archive_size:]]

    def differential_mutation(self):
        """差分变异"""
        pop_size = len(self.population)
        mutants = np.zeros_like(self.population)

        for i in range(pop_size):
            # 从种群中随机选择三个不同的个体
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            # 差分变异
            mutants[i] = self.population[r1] + self.F * (self.population[r2] - self.population[r3])

        return mutants

    def binomial_crossover(self, population, mutants):
        """二项式交叉"""
        pop_size = len(population)
        dim = self.dim
        trials = np.zeros_like(population)

        for i in range(pop_size):
            # 确保至少有一个维度来自变异个体
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < self.CR or j == j_rand:
                    trials[i, j] = mutants[i, j]
                else:
                    trials[i, j] = population[i, j]

        return trials

    def select_based_on_crowding(self, fitness1, fitness2):
        """基于拥挤度的选择"""
        # 创建一个临时种群用于计算拥挤度
        temp_fitness = [fitness1, fitness2]
        cd = crowding_distance(temp_fitness)

        # 选择拥挤度大的解
        if cd[0] > cd[1]:
            return 0  # 选择第一个解
        else:
            return 1  # 选择第二个解

    def evolve(self):
        """进化一代"""
        # 差分变异
        mutants = self.differential_mutation()

        # 二项式交叉，创建试验向量
        trials = self.binomial_crossover(self.population, mutants)

        # 修复试验向量
        for i in range(len(trials)):
            trials[i] = repair_solution(trials[i])

        # 评估试验向量
        trial_fitness = []
        for i in range(len(trials)):
            if is_feasible(trials[i]):
                f1 = compute_distance(trials[i])
                f2 = compute_risk(trials[i], self.risk_map)
                f3 = compute_battery_consumption(trials[i], self.risk_map)
                trial_fitness.append((f1, f2, f3))
            else:
                f1 = compute_distance(trials[i]) + 1000
                f2 = compute_risk(trials[i], self.risk_map) + 1000
                f3 = compute_battery_consumption(trials[i], self.risk_map) + 1000
                trial_fitness.append((f1, f2, f3))

        trial_fitness = np.array(trial_fitness)

        # 多目标选择
        new_population = []
        new_fitness = []

        for i in range(len(self.population)):
            parent_fit = self.fitness[i]
            trial_fit = trial_fitness[i]

            # 支配关系判断
            if dominates(trial_fit, parent_fit):
                # 试验向量支配父代，选择试验向量
                new_population.append(trials[i])
                new_fitness.append(trial_fit)
            elif dominates(parent_fit, trial_fit):
                # 父代支配试验向量，选择父代
                new_population.append(self.population[i])
                new_fitness.append(parent_fit)
            else:
                # 互不支配，基于拥挤度选择
                if self.select_based_on_crowding(parent_fit, trial_fit) == 0:
                    new_population.append(self.population[i])
                    new_fitness.append(parent_fit)
                else:
                    new_population.append(trials[i])
                    new_fitness.append(trial_fit)

        self.population = np.array(new_population)
        self.fitness = np.array(new_fitness)

        # 更新外部档案
        self.update_archive()

    def run(self):
        """运行算法"""
        pareto_fronts = {}

        for gen in range(self.max_gen):
            self.evolve()

            # 记录特定代的Pareto前沿
            if gen in [0, 9, 49, 99, 199]:
                # 获取当前档案中的解
                current_pareto = []
                for item in self.archive:
                    if (item['fitness'][0] < 1000 and
                            item['fitness'][1] < 1000 and
                            item['fitness'][2] < 1000):
                        current_pareto.append(item['fitness'])

                if current_pareto:
                    pareto_fronts[gen + 1] = current_pareto
                else:
                    # 如果没有可行解，使用档案中的所有解
                    pareto_fronts[gen + 1] = [item['fitness'] for item in self.archive]

            # 显示进度
            if (gen + 1) % 20 == 0:
                feasible_count = sum(1 for item in self.archive
                                     if item['fitness'][0] < 1000 and
                                     item['fitness'][1] < 1000 and
                                     item['fitness'][2] < 1000)
                print(f"MODE原始 Generation {gen + 1}: Archive size = {len(self.archive)}, Feasible = {feasible_count}")

        return pareto_fronts

    def get_pareto_front(self):
        """获取最终的Pareto前沿"""
        pareto_front = []
        for item in self.archive:
            if (item['fitness'][0] < 1000 and
                    item['fitness'][1] < 1000 and
                    item['fitness'][2] < 1000):
                pareto_front.append({
                    'solution': item['solution'],
                    'fitness': item['fitness']
                })

        if not pareto_front:
            # 如果没有可行解，返回档案中的所有解
            pareto_front = [{'solution': item['solution'], 'fitness': item['fitness']}
                            for item in self.archive]

        return pareto_front


# ==================== MOPSO 算法实现（修复版本） ====================
class MOPSO:
    def __init__(self, pop_size=100, K=5, max_gen=200, w=0.4, c1=1.5, c2=1.5):
        self.pop_size = pop_size
        self.K = K  # 控制点数量
        self.dim = K * 2  # 决策变量维度
        self.max_gen = max_gen
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        self.risk_map = create_risk_map()

        # 初始化粒子群
        self.positions = np.random.uniform(0, 100, (pop_size, self.dim))
        self.velocities = np.random.uniform(-10, 10, (pop_size, self.dim))

        # 初始化个体最优
        self.pbest_positions = self.positions.copy()

        # 评估初始种群
        self.fitness = []
        self.pbest_fitness = []
        for i in range(pop_size):
            self.positions[i] = repair_solution(self.positions[i])

            if is_feasible(self.positions[i]):
                f1 = compute_distance(self.positions[i])
                f2 = compute_risk(self.positions[i], self.risk_map)
                f3 = compute_battery_consumption(self.positions[i], self.risk_map)
                fitness_val = (f1, f2, f3)
            else:
                f1 = compute_distance(self.positions[i]) + 1000
                f2 = compute_risk(self.positions[i], self.risk_map) + 1000
                f3 = compute_battery_consumption(self.positions[i], self.risk_map) + 1000
                fitness_val = (f1, f2, f3)

            self.fitness.append(fitness_val)
            self.pbest_fitness.append(fitness_val)

        self.fitness = np.array(self.fitness)
        self.pbest_fitness = np.array(self.pbest_fitness)

        # 初始化全局最优（外部档案）
        self.archive = []
        self.update_archive()

        # 存储每代的最优解
        self.pareto_fronts_history = {}

    def update_archive(self):
        """更新外部档案"""
        # 添加当前种群的非支配解
        for i in range(self.pop_size):
            # 检查是否被档案中的任何解支配
            dominated = False
            for arch_item in list(self.archive):
                if dominates(arch_item['fitness'], self.fitness[i]):
                    dominated = True
                    break

            if not dominated:
                # 移除被新解支配的档案解
                self.archive = [item for item in self.archive
                                if not dominates(self.fitness[i], item['fitness'])]

                # 添加新解
                self.archive.append({
                    'position': self.positions[i].copy(),
                    'fitness': self.fitness[i].copy()
                })

        # 限制档案大小
        if len(self.archive) > self.pop_size:
            # 根据拥挤距离剪裁
            archive_fitness = [item['fitness'] for item in self.archive]
            cd = crowding_distance(archive_fitness)

            # 按拥挤距离排序，保留距离大的
            sorted_indices = np.argsort(cd)[::-1]
            self.archive = [self.archive[i] for i in sorted_indices[:self.pop_size]]

    def select_leader(self):
        """选择全局最优引导者 - 修复版本"""
        if not self.archive:
            return None

        # 根据拥挤距离选择引导者
        archive_fitness = [item['fitness'] for item in self.archive]
        cd = crowding_distance(archive_fitness)

        # 修复：避免除以零，处理NaN概率
        cd_array = np.array(cd)

        # 处理NaN和无穷值
        cd_array = np.nan_to_num(cd_array, nan=0.0, posinf=1.0, neginf=0.0)

        # 如果所有拥挤距离都为0，使用均匀分布
        if np.sum(cd_array) == 0:
            probabilities = np.ones(len(cd_array)) / len(cd_array)
        else:
            # 确保所有值都是非负的
            cd_array = np.maximum(cd_array, 0)
            total = np.sum(cd_array)

            # 再次检查是否为零
            if total > 0:
                probabilities = cd_array / total
            else:
                probabilities = np.ones(len(cd_array)) / len(cd_array)

        # 确保概率有效
        probabilities = np.maximum(probabilities, 0)  # 确保非负
        probabilities = probabilities / np.sum(probabilities)  # 重新归一化

        # 使用轮盘赌选择引导者
        try:
            leader_idx = np.random.choice(len(self.archive), p=probabilities)
        except ValueError as e:
            # 如果概率有问题，使用均匀分布
            leader_idx = np.random.randint(len(self.archive))

        return self.archive[leader_idx]

    def evolve(self):
        """进化一代"""
        for i in range(self.pop_size):
            # 选择引导者
            leader = self.select_leader()
            if leader is None:
                continue

            # 更新速度
            r1, r2 = np.random.rand(2)
            cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
            social = self.c2 * r2 * (leader['position'] - self.positions[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social

            # 限制速度范围
            self.velocities[i] = np.clip(self.velocities[i], -20, 20)

            # 更新位置
            self.positions[i] += self.velocities[i]
            self.positions[i] = repair_solution(self.positions[i])

            # 评估新位置
            if is_feasible(self.positions[i]):
                f1 = compute_distance(self.positions[i])
                f2 = compute_risk(self.positions[i], self.risk_map)
                f3 = compute_battery_consumption(self.positions[i], self.risk_map)
                new_fitness = (f1, f2, f3)
            else:
                f1 = compute_distance(self.positions[i]) + 1000
                f2 = compute_risk(self.positions[i], self.risk_map) + 1000
                f3 = compute_battery_consumption(self.positions[i], self.risk_map) + 1000
                new_fitness = (f1, f2, f3)

            # 更新个体最优
            # 如果新解支配原最优，则更新
            if dominates(new_fitness, self.pbest_fitness[i]):
                self.pbest_positions[i] = self.positions[i].copy()
                self.pbest_fitness[i] = new_fitness
            # 如果互不支配，随机选择（或根据拥挤距离）
            elif (not dominates(new_fitness, self.pbest_fitness[i]) and
                  not dominates(self.pbest_fitness[i], new_fitness)):
                if np.random.rand() < 0.5:  # 50%概率更新
                    self.pbest_positions[i] = self.positions[i].copy()
                    self.pbest_fitness[i] = new_fitness

            self.fitness[i] = new_fitness

        # 更新外部档案
        self.update_archive()

    def run(self):
        """运行算法"""
        pareto_fronts = {}

        for gen in range(self.max_gen):
            self.current_gen = gen
            self.evolve()

            # 记录特定代的Pareto前沿
            if gen in [0, 9, 49, 99, 199]:
                # 获取当前档案中的解
                current_pareto = []
                for item in self.archive:
                    if (item['fitness'][0] < 1000 and
                            item['fitness'][1] < 1000 and
                            item['fitness'][2] < 1000):
                        current_pareto.append(item['fitness'])

                if current_pareto:
                    pareto_fronts[gen + 1] = current_pareto
                else:
                    # 如果没有可行解，使用档案中的所有解
                    pareto_fronts[gen + 1] = [item['fitness'] for item in self.archive]

            # 显示进度
            if (gen + 1) % 20 == 0:
                feasible_count = sum(1 for item in self.archive
                                     if item['fitness'][0] < 1000 and
                                     item['fitness'][1] < 1000 and
                                     item['fitness'][2] < 1000)
                print(f"MOPSO Generation {gen + 1}: Archive size = {len(self.archive)}, Feasible = {feasible_count}")

        return pareto_fronts

    def get_pareto_front(self):
        """获取最终的Pareto前沿"""
        pareto_front = []
        for item in self.archive:
            if (item['fitness'][0] < 1000 and
                    item['fitness'][1] < 1000 and
                    item['fitness'][2] < 1000):
                pareto_front.append({
                    'solution': item['position'],
                    'fitness': item['fitness']
                })

        if not pareto_front:
            # 如果没有可行解，返回档案中的所有解
            pareto_front = [{'solution': item['position'], 'fitness': item['fitness']}
                            for item in self.archive]

        return pareto_front


# ==================== MODEM算法（用于比较） ====================
class ThreeObjective_MODEM:
    def __init__(self, pop_size=100, K=5, F=0.5, CR=0.9, max_gen=200):
        self.pop_size = pop_size
        self.K = K  # 控制点数量
        self.dim = K * 2  # 决策变量维度
        self.F = F
        self.CR = CR
        self.max_gen = max_gen
        self.risk_map = create_risk_map()

        # 初始化种群
        self.population = np.random.uniform(0, 100, (pop_size, self.dim))

        # 评估初始种群
        self.fitness = []
        for i in range(pop_size):
            # 先修复边界
            self.population[i] = repair_solution(self.population[i])

            if is_feasible(self.population[i]):
                f1 = compute_distance(self.population[i])
                f2 = compute_risk(self.population[i], self.risk_map)
                f3 = compute_battery_consumption(self.population[i], self.risk_map)
                self.fitness.append((f1, f2, f3))
            else:
                # 不可行解添加大罚项
                f1 = compute_distance(self.population[i]) + 1000
                f2 = compute_risk(self.population[i], self.risk_map) + 1000
                f3 = compute_battery_consumption(self.population[i], self.risk_map) + 1000
                self.fitness.append((f1, f2, f3))

        self.fitness = np.array(self.fitness)

        # 初始化档案
        self.archive = []
        self.update_archive(self.population, self.fitness)

    def update_archive(self, population, fitness):
        """更新档案库 - 简化的精英保留策略"""
        # 合并当前档案和新种群
        all_solutions = []
        all_fitness = []

        # 添加档案中的解
        for item in self.archive:
            all_solutions.append(item['solution'])
            all_fitness.append(item['fitness'])

        # 添加新种群中的解
        for i in range(len(population)):
            all_solutions.append(population[i])
            all_fitness.append(fitness[i])

        # 去重
        unique_solutions = []
        unique_fitness = []
        seen = set()
        for i in range(len(all_solutions)):
            sol_tuple = tuple(np.round(all_solutions[i], 4))
            if sol_tuple not in seen:
                seen.add(sol_tuple)
                unique_solutions.append(all_solutions[i])
                unique_fitness.append(all_fitness[i])

        # 非支配排序
        if len(unique_solutions) > 0:
            fronts = non_dominated_sort(unique_solutions, unique_fitness)

            # 构建新档案
            new_archive = []
            for front_idx, front in enumerate(fronts):
                if front_idx == 0:
                    # 第一前沿（Pareto前沿）全部保留
                    for idx in front:
                        new_archive.append({
                            'solution': unique_solutions[idx].copy(),
                            'fitness': unique_fitness[idx]
                        })
                else:
                    # 如果第一前沿已经足够多，停止添加
                    if len(new_archive) >= self.pop_size:
                        break

                    # 否则添加第二前沿的部分解
                    front_fitness = [unique_fitness[idx] for idx in front]
                    cd = crowding_distance(front_fitness)

                    # 按拥挤距离排序，选择最大的
                    sorted_indices = np.argsort(cd)[::-1]
                    remaining = self.pop_size - len(new_archive)
                    for i in range(min(remaining, len(front))):
                        idx = front[sorted_indices[i]]
                        new_archive.append({
                            'solution': unique_solutions[idx].copy(),
                            'fitness': unique_fitness[idx]
                        })

            # 如果档案还不够大，随机补充一些解
            if len(new_archive) < self.pop_size:
                remaining = self.pop_size - len(new_archive)
                for i in range(remaining):
                    idx = np.random.randint(len(unique_solutions))
                    new_archive.append({
                        'solution': unique_solutions[idx].copy(),
                        'fitness': unique_fitness[idx]
                    })

            self.archive = new_archive[:self.pop_size]  # 确保档案大小不超过种群大小

    def evolve(self):
        """进化一代"""
        # 从档案中选择父代
        if len(self.archive) < 3:
            # 如果档案太小，使用当前种群
            parent_pop = self.population
        else:
            # 从档案中选择父代
            indices = np.random.choice(len(self.archive), self.pop_size, replace=True)
            parent_pop = np.array([self.archive[i]['solution'] for i in indices])

        # 差分变异
        mutants = self.differential_mutation(parent_pop, self.F)

        # 交叉产生子代
        offspring = np.zeros_like(parent_pop)
        for i in range(self.pop_size):
            offspring[i] = self.binomial_crossover(parent_pop[i], mutants[i], self.CR)

        # 修复边界
        for i in range(self.pop_size):
            offspring[i] = repair_solution(offspring[i])

        # 评估子代
        offspring_fitness = []
        for i in range(self.pop_size):
            if is_feasible(offspring[i]):
                f1 = compute_distance(offspring[i])
                f2 = compute_risk(offspring[i], self.risk_map)
                f3 = compute_battery_consumption(offspring[i], self.risk_map)
                offspring_fitness.append((f1, f2, f3))
            else:
                # 不可行解添加大罚项
                f1 = compute_distance(offspring[i]) + 1000
                f2 = compute_risk(offspring[i], self.risk_map) + 1000
                f3 = compute_battery_consumption(offspring[i], self.risk_map) + 1000
                offspring_fitness.append((f1, f2, f3))

        offspring_fitness = np.array(offspring_fitness)

        # 合并父代和子代
        combined_pop = np.vstack([parent_pop, offspring])
        combined_fit = np.vstack([self.fitness, offspring_fitness])

        # 更新档案库
        self.update_archive(combined_pop, combined_fit)

        # 从档案库中选择新种群
        self.population = np.array([item['solution'] for item in self.archive])
        self.fitness = np.array([item['fitness'] for item in self.archive])

    def differential_mutation(self, population, F=0.5):
        """差分变异"""
        pop_size = len(population)
        mutated = np.zeros_like(population)

        for i in range(pop_size):
            # 选择三个不同的个体
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            # 差分变异
            mutated[i] = population[r1] + F * (population[r2] - population[r3])

        return mutated

    def binomial_crossover(self, parent, mutant, CR=0.9):
        """二项式交叉"""
        child = parent.copy()
        dim = len(parent)

        # 确保至少有一个维度来自变异个体
        j_rand = np.random.randint(dim)
        for j in range(dim):
            if np.random.rand() < CR or j == j_rand:
                child[j] = mutant[j]

        return child

    def run(self):
        """运行算法"""
        pareto_fronts = {}

        for gen in range(self.max_gen):
            self.evolve()

            # 记录第1, 10, 50, 100, 200代的Pareto前沿
            if gen in [0, 9, 49, 99, 199]:
                # 获取当前的非支配解
                current_pareto = []
                for item in self.archive:
                    # 只考虑可行解（没有罚项的）
                    if item['fitness'][0] < 1000 and item['fitness'][1] < 1000 and item['fitness'][2] < 1000:
                        current_pareto.append(item['fitness'])

                if current_pareto:
                    pareto_fronts[gen + 1] = current_pareto
                else:
                    # 如果没有可行解，使用档案中的所有解
                    pareto_fronts[gen + 1] = [item['fitness'] for item in self.archive]

            # 显示进度
            if (gen + 1) % 20 == 0:
                # 计算可行解数量
                feasible_count = sum(1 for item in self.archive
                                     if item['fitness'][0] < 1000 and item['fitness'][1] < 1000 and item['fitness'][
                                         2] < 1000)
                print(f"MODEM Generation {gen + 1}: Archive size = {len(self.archive)}, Feasible = {feasible_count}")

        return pareto_fronts

    def get_pareto_front(self):
        """获取最终的Pareto前沿"""
        pareto_front = []
        for item in self.archive:
            if (item['fitness'][0] < 1000 and
                    item['fitness'][1] < 1000 and
                    item['fitness'][2] < 1000):
                pareto_front.append({
                    'solution': item['solution'],
                    'fitness': item['fitness']
                })

        if not pareto_front:
            # 如果没有可行解，返回档案中的所有解
            pareto_front = [{'solution': item['solution'], 'fitness': item['fitness']}
                            for item in self.archive]

        return pareto_front


# ==================== 可视化函数 ====================
def visualize_environment():
    """可视化环境（障碍物和风险场）"""
    risk_map = create_risk_map()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：障碍物图
    ax1 = axes[0]
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')

    # 绘制障碍物
    for obs in OBSTACLES:
        rect = plt.Rectangle((obs[0], obs[2]),
                             obs[1] - obs[0],
                             obs[3] - obs[2],
                             facecolor='gray', alpha=0.7, label='障碍物')
        ax1.add_patch(rect)

    # 标记起点和终点
    ax1.scatter(START_POINT[0], START_POINT[1], color='green', s=200,
                marker='^', edgecolors='black', linewidth=2, label='起点', zorder=10)
    ax1.scatter(END_POINT[0], END_POINT[1], color='red', s=200,
                marker='v', edgecolors='black', linewidth=2, label='终点', zorder=10)

    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.set_title('障碍物分布图')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 右图：风险场图
    ax2 = axes[1]
    im = ax2.imshow(risk_map, extent=[0, 100, 0, 100], origin='lower',
                    cmap='YlOrRd', alpha=0.8, aspect='auto')

    # 添加颜色条
    plt.colorbar(im, ax=ax2, label='风险等级')

    # 绘制障碍物轮廓
    for obs in OBSTACLES:
        rect = plt.Rectangle((obs[0], obs[2]),
                             obs[1] - obs[0],
                             obs[3] - obs[2],
                             facecolor='none', edgecolor='black', alpha=0.3, linewidth=2)
        ax2.add_patch(rect)

    # 标记起点和终点
    ax2.scatter(START_POINT[0], START_POINT[1], color='green', s=200,
                marker='^', edgecolors='black', linewidth=2, label='起点', zorder=10)
    ax2.scatter(END_POINT[0], END_POINT[1], color='red', s=200,
                marker='v', edgecolors='black', linewidth=2, label='终点', zorder=10)

    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')
    ax2.set_title('风险场分布图')

    plt.tight_layout()
    plt.savefig('environment_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    return risk_map


def visualize_3d_pareto_comparison(pareto_fronts_dict):
    """比较不同算法的三目标Pareto前沿"""
    algorithms = list(pareto_fronts_dict.keys())
    colors = {'MOPSO': 'green', 'MODEM': 'red', 'MODE原始': 'orange'}
    markers = {'MOPSO': 's', 'MODEM': '^', 'MODE原始': 'D'}

    fig = plt.figure(figsize=(15, 10))

    # 1. 三维Pareto前沿比较（最终代）
    ax1 = fig.add_subplot(221, projection='3d')
    for algo in algorithms:
        if 200 in pareto_fronts_dict[algo]:
            front = pareto_fronts_dict[algo][200]
            if front and len(front) > 0:
                f1_vals = [f[0] for f in front]
                f2_vals = [f[1] for f in front]
                f3_vals = [f[2] for f in front]
                ax1.scatter(f1_vals, f2_vals, f3_vals,
                            c=colors[algo], marker=markers[algo], s=20,
                            alpha=0.6, label=algo)

    ax1.set_xlabel('路径长度 (f1)', labelpad=10)
    ax1.set_ylabel('风险暴露 (f2)', labelpad=10)
    ax1.set_zlabel('电池损耗% (f3)', labelpad=10)
    ax1.set_title('算法比较：三维Pareto前沿', fontsize=12)
    ax1.legend(fontsize=9)

    # 2. f1-f2平面投影比较
    ax2 = fig.add_subplot(222)
    for algo in algorithms:
        if 200 in pareto_fronts_dict[algo]:
            front = pareto_fronts_dict[algo][200]
            if front and len(front) > 0:
                f1_vals = [f[0] for f in front]
                f2_vals = [f[1] for f in front]
                ax2.scatter(f1_vals, f2_vals,
                            c=colors[algo], marker=markers[algo], s=30,
                            alpha=0.7, label=algo)

    ax2.set_xlabel('路径长度 (f1)')
    ax2.set_ylabel('风险暴露 (f2)')
    ax2.set_title('f1-f2平面投影比较')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. f1-f3平面投影比较
    ax3 = fig.add_subplot(223)
    for algo in algorithms:
        if 200 in pareto_fronts_dict[algo]:
            front = pareto_fronts_dict[algo][200]
            if front and len(front) > 0:
                f1_vals = [f[0] for f in front]
                f3_vals = [f[2] for f in front]
                ax3.scatter(f1_vals, f3_vals,
                            c=colors[algo], marker=markers[algo], s=30,
                            alpha=0.7, label=algo)

    ax3.set_xlabel('路径长度 (f1)')
    ax3.set_ylabel('电池损耗% (f3)')
    ax3.set_title('f1-f3平面投影比较')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. 算法性能指标比较
    ax4 = fig.add_subplot(224)

    # 计算每个算法的指标
    metrics = {}
    for algo in algorithms:
        if 200 in pareto_fronts_dict[algo]:
            front = pareto_fronts_dict[algo][200]
            if front and len(front) > 0:
                f1_vals = [f[0] for f in front]
                f2_vals = [f[1] for f in front]
                f3_vals = [f[2] for f in front]

                # 计算超体积的近似值（使用解的数量和多样性）
                if len(front) > 1:
                    cd = crowding_distance(front)
                    # 过滤掉无穷大值
                    finite_cd = [d for d in cd if np.isfinite(d)]
                    if finite_cd:
                        avg_cd = np.mean(finite_cd)
                        diversity_score = 1 / (1 + avg_cd) if avg_cd > 0 else 1
                    else:
                        diversity_score = 0.5
                else:
                    diversity_score = 0.5

                hypervolume_approx = len(front) * diversity_score

                metrics[algo] = {
                    '解的数量': len(front),
                    '平均f1': np.mean(f1_vals),
                    '平均f2': np.mean(f2_vals),
                    '平均f3': np.mean(f3_vals),
                    'f1范围': max(f1_vals) - min(f1_vals),
                    'f2范围': max(f2_vals) - min(f2_vals),
                    'f3范围': max(f3_vals) - min(f3_vals),
                    '超体积近似': hypervolume_approx
                }

    # 绘制柱状图
    if metrics:
        algo_names = list(metrics.keys())
        num_metrics = 4  # 显示4个指标
        x = np.arange(len(algo_names))
        width = 0.15

        # 创建多个子柱状图
        metric_names = ['解的数量', '平均f1', '平均f2', '超体积近似']
        metric_colors = ['blue', 'green', 'orange', 'red']

        for i, metric_name in enumerate(metric_names[:4]):
            values = [metrics[algo][metric_name] for algo in algo_names]
            # 归一化显示
            max_val = max(values) if max(values) > 0 else 1
            norm_values = [v / max_val for v in values]
            ax4.bar(x + i * width - width * 1.5, norm_values, width,
                    label=metric_name, color=metric_colors[i], alpha=0.7)

        ax4.set_xlabel('算法')
        ax4.set_ylabel('归一化值')
        ax4.set_title('算法性能比较')
        ax4.set_xticks(x)
        ax4.set_xticklabels(algo_names)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('三算法三目标无人机路径规划比较', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return metrics


def visualize_optimal_paths_comparison(algorithms_dict):
    """比较不同算法的最优路径"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    algorithm_names = list(algorithms_dict.keys())
    colors = ['red', 'blue', 'green']

    for i, (algo_name, algorithm) in enumerate(zip(algorithm_names, algorithms_dict.values())):
        if i >= len(axes):
            break

        ax = axes[i]

        # 获取Pareto前沿
        pareto_front = algorithm.get_pareto_front()
        if not pareto_front:
            ax.text(0.5, 0.5, f'{algo_name}\n无可行解',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            continue

        # 绘制风险场背景
        risk_map = algorithm.risk_map if hasattr(algorithm, 'risk_map') else create_risk_map()
        im = ax.imshow(risk_map, extent=[0, 100, 0, 100], origin='lower',
                       cmap='YlOrRd', alpha=0.3, aspect='auto')

        # 绘制障碍物
        for obs in OBSTACLES:
            rect = plt.Rectangle((obs[0], obs[2]),
                                 obs[1] - obs[0],
                                 obs[3] - obs[2],
                                 facecolor='darkgray', alpha=0.7, edgecolor='black',
                                 linewidth=1.5)
            ax.add_patch(rect)

        # 从Pareto前沿中选择几个代表性的解
        if len(pareto_front) >= 3:
            # 选择f1最小、f2最小、f3最小的解
            f1_min_idx = np.argmin([item['fitness'][0] for item in pareto_front])
            f2_min_idx = np.argmin([item['fitness'][1] for item in pareto_front])
            f3_min_idx = np.argmin([item['fitness'][2] for item in pareto_front])

            selected_solutions = [
                (pareto_front[f1_min_idx], '最短路径', colors[0]),
                (pareto_front[f2_min_idx], '最低风险', colors[1]),
                (pareto_front[f3_min_idx], '最低电池', colors[2])
            ]
        else:
            selected_solutions = [
                (pareto_front[0], '最优解', colors[0])
            ]

        # 绘制路径
        for item, label, color in selected_solutions:
            path = decode_path(item['solution'])
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]

            ax.plot(x_coords, y_coords, color=color, linewidth=2,
                    marker='o', markersize=6, label=label)

        # 标记起点和终点
        ax.scatter(START_POINT[0], START_POINT[1], color='lime', s=200,
                   marker='^', edgecolors='black', linewidth=2, label='起点', zorder=10)
        ax.scatter(END_POINT[0], END_POINT[1], color='darkred', s=200,
                   marker='v', edgecolors='black', linewidth=2, label='终点', zorder=10)

        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title(f'{algo_name} - 代表性路径')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle('不同算法路径规划结果比较', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('path_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_convergence_analysis(pareto_fronts_dict):
    """绘制算法收敛性分析"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    algorithms = list(pareto_fronts_dict.keys())
    colors = {'MOPSO': 'green', 'MODEM': 'red', 'MODE原始': 'orange'}
    markers = {'MOPSO': 's-', 'MODEM': '^-', 'MODE原始': 'D-'}

    # 记录的各代
    generations = [1, 10, 50, 100, 200]

    # 1. 每代的Pareto前沿大小
    ax1 = axes[0]
    for algo in algorithms:
        sizes = []
        for gen in generations:
            if gen in pareto_fronts_dict[algo]:
                sizes.append(len(pareto_fronts_dict[algo][gen]))
            else:
                sizes.append(0)
        ax1.plot(generations, sizes, markers[algo],
                 color=colors[algo], label=algo, alpha=0.7)

    ax1.set_xlabel('代数')
    ax1.set_ylabel('Pareto前沿大小')
    ax1.set_title('Pareto前沿大小收敛曲线')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. 每代的平均路径长度
    ax2 = axes[1]
    for algo in algorithms:
        avg_f1 = []
        for gen in generations:
            if gen in pareto_fronts_dict[algo]:
                front = pareto_fronts_dict[algo][gen]
                if front:
                    # 只考虑可行解
                    feasible_front = [f for f in front if f[0] < 1000]
                    if feasible_front:
                        avg_f1.append(np.mean([f[0] for f in feasible_front]))
                    else:
                        avg_f1.append(np.nan)
                else:
                    avg_f1.append(np.nan)
            else:
                avg_f1.append(np.nan)
        ax2.plot(generations, avg_f1, markers[algo],
                 color=colors[algo], label=algo, alpha=0.7)

    ax2.set_xlabel('代数')
    ax2.set_ylabel('平均路径长度')
    ax2.set_title('路径长度收敛曲线')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. 每代的平均风险暴露
    ax3 = axes[2]
    for algo in algorithms:
        avg_f2 = []
        for gen in generations:
            if gen in pareto_fronts_dict[algo]:
                front = pareto_fronts_dict[algo][gen]
                if front:
                    feasible_front = [f for f in front if f[1] < 1000]
                    if feasible_front:
                        avg_f2.append(np.mean([f[1] for f in feasible_front]))
                    else:
                        avg_f2.append(np.nan)
                else:
                    avg_f2.append(np.nan)
            else:
                avg_f2.append(np.nan)
        ax3.plot(generations, avg_f2, markers[algo],
                 color=colors[algo], label=algo, alpha=0.7)

    ax3.set_xlabel('代数')
    ax3.set_ylabel('平均风险暴露')
    ax3.set_title('风险暴露收敛曲线')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. 每代的平均电池损耗
    ax4 = axes[3]
    for algo in algorithms:
        avg_f3 = []
        for gen in generations:
            if gen in pareto_fronts_dict[algo]:
                front = pareto_fronts_dict[algo][gen]
                if front:
                    feasible_front = [f for f in front if f[2] < 1000]
                    if feasible_front:
                        avg_f3.append(np.mean([f[2] for f in feasible_front]))
                    else:
                        avg_f3.append(np.nan)
                else:
                    avg_f3.append(np.nan)
            else:
                avg_f3.append(np.nan)
        ax4.plot(generations, avg_f3, markers[algo],
                 color=colors[algo], label=algo, alpha=0.7)

    ax4.set_xlabel('代数')
    ax4.set_ylabel('平均电池损耗%')
    ax4.set_title('电池损耗收敛曲线')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.suptitle('算法收敛性分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("三目标无人机路径规划 - 三种算法比较")
    print("=" * 70)
    print("算法：MOPSO, MODEM, 原始MODE")
    print("目标函数：路径长度、风险暴露、电池损耗")
    print("=" * 70)

    # 算法参数
    pop_size = 100
    max_gen = 200
    K = 5

    print(f"\n算法参数:")
    print(f"  种群大小: {pop_size}")
    print(f"  最大代数: {max_gen}")
    print(f"  控制点数量: {K}")
    print(f"  起点: {START_POINT}")
    print(f"  终点: {END_POINT}")

    # 可视化环境
    print("\n可视化环境配置...")
    risk_map = visualize_environment()

    # 运行不同算法
    algorithms = {}
    pareto_fronts_dict = {}

    # 1. 运行MOPSO（修复版本）
    print("\n" + "=" * 50)
    print("开始运行MOPSO算法（修复版本）...")
    print("=" * 50)
    mopso = MOPSO(pop_size=pop_size, K=K, max_gen=max_gen, w=0.4, c1=1.5, c2=1.5)
    pareto_fronts_mopso = mopso.run()
    algorithms['MOPSO'] = mopso
    pareto_fronts_dict['MOPSO'] = pareto_fronts_mopso

    # 2. 运行MODEM
    print("\n" + "=" * 50)
    print("开始运行MODEM算法...")
    print("=" * 50)
    modem = ThreeObjective_MODEM(pop_size=pop_size, K=K, F=0.5, CR=0.9, max_gen=max_gen)
    pareto_fronts_modem = modem.run()
    algorithms['MODEM'] = modem
    pareto_fronts_dict['MODEM'] = pareto_fronts_modem

    # 3. 运行原始MODE
    print("\n" + "=" * 50)
    print("开始运行原始MODE算法...")
    print("=" * 50)
    mode_original = MODE_Original(pop_size=pop_size, K=K, F=0.5, CR=0.9, max_gen=max_gen)
    pareto_fronts_mode = mode_original.run()
    algorithms['MODE原始'] = mode_original
    pareto_fronts_dict['MODE原始'] = pareto_fronts_mode

    # 可视化比较结果
    print("\n生成算法比较可视化...")

    # 1. Pareto前沿比较
    metrics = visualize_3d_pareto_comparison(pareto_fronts_dict)

    # 2. 路径比较
    visualize_optimal_paths_comparison(algorithms)

    # 3. 收敛性分析
    plot_convergence_analysis(pareto_fronts_dict)

    # 算法性能分析
    print("\n" + "=" * 70)
    print("算法性能分析报告：")
    print("=" * 70)

    if metrics:
        # 输出详细指标
        for algo_name, algo_metrics in metrics.items():
            print(f"\n{algo_name} 性能指标:")
            print(f"  Pareto前沿大小: {algo_metrics['解的数量']} 个解")
            print(f"  平均路径长度: {algo_metrics['平均f1']:.2f}")
            print(f"  平均风险暴露: {algo_metrics['平均f2']:.2f}")
            print(f"  平均电池损耗: {algo_metrics['平均f3']:.2f}%")
            print(f"  路径长度范围: {algo_metrics['f1范围']:.2f}")
            print(f"  风险暴露范围: {algo_metrics['f2范围']:.2f}")
            print(f"  电池损耗范围: {algo_metrics['f3范围']:.2f}%")
            print(f"  超体积近似: {algo_metrics['超体积近似']:.4f}")

        # 找到最优算法
        best_hypervolume = max([metrics[algo]['超体积近似'] for algo in metrics])
        best_algorithms = [algo for algo in metrics if metrics[algo]['超体积近似'] == best_hypervolume]

        print(f"\n基于超体积近似的最佳算法: {', '.join(best_algorithms)} (值: {best_hypervolume:.4f})")
    else:
        print("\n无法计算算法性能指标，可能没有找到可行解。")

    print("\n" + "=" * 70)
    print("算法特点总结：")
    print("=" * 70)
    print("1. MOPSO:")
    print("   - 优点：收敛速度快，实现相对简单")
    print("   - 缺点：容易陷入局部最优，需要良好的参数调整")
    print("   - 适用场景：需要快速找到可行解，计算资源有限")
    print("   - 修复点：修复了select_leader函数中的概率计算问题")

    print("\n2. MODEM:")
    print("   - 优点：结合差分进化和档案库，探索能力强")
    print("   - 缺点：参数敏感，需要调整F和CR参数")
    print("   - 适用场景：复杂多模态问题，需要强探索能力")

    print("\n3. 原始MODE:")
    print("   - 优点：经典差分进化框架，实现简单直观")
    print("   - 缺点：选择机制相对简单，可能不如其他算法高效")
    print("   - 适用场景：作为基准算法，对比其他改进算法")

    print("\n算法实现细节对比:")
    print("- MOPSO: 基于粒子群优化，使用外部档案和拥挤距离选择引导者")
    print("- MODEM: 基于差分进化，结合档案库和非支配排序")
    print("- 原始MODE: 经典差分进化，使用支配关系和拥挤度选择")

    print("\n综合建议:")
    print("- 如果追求收敛速度：推荐使用MOPSO")
    print("- 如果问题非常复杂：推荐使用MODEM")
    print("- 原始MODE适合作为基准对比")
    print("- 实际应用中可根据需求选择或组合使用这些算法")

    # 保存结果
    print("\n保存结果...")
    import pickle

    results = {
        'algorithms': algorithms,
        'pareto_fronts': pareto_fronts_dict,
        'metrics': metrics
    }

    with open('algorithm_comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("实验完成！结果已保存到文件。")
    print("\n生成的文件:")
    print("  1. environment_visualization.png - 环境可视化")
    print("  2. algorithm_comparison.png - 算法比较")
    print("  3. path_comparison.png - 路径比较")
    print("  4. convergence_analysis.png - 收敛性分析")
    print("  5. algorithm_comparison_results.pkl - 完整结果数据")


if __name__ == "__main__":
    main()