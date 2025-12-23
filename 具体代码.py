import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


# ==================== 三目标优化算法 ====================
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


# ==================== 差分进化 (DE) 算子 ====================
def differential_mutation(population, F=0.5):
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


def binomial_crossover(parent, mutant, CR=0.9):
    """二项式交叉"""
    child = parent.copy()
    dim = len(parent)

    # 确保至少有一个维度来自变异个体
    j_rand = np.random.randint(dim)
    for j in range(dim):
        if np.random.rand() < CR or j == j_rand:
            child[j] = mutant[j]

    return child


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


# ==================== 改进的 MODEM 算法（三目标版本） ====================
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
        mutants = differential_mutation(parent_pop, self.F)

        # 交叉产生子代
        offspring = np.zeros_like(parent_pop)
        for i in range(self.pop_size):
            offspring[i] = binomial_crossover(parent_pop[i], mutants[i], self.CR)

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
                print(f"Generation {gen + 1}: Archive size = {len(self.archive)}, Feasible = {feasible_count}")

        return pareto_fronts


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


def visualize_3d_pareto(pareto_fronts, algorithm):
    """可视化三目标Pareto前沿"""

    # 使用更灵活的布局，创建4x3的子图（但有些位置留空）
    fig = plt.figure(figsize=(18, 16))

    # 设置子图布局
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.2], hspace=0.4, wspace=0.3)

    # 1. 三维Pareto前沿演化
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    generations = [1, 50, 100, 200]
    colors = ['red', 'blue', 'green', 'orange']

    for i, gen in enumerate(generations):
        if gen in pareto_fronts:
            front = pareto_fronts[gen]
            if front and len(front) > 0:
                f1_vals = [f[0] for f in front]
                f2_vals = [f[1] for f in front]
                f3_vals = [f[2] for f in front]
                ax1.scatter(f1_vals, f2_vals, f3_vals, c=colors[i], s=20,
                            alpha=0.6, label=f'第{gen}代')

    ax1.set_xlabel('路径长度 (f1)', labelpad=10)
    ax1.set_ylabel('风险暴露 (f2)', labelpad=10)
    ax1.set_zlabel('电池损耗% (f3)', labelpad=10)
    ax1.set_title('三目标Pareto前沿演化', fontsize=12)
    ax1.legend(fontsize=9)

    # 2. 二维投影：f1-f2
    ax2 = fig.add_subplot(gs[0, 1])
    if 200 in pareto_fronts:
        front = pareto_fronts[200]
        if front and len(front) > 0:
            f1_vals = [f[0] for f in front]
            f2_vals = [f[1] for f in front]
            ax2.scatter(f1_vals, f2_vals, c='blue', s=30, alpha=0.7)
            # 连接点形成前沿
            if len(front) > 1:
                sorted_front = sorted(front, key=lambda x: x[0])
                f1_sorted = [f[0] for f in sorted_front]
                f2_sorted = [f[1] for f in sorted_front]
                ax2.plot(f1_sorted, f2_sorted, c='blue', alpha=0.3, linewidth=1)

    ax2.set_xlabel('路径长度 (f1)')
    ax2.set_ylabel('风险暴露 (f2)')
    ax2.set_title('f1-f2 二维投影')
    ax2.grid(True, alpha=0.3)

    # 3. 二维投影：f1-f3
    ax3 = fig.add_subplot(gs[0, 2])
    if 200 in pareto_fronts:
        front = pareto_fronts[200]
        if front and len(front) > 0:
            f1_vals = [f[0] for f in front]
            f3_vals = [f[2] for f in front]
            ax3.scatter(f1_vals, f3_vals, c='green', s=30, alpha=0.7)
            # 连接点形成前沿
            if len(front) > 1:
                sorted_front = sorted(front, key=lambda x: x[0])
                f1_sorted = [f[0] for f in sorted_front]
                f3_sorted = [f[2] for f in sorted_front]
                ax3.plot(f1_sorted, f3_sorted, c='green', alpha=0.3, linewidth=1)

    ax3.set_xlabel('路径长度 (f1)')
    ax3.set_ylabel('电池损耗% (f3)')
    ax3.set_title('f1-f3 二维投影')
    ax3.grid(True, alpha=0.3)

    # 4. 二维投影：f2-f3
    ax4 = fig.add_subplot(gs[1, 0])
    if 200 in pareto_fronts:
        front = pareto_fronts[200]
        if front and len(front) > 0:
            f2_vals = [f[1] for f in front]
            f3_vals = [f[2] for f in front]
            ax4.scatter(f2_vals, f3_vals, c='red', s=30, alpha=0.7)
            # 连接点形成前沿
            if len(front) > 1:
                sorted_front = sorted(front, key=lambda x: x[1])
                f2_sorted = [f[1] for f in sorted_front]
                f3_sorted = [f[2] for f in sorted_front]
                ax4.plot(f2_sorted, f3_sorted, c='red', alpha=0.3, linewidth=1)

    ax4.set_xlabel('风险暴露 (f2)')
    ax4.set_ylabel('电池损耗% (f3)')
    ax4.set_title('f2-f3 二维投影')
    ax4.grid(True, alpha=0.3)

    # 5. 电池损耗与路径长度、风险的关系（热力图形式）
    ax5 = fig.add_subplot(gs[1, 1])
    if 200 in pareto_fronts:
        front = pareto_fronts[200]
        if front and len(front) > 0:
            f1_vals = np.array([f[0] for f in front])
            f2_vals = np.array([f[1] for f in front])
            f3_vals = np.array([f[2] for f in front])

            # 创建2D直方图/热力图
            # 使用hexbin创建六边形热力图
            hexbin = ax5.hexbin(f1_vals, f2_vals, C=f3_vals,
                                gridsize=15, cmap='viridis',
                                reduce_C_function=np.mean)

            ax5.set_xlabel('路径长度 (f1)')
            ax5.set_ylabel('风险暴露 (f2)')
            ax5.set_title('电池损耗热力图 (f1-f2平面)')
            ax5.grid(True, alpha=0.3)

            # 将颜色条放在底部 - 简单方法
            cbar = plt.colorbar(hexbin, ax=ax5, orientation='horizontal',
                                fraction=0.05, pad=0.15)
            cbar.set_label('电池损耗% (f3)', fontsize=10)
            cbar.ax.tick_params(labelsize=8)

    # 6. 电池损耗与路径长度、风险的关系（3D散点图）
    ax6 = fig.add_subplot(gs[1, 2], projection='3d')
    if 200 in pareto_fronts:
        front = pareto_fronts[200]
        if front and len(front) > 0:
            f1_vals = [f[0] for f in front]
            f2_vals = [f[1] for f in front]
            f3_vals = [f[2] for f in front]

            # 用颜色表示电池损耗
            scatter = ax6.scatter(f1_vals, f2_vals, f3_vals, c=f3_vals,
                                  cmap='viridis', s=30, alpha=0.8)

            # 对于3D图，使用一个简单的方法添加颜色条
            # 获取当前轴的位置
            pos = ax6.get_position()

            # 创建颜色条的轴（放在3D轴的正下方，但在第四行）
            cax = fig.add_axes([pos.x0, 0.05, pos.width, 0.03])

            # 添加水平颜色条
            cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
            cbar.set_label('电池损耗% (f3)', fontsize=10)
            cbar.ax.tick_params(labelsize=8)

    ax6.set_xlabel('路径长度 (f1)', labelpad=10)
    ax6.set_ylabel('风险暴露 (f2)', labelpad=10)
    ax6.set_zlabel('电池损耗% (f3)', labelpad=10)
    ax6.set_title('三目标分布 (3D散点图)', fontsize=12)

    # 7. 路径长度分布直方图
    ax7 = fig.add_subplot(gs[2, 0])
    if 200 in pareto_fronts:
        front = pareto_fronts[200]
        if front and len(front) > 0:
            f1_vals = [f[0] for f in front]
            ax7.hist(f1_vals, bins=15, color='blue', alpha=0.7, edgecolor='black')
            ax7.axvline(np.mean(f1_vals), color='red', linestyle='--',
                        label=f'平均值: {np.mean(f1_vals):.1f}')
            ax7.axvline(np.median(f1_vals), color='green', linestyle='--',
                        label=f'中位数: {np.median(f1_vals):.1f}')

    ax7.set_xlabel('路径长度 (f1)')
    ax7.set_ylabel('频率')
    ax7.set_title('路径长度分布')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # 8. 风险暴露分布直方图
    ax8 = fig.add_subplot(gs[2, 1])
    if 200 in pareto_fronts:
        front = pareto_fronts[200]
        if front and len(front) > 0:
            f2_vals = [f[1] for f in front]
            ax8.hist(f2_vals, bins=15, color='green', alpha=0.7, edgecolor='black')
            ax8.axvline(np.mean(f2_vals), color='red', linestyle='--',
                        label=f'平均值: {np.mean(f2_vals):.1f}')
            ax8.axvline(np.median(f2_vals), color='blue', linestyle='--',
                        label=f'中位数: {np.median(f2_vals):.1f}')

    ax8.set_xlabel('风险暴露 (f2)')
    ax8.set_ylabel('频率')
    ax8.set_title('风险暴露分布')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # 9. 电池损耗分布直方图
    ax9 = fig.add_subplot(gs[2, 2])
    if 200 in pareto_fronts:
        front = pareto_fronts[200]
        if front and len(front) > 0:
            f3_vals = [f[2] for f in front]
            ax9.hist(f3_vals, bins=15, color='orange', alpha=0.7, edgecolor='black')
            ax9.axvline(np.mean(f3_vals), color='red', linestyle='--',
                        label=f'平均值: {np.mean(f3_vals):.1f}')
            ax9.axvline(np.median(f3_vals), color='blue', linestyle='--',
                        label=f'中位数: {np.median(f3_vals):.1f}')

    ax9.set_xlabel('电池损耗% (f3)')
    ax9.set_ylabel('频率')
    ax9.set_title('电池损耗分布')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)

    # 添加一个空白轴作为底部间距（占位用）
    ax_blank = fig.add_subplot(gs[3, :])
    ax_blank.axis('off')

    plt.suptitle(f'三目标无人机路径规划结果 ({algorithm})', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(f'3d_pareto_{algorithm}.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_optimal_paths(algorithm_instance):
    """可视化最优路径（三目标）"""
    if not algorithm_instance.archive:
        print("没有找到可行解！")
        return

    # 只考虑可行解
    feasible_solutions = [item for item in algorithm_instance.archive
                          if item['fitness'][0] < 1000 and item['fitness'][1] < 1000 and item['fitness'][2] < 1000]

    if len(feasible_solutions) < 3:
        print(f"可行解不足，只有 {len(feasible_solutions)} 个可行解")
        # 使用所有解
        feasible_solutions = algorithm_instance.archive

    # 获取适应度和解
    archive_fitness = [item['fitness'] for item in feasible_solutions]
    archive_solutions = [item['solution'] for item in feasible_solutions]

    # 1. 最短路径（f1最小）
    shortest_idx = np.argmin([f[0] for f in archive_fitness])

    # 2. 最低风险路径（f2最小）
    safest_idx = np.argmin([f[1] for f in archive_fitness])

    # 3. 最低电池损耗路径（f3最小）
    lowest_battery_idx = np.argmin([f[2] for f in archive_fitness])

    # 4. 折中路径（到理想点距离最小，但排除前三个）
    # 计算理想点（最小值）
    f1_min = min([f[0] for f in archive_fitness])
    f2_min = min([f[1] for f in archive_fitness])
    f3_min = min([f[2] for f in archive_fitness])
    ideal_point = (f1_min, f2_min, f3_min)

    # 计算每个解到理想点的距离
    distances = []
    for i, fitness in enumerate(archive_fitness):
        if i not in [shortest_idx, safest_idx, lowest_battery_idx]:  # 排除前三个
            # 归一化距离
            f1_range = max([f[0] for f in archive_fitness]) - f1_min
            f2_range = max([f[1] for f in archive_fitness]) - f2_min
            f3_range = max([f[2] for f in archive_fitness]) - f3_min

            if f1_range > 0 and f2_range > 0 and f3_range > 0:
                norm_dist = np.sqrt(((fitness[0] - ideal_point[0]) / f1_range) ** 2 +
                                    ((fitness[1] - ideal_point[1]) / f2_range) ** 2 +
                                    ((fitness[2] - ideal_point[2]) / f3_range) ** 2)
            else:
                norm_dist = np.sqrt((fitness[0] - ideal_point[0]) ** 2 +
                                    (fitness[1] - ideal_point[1]) ** 2 +
                                    (fitness[2] - ideal_point[2]) ** 2)
            distances.append((norm_dist, i))

    # 选择距离最小的作为折中解
    if distances:
        balanced_idx = min(distances, key=lambda x: x[0])[1]
    else:
        # 如果没有其他解，选择第四个不同的解
        candidates = [i for i in range(len(archive_fitness))
                      if i not in [shortest_idx, safest_idx, lowest_battery_idx]]
        if candidates:
            balanced_idx = candidates[0]
        else:
            # 如果只有三个解，复制其中一个
            balanced_idx = shortest_idx

    # 选择四条路径
    selected_indices = [shortest_idx, safest_idx, lowest_battery_idx, balanced_idx]

    # 确保索引唯一
    unique_indices = []
    for idx in selected_indices:
        if idx not in unique_indices:
            unique_indices.append(idx)

    # 如果不足4个，补充随机解
    while len(unique_indices) < 4 and len(archive_solutions) >= 4:
        new_idx = np.random.randint(len(archive_solutions))
        if new_idx not in unique_indices:
            unique_indices.append(new_idx)

    # 创建标签和颜色
    labels = ["最短路径", "最低风险路径", "最低电池损耗", "折中解"]
    colors = ['red', 'blue', 'purple', 'green']

    selected_paths = []
    for i, idx in enumerate(unique_indices[:4]):
        label = labels[i] if i < len(labels) else f"路径{i + 1}"
        color = colors[i] if i < len(colors) else 'orange'
        selected_paths.append((label, archive_solutions[idx], color))

    # 绘制路径
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制风险场背景
    risk_map = algorithm_instance.risk_map
    im = ax.imshow(risk_map, extent=[0, 100, 0, 100], origin='lower',
                   cmap='YlOrRd', alpha=0.3, aspect='auto')

    # 将颜色条放在图下方
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.6)
    plt.colorbar(im, cax=cax, orientation='horizontal', label='风险等级')

    # 绘制障碍物
    for obs in OBSTACLES:
        rect = plt.Rectangle((obs[0], obs[2]),
                             obs[1] - obs[0],
                             obs[3] - obs[2],
                             facecolor='darkgray', alpha=0.7, edgecolor='black',
                             linewidth=1.5, label='障碍物')
        ax.add_patch(rect)

    # 绘制路径
    for label, solution, color in selected_paths:
        path = decode_path(solution)
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]

        # 计算路径的目标值
        f1_val = compute_distance(solution)
        f2_val = compute_risk(solution, risk_map)
        f3_val = compute_battery_consumption(solution, risk_map)

        ax.plot(x_coords, y_coords, color=color, linewidth=3,
                marker='o', markersize=8,
                label=f'{label}\n长度: {f1_val:.1f}\n风险: {f2_val:.0f}\n电池: {f3_val:.1f}%')

        # 标记控制点（不包括起点终点）
        if len(x_coords) > 2:
            ax.scatter(x_coords[1:-1], y_coords[1:-1], color=color, s=100,
                       marker='s', edgecolors='black', zorder=5)

    # 标记起点和终点
    ax.scatter(START_POINT[0], START_POINT[1], color='lime', s=250,
               marker='^', edgecolors='black', linewidth=3, label='起点', zorder=10)
    ax.scatter(END_POINT[0], END_POINT[1], color='darkred', s=250,
               marker='v', edgecolors='black', linewidth=3, label='终点', zorder=10)

    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title('三目标无人机路径规划结果 - 电池损耗模型', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    plt.tight_layout()
    plt.savefig('3objective_path_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 输出路径信息
    print("\n" + "=" * 70)
    print("代表性路径信息（三目标）：")
    print("=" * 70)
    for label, solution, _ in selected_paths:
        f1_val = compute_distance(solution)
        f2_val = compute_risk(solution, risk_map)
        f3_val = compute_battery_consumption(solution, risk_map)
        print(f"\n{label}:")
        print(f"  路径长度: {f1_val:.2f}")
        print(f"  风险暴露: {f2_val:.2f}")
        print(f"  电池损耗: {f3_val:.2f}%")

        # 显示控制点坐标
        path = decode_path(solution)
        print("  控制点坐标:")
        for i, point in enumerate(path[1:-1]):  # 排除起点终点
            print(f"    控制点{i + 1}: ({point[0]:.1f}, {point[1]:.1f})")


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("三目标无人机路径规划 - 改进的MODEM算法实现")
    print("=" * 70)
    print("新增目标：电池损耗模型")
    print("=" * 70)

    # 算法参数
    pop_size = 100
    max_gen = 200
    F = 0.5
    CR = 0.9

    print(f"\n算法参数:")
    print(f"  种群大小: {pop_size}")
    print(f"  最大代数: {max_gen}")
    print(f"  控制点数量: {K}")
    print(f"  起点: {START_POINT}")
    print(f"  终点: {END_POINT}")
    print(f"  障碍物数量: {len(OBSTACLES)}")

    print(f"\n电池模型参数:")
    print(f"  电池总容量: {BATTERY_CAPACITY} mAh")
    print(f"  每米飞行能耗: {ENERGY_PER_METER} mAh/m")
    print(f"  每单位风险能耗: {ENERGY_PER_RISK} mAh/单位风险")

    # 可视化环境
    print("\n可视化环境配置...")
    risk_map = visualize_environment()

    # 创建并运行三目标MODEM算法
    print("\n开始运行三目标MODEM算法...")
    modem_3d = ThreeObjective_MODEM(pop_size=pop_size, K=K, F=F, CR=CR, max_gen=max_gen)
    pareto_fronts_3d = modem_3d.run()

    # 可视化结果
    print("\n生成三目标Pareto前沿可视化...")
    visualize_3d_pareto(pareto_fronts_3d, "ThreeObjective_MODEM")

    print("\n生成最优路径可视化...")
    visualize_optimal_paths(modem_3d)

    # 算法分析
    print("\n" + "=" * 70)
    print("三目标算法性能分析：")
    print("=" * 70)

    if modem_3d.archive:
        # 统计可行解
        feasible_solutions = [item for item in modem_3d.archive
                              if item['fitness'][0] < 1000 and item['fitness'][1] < 1000 and item['fitness'][2] < 1000]

        print(f"最终档案库包含 {len(modem_3d.archive)} 个解")
        print(f"其中可行解: {len(feasible_solutions)} 个")

        if feasible_solutions:
            final_front = [item['fitness'] for item in feasible_solutions]
            f1_vals = [f[0] for f in final_front]
            f2_vals = [f[1] for f in final_front]
            f3_vals = [f[2] for f in final_front]

            print(f"\n目标值范围:")
            print(f"  路径长度范围: {min(f1_vals):.2f} - {max(f1_vals):.2f}")
            print(f"  风险暴露范围: {min(f2_vals):.2f} - {max(f2_vals):.2f}")
            print(f"  电池损耗范围: {min(f3_vals):.2f}% - {max(f3_vals):.2f}%")

            # 计算相关性
            corr_f1_f2 = np.corrcoef(f1_vals, f2_vals)[0, 1]
            corr_f1_f3 = np.corrcoef(f1_vals, f3_vals)[0, 1]
            corr_f2_f3 = np.corrcoef(f2_vals, f3_vals)[0, 1]

            print(f"\n目标间相关性:")
            print(f"  路径长度 vs 风险暴露: {corr_f1_f2:.3f}")
            print(f"  路径长度 vs 电池损耗: {corr_f1_f3:.3f}")
            print(f"  风险暴露 vs 电池损耗: {corr_f2_f3:.3f}")

        # 工程意义分析
        print("\n三目标工程意义分析:")
        print("1. 最短路径方案：最快到达，但风险高、电池消耗大")
        print("2. 最低风险方案：最安全，但路径长、电池消耗中等")
        print("3. 最低电池损耗：最省电，通常路径短且避开高风险区")
        print("4. 折中方案：平衡三个目标，适合大多数应用场景")

        print("\n应用建议:")
        print("- 紧急救援：选择最短路径，时间最重要")
        print("- 长时间监测：选择最低电池损耗，延长续航")
        print("- 危险品运输：选择最低风险，安全第一")
        print("- 常规任务：选择折中方案，平衡各项指标")

        print("\n电池模型说明:")
        print("- 电池损耗 = 基础能耗(与距离成正比) + 风险能耗(与风险暴露成正比)")
        print("- 基础能耗模拟电机和飞行控制系统的消耗")
        print("- 风险能耗模拟在危险区域需要额外能量(如加速、开启避障系统)")
        print("- 电池损耗百分比表示消耗的电池容量占总容量的比例")
    else:
        print("未找到可行解！")

    print("\n实验完成！")


if __name__ == "__main__":
    main()