import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
import matplotlib.pyplot as plt


class MultiAGVEnv(gym.Env):
    def __init__(self):
        super(MultiAGVEnv, self).__init__()
        self.num_agvs = 3  # AGV数量
        self.num_tasks = 10  # 任务数量
        self.num_actions = 4  # 动作数量：0-上，1-下，2-左，3-右
        self.grid_size = (20, 20)  # 工厂栅格大小
        self.charge_pos = [(0, i + 3) for i in range(self.num_agvs)]  # 充电区
        self.pickup_pos = [(i + 4, 0) for i in range(1, 6)]  # 拣货区
        self.rack_size = (2, 6)  # 货架大小
        self.create_warehouse()  # 创建仓库地图

        self.state_space = self.num_agvs * 4 + self.num_tasks * 4  # 状态空间维度
        self.action_space = spaces.Discrete(self.num_actions * self.num_agvs)  # 动作空间维度
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_space,), dtype=np.float32)

        self.state = None
        self.tasks = None
        self.batteries = None
        self.agv_positions = None
        self.task_completion_times = []
        self.reset()

        # 可视化初始化
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def create_warehouse(self):
        self.road_positions = set()  # 存储道路位置的集合
        self.rack_positions = []  # 存储货架位置的列表
        self.pickup_positions = set()  # 存储取货位置的集合
        start_row = 3  # 货架起始行，从第三行开始放置货架
        distance_between_racks = 2  # 水平方向上货架之间的间距为2
        left_distance = 3  # 最左面货架与拣货区的水平距离为3

        # 设置货架的列数（水平放置的货架数）
        num_rack_columns = 2
        # 设置货架的行数（垂直放置的货架数）
        num_rack_rows = 5

        for j in range(left_distance, left_distance + num_rack_columns * (self.rack_size[1] + distance_between_racks),
                       self.rack_size[1] + distance_between_racks):
            for i in range(start_row, start_row + num_rack_rows * (self.rack_size[0] + 1), self.rack_size[0] + 1):
                for x in range(i, i + self.rack_size[0]):
                    for y in range(j, j + self.rack_size[1]):
                        self.rack_positions.append((x, y))
                        self.pickup_positions.update({
                            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)
                        })

        # 道路位置包括整个网格的所有格子，除了货架位置和拣货位置
        non_road_positions = set(self.rack_positions).union(set(self.pickup_pos))
        self.road_positions = set(
            (x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])) - non_road_positions

        # 调试信息
        print(f"Rack Positions: {self.rack_positions}")
        print(f"Pickup Positions: {self.pickup_positions}")
        print(f"Charge Positions: {self.charge_pos}")
        print(f"Road Positions: {self.road_positions}")
        print(f"Non-road Positions: {non_road_positions}")

    def reset(self):
        # 初始化状态
        self.state = np.zeros(self.state_space)
        rack_positions = np.array(self.rack_positions)
        task_positions = rack_positions[
            np.random.choice(len(rack_positions), self.num_tasks, replace=False)]  # 确保任务位置在货架上
        task_priorities = np.random.uniform(0.1, 1.0, self.num_tasks)  # 随机任务优先级
        self.tasks = np.column_stack(
            (task_priorities, np.zeros(self.num_tasks), task_positions[:, 0], task_positions[:, 1]))  # 固定任务
        self.batteries = np.ones(self.num_agvs) * 2  # 初始化每个AGV的电量为满电
        self.agv_positions = [self.charge_pos[i % len(self.charge_pos)] for i in range(self.num_agvs)]  # 初始化AGV位置为充电区

        # 分配最近的任务给每个AGV
        for i in range(self.num_agvs):
            closest_task_idx = np.argmin(np.sum(np.abs(task_positions - np.array(self.agv_positions[i])), axis=1))
            self.state[i * 4 + 2] = task_positions[closest_task_idx, 0]
            self.state[i * 4 + 3] = task_positions[closest_task_idx, 1]

        self.state[self.num_agvs * 4:] = self.tasks.flatten()  # 更新状态为初始任务状态
        self.time_steps = 0  # 初始化时间步
        self.task_completion_times = []  # 任务完成时间
        return self.state

    def step(self, action):
        agv_idx = action // self.num_actions  # AGV编号
        agv_action = action % self.num_actions  # AGV动作：0-上，1-下，2-左，3-右

        reward = 0
        done = False

        # 更新AGV位置
        old_pos = self.agv_positions[agv_idx]
        new_pos = list(old_pos)

        # 尝试移动到指定方向
        if agv_action == 0:  # 上
            new_pos[0] = max(new_pos[0] - 1, 0)
        elif agv_action == 1:  # 下
            new_pos[0] = min(new_pos[0] + 1, self.grid_size[0] - 1)
        elif agv_action == 2:  # 左
            new_pos[1] = max(new_pos[1] - 1, 0)
        elif agv_action == 3:  # 右
            new_pos[1] = min(new_pos[1] + 1, self.grid_size[1] - 1)

        # 检查是否回到原位置
        if tuple(new_pos) == old_pos:
            reward -= 0.5  # 返回原位置的惩罚
            print(f"AGV {agv_idx} returned to original position: {new_pos}")

        # 更新AGV位置
        self.agv_positions[agv_idx] = tuple(new_pos)

        # 更新电量和状态
        self.batteries[agv_idx] -= 0.02  # 每步消耗电量
        if self.batteries[agv_idx] <= 0:
            reward -= 5  # 电量耗尽的惩罚
            print(f"AGV {agv_idx} battery depleted")

        # 检查是否在充电区
        if self.agv_positions[agv_idx] in self.charge_pos:
            if self.batteries[agv_idx] == 10:
                reward -= 1
                print(f"No meaning")
            self.batteries[agv_idx] = 10  # 充电
            print(f"AGV {agv_idx} charging")

        # 检查是否需要改变任务为充电
        if self.batteries[agv_idx] <= 4:
            # 寻找最近的空余充电区格子
            min_distance = float('inf')
            nearest_charge_pos = None
            for charge_pos in self.charge_pos:
                if charge_pos not in self.agv_positions:  # 空余的充电区格子
                    distance = abs(charge_pos[0] - self.agv_positions[agv_idx][0]) + abs(
                        charge_pos[1] - self.agv_positions[agv_idx][1])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_charge_pos = charge_pos

            # 更新当前AGV的任务为前往充电
            self.state[agv_idx * 4 + 2] = nearest_charge_pos[0]
            self.state[agv_idx * 4 + 3] = nearest_charge_pos[1]

        # 检查新位置是否在道路上或者是取货位置
        if tuple(new_pos) in self.rack_positions or tuple(new_pos) in self.pickup_pos:
            print(f"Invalid position, trying alternative direction.")
            # 尝试选择其他方向移动
            alternative_actions = [act for act in range(self.num_actions) if act != agv_action]
            np.random.shuffle(alternative_actions)
            for alt_action in alternative_actions:
                alt_pos = list(old_pos)
                if alt_action == 0:  # 上
                    alt_pos[0] = max(alt_pos[0] - 1, 0)
                elif alt_action == 1:  # 下
                    alt_pos[0] = min(alt_pos[0] + 1, self.grid_size[0] - 1)
                elif alt_action == 2:  # 左
                    alt_pos[1] = max(alt_pos[1] - 1, 0)
                elif alt_action == 3:  # 右
                    alt_pos[1] = min(alt_pos[1] + 1, self.grid_size[1] - 1)

                if tuple(alt_pos) not in self.rack_positions and tuple(alt_pos) not in self.pickup_pos:
                    new_pos = alt_pos
                    print(f"Alternative position found: {new_pos}")
                    break

        # 更新AGV位置
        self.agv_positions[agv_idx] = tuple(new_pos)
        print(f"AGV {agv_idx} moved to: {new_pos}")

        # 检查是否在取货区并且相邻货架格子有任务
        if self.agv_positions[agv_idx] in self.pickup_positions:
            for i in range(self.num_tasks):
                task_x, task_y = int(self.tasks[i, 2]), int(self.tasks[i, 3])
                if self.state[self.num_agvs * 4 + 4 * i + 1] == 0:  # 找到未完成任务
                    if (abs(task_x - self.agv_positions[agv_idx][0]) == 1 and task_y == self.agv_positions[agv_idx][
                        1]) or \
                            (abs(task_y - self.agv_positions[agv_idx][1]) == 1 and task_x ==
                             self.agv_positions[agv_idx][0]):
                        task_priority = self.tasks[i, 0]  # 获取任务优先级
                        reward += task_priority * 10  # 基于任务优先级的奖励
                        self.state[self.num_agvs * 4 + 4 * i + 1] = 1  # 更新任务为完成状态
                        self.task_completion_times.append(self.time_steps)
                        print(f"AGV {agv_idx} completed task at: {(task_x, task_y)}")
                        # 检查电量是否足够完成下一个任务
                        if self.batteries[agv_idx] <= 0.1 * np.sum(
                                np.abs(np.array(self.agv_positions[agv_idx]) - np.array(
                                    self.charge_pos[agv_idx % len(self.charge_pos)]))):
                            self.agv_positions[agv_idx] = self.charge_pos[agv_idx % len(self.charge_pos)]
                        break

        # 检查是否所有任务都已完成
        if np.all(self.state[self.num_agvs * 4 + 1::4] == 1):
            done = True
            reward += 100  # 完成所有任务的大额奖励
            print("All tasks completed")

        # 计算曼哈顿距离变化奖励或惩罚
        for i in range(self.num_tasks):
            task_x, task_y = int(self.tasks[i, 2]), int(self.tasks[i, 3])
            current_dist = abs(task_x - self.agv_positions[agv_idx][0]) + abs(task_y - self.agv_positions[agv_idx][1])
            if self.state[self.num_agvs * 4 + 4 * i + 1] == 0:  # 未完成任务
                old_dist = current_dist + 1  # 避免除以零
                if old_dist > 0:
                    distance_change = (old_dist - current_dist) / old_dist
                    if self.batteries[agv_idx] <= 3:
                        # 计算与充电区空余位置的曼哈顿距离
                        charge_distances = [abs(pos[0] - self.agv_positions[agv_idx][0]) +
                                            abs(pos[1] - self.agv_positions[agv_idx][1]) for pos in self.charge_pos]
                        min_charge_distance = min(charge_distances)
                        reward += distance_change * (10 - min_charge_distance)  # 根据距离变化奖励或惩罚
                    else:
                        reward += distance_change * 5  # 根据距离变化奖励或惩罚

        # 时间步增加
        self.time_steps += 1
        reward -= 1  # 每一步时间成本的惩罚

        # 更新状态
        for i in range(self.num_agvs):
            self.state[i * 4] = self.agv_positions[i][0]
            self.state[i * 4 + 1] = self.agv_positions[i][1]
            self.state[i * 4 + 2] = self.batteries[i]
        self.state[self.num_agvs * 4:] = self.tasks.flatten()

        return self.state, reward, done, {}

    def render(self, mode='human'):
        self.ax.clear()
        # 绘制道路
        for road_pos in self.road_positions:
            self.ax.plot(road_pos[1], road_pos[0], 's', color='gray')
        # 绘制货架
        for rack_pos in self.rack_positions:
            self.ax.plot(rack_pos[1], rack_pos[0], 's', color='brown')
        # 绘制充电区
        for charge in self.charge_pos:
            self.ax.plot(charge[1], charge[0], 's', color='blue')
        # 绘制拣货区
        for pickup in self.pickup_pos:
            self.ax.plot(pickup[1], pickup[0], 's', color='green')
        # 绘制AGV位置
        for i, agv_pos in enumerate(self.agv_positions):
            self.ax.plot(agv_pos[1], agv_pos[0], 'o', color='red', label=f'AGV {i + 1}')
        # 绘制任务
        for task in self.tasks:
            if task[1] == 0:  # 未完成任务
                self.ax.plot(task[3], task[2], 'x', color='yellow')

        self.ax.set_xlim(-1, self.grid_size[1])
        self.ax.set_ylim(-1, self.grid_size[0])
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        plt.draw()
        plt.pause(0.01)


# 创建环境实例
env = MultiAGVEnv()

# 初始化DQN算法
model = DQN('MlpPolicy', env, verbose=1)

# 训练代理
model.learn(total_timesteps=50000)

# 测试训练好的代理并进行可视化
state = env.reset()
done = False
step_count = 0

while not done:
    action, _ = model.predict(state, deterministic=True)
    print(f"Step {step_count}, Action: {action}")
    state, reward, done, info = env.step(action)
    print(f"Step {step_count}, State: {state}, Reward: {reward}, Done: {done}")
    env.render()
    step_count += 1

plt.ioff()
plt.show()
