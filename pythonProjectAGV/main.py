import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN

class MultiAGVEnv(gym.Env):
    def __init__(self):
        super(MultiAGVEnv, self).__init__()
        self.num_agvs = 3  # AGV数量
        self.num_tasks = 10  # 任务数量
        self.num_actions = 4  # 动作数量：0-上，1-下，2-左，3-右
        self.grid_size = (10, 10)  # 工厂栅格大小
        self.charge_pos = [(0, i) for i in range(self.grid_size[1])]  # 充电区
        self.pickup_pos = [(i, 0) for i in range(1, self.grid_size[0])]  # 拣货区
        self.rack_pos = [(i, j) for i in range(1, self.grid_size[0]) for j in range(1, self.grid_size[1]) if (i + j) % 3 != 0]  # 货架位置

        self.state_space = self.num_agvs * 4 + self.num_tasks * 4  # 状态空间维度：每个AGV的状态（坐标、电量、任务），每个任务的状态（优先级、完成状态、位置）
        self.action_space = spaces.Discrete(self.num_actions * self.num_agvs)  # 动作空间维度
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_space,), dtype=np.float32)

        self.state = None
        self.tasks = None
        self.batteries = None
        self.agv_positions = None
        self.reset()

    def reset(self):
        # 初始化状态
        self.state = np.zeros(self.state_space)
        self.tasks = np.random.rand(self.num_tasks, 4)  # 每个任务有四个属性：优先级、完成状态、x位置、y位置
        self.tasks[:, 1] = 0  # 初始化任务完成状态为0（未完成）
        self.batteries = np.ones(self.num_agvs) * 2  # 初始化每个AGV的电量为满电
        self.agv_positions = [self.charge_pos[i % len(self.charge_pos)] for i in range(self.num_agvs)]  # 初始化AGV位置为充电区

        self.state[self.num_agvs * 4:] = self.tasks.flatten()  # 更新状态为初始任务状态
        self.time_steps = 0  # 初始化时间步
        return self.state

    def step(self, action):
        agv_idx = action // self.num_actions  # AGV编号
        agv_action = action % self.num_actions  # AGV动作：0-上，1-下，2-左，3-右

        reward = 0
        done = False

        # 更新AGV位置
        new_pos = list(self.agv_positions[agv_idx])
        if agv_action == 0:  # 上
            new_pos[0] = max(new_pos[0] - 1, 0)
        elif agv_action == 1:  # 下
            new_pos[0] = min(new_pos[0] + 1, self.grid_size[0] - 1)
        elif agv_action == 2:  # 左
            new_pos[1] = max(new_pos[1] - 1, 0)
        elif agv_action == 3:  # 右
            new_pos[1] = min(new_pos[1] + 1, self.grid_size[1] - 1)

        # 检测碰撞
        if new_pos in self.agv_positions:
            reward -= 10  # 碰撞惩罚
        else:
            self.agv_positions[agv_idx] = tuple(new_pos)

        # 更新电量和状态
        self.batteries[agv_idx] -= 0.1  # 每步消耗电量
        if self.batteries[agv_idx] <= 0:
            reward -= 5  # 电量耗尽的惩罚

        # 检查是否在充电区
        if self.agv_positions[agv_idx] in self.charge_pos:
            self.batteries[agv_idx] = 2  # 充电

        # 检查是否在拣货区
        if self.agv_positions[agv_idx] in self.pickup_pos:
            for i in range(self.num_tasks):
                if self.state[self.num_agvs * 4 + 4 * i + 1] == 0:  # 找到未完成任务
                    task_priority = self.tasks[i, 0]  # 获取任务优先级
                    self.state[agv_idx * 4 + 2] = self.tasks[i, 2]  # 任务x位置
                    self.state[agv_idx * 4 + 3] = self.tasks[i, 3]  # 任务y位置
                    reward += task_priority * 10  # 基于任务优先级的奖励
                    self.state[self.num_agvs * 4 + 4 * i + 1] = 1  # 更新任务为完成状态
                    break

        # 检查是否所有任务都已完成
        if np.all(self.state[self.num_agvs * 4 + 1::4] == 1):
            done = True
            reward += 100  # 完成所有任务的大额奖励

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
        pass

# 创建环境实例
env = MultiAGVEnv()

# 初始化DQN算法
model = DQN('MlpPolicy', env, verbose=1)

# 训练代理
model.learn(total_timesteps=10000)

# 测试训练好的代理
state = env.reset()
done = False
while not done:
    action, _states = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
