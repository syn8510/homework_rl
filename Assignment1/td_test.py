from pathlib import Path
import sys
root = Path(__file__).parent.parent
sys.path.append(str(root))
from env.grid_scenarios import MiniWorld
import numpy as np

GAMMA = 0.9
EPSILON = 0.001
LR = 0.001
BLOCKS = [14, 15, 21, 27]


class TemporalDifference():
    def __init__(self, env):
        self.env = env
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n
        self.Q_table = np.zeros([self.n_state, self.n_action])  # 初始化Q(s,a)表格
        self.alpha = LR  # 学习率
        self.gamma = GAMMA  # 折扣因子
        self.epsilon = EPSILON  # epsilon-贪婪策略中的参数
        self.V = np.zeros(self.n_state)

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])  # 这些Q值中最大值所对应的动作的索引。
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        cnt = 0
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
                cnt += 1
        return a, cnt

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


def print_agent(agent, action_meaning):
    for i in range(agent.env.n_height - 1, -1, -1):
        for j in range(agent.env.n_width):
            if agent.env._is_end_state(j, i):
                print('到终点了', end=' ')
            elif agent.env.grids.get_dtype(j, i) == 1:  # 墙
                print('此路不通', end=' ')
            else:
                a, cnt = agent.best_action(i * agent.env.n_width + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else '无'
                print(pi_str, end=' ')
        print()


if __name__ == "__main__":
    env = MiniWorld()
    env.reset()  # 重置环境状态
    np.random.seed(0)#设置随机种子
    agent = TemporalDifference(env)

    num_episodes = 20000  # 智能体在环境中运行的序列的数量
    return_list = []  # 记录每一条序列的回报
    for i in range(num_episodes):  
        episode_return = 0
        state = env.reset()
        action = agent.take_action(state)
        done = False
        t = 0
        while not done and t < env.max_step:
            t += 1
            next_state, reward, done, info = env.step(action)
            next_action = agent.take_action(next_state)
            episode_return += reward  # 这里回报的计算不进行折扣因子衰减
            # print(state, action, reward, next_state, next_action)
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
        return_list.append(episode_return)
    
    action_meaning = ['左', '右', '上', '下']
    for i in range(agent.n_state):
        agent.V[i] = EPSILON * np.sum(agent.Q_table[i]) / agent.n_action + (1 - EPSILON) * np.max(agent.Q_table[i])

    for j in range(agent.env.n_height):
        for i in range(agent.env.n_width):
            if agent.env._is_end_state(i, j) or agent.env.grids.get_dtype(i, j) == 1:
                agent.V[i + j * agent.env.n_width] = agent.env.grids.get_reward(i, j)
    for i in range(5, -1, -1):
        for j in range(6):
            print("%.4f" % (agent.V[i * 6 + j]), end='\t')
        print()
    env.update_r(agent.V)
    print('Sarsa算法最终收敛得到的策略为：')
    print_agent(agent, action_meaning)
    env.reset()
    for _ in range(2000):
        env.render()
    env.close()
    print("env closed")
