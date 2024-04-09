from pathlib import Path
import sys
root = Path(__file__).parent.parent
sys.path.append(str(root))
from env.grid_scenarios import MiniWorld
import numpy as np

GAMMA = 0.9
EPSILON = 0.3
LR = 0.001
BLOCKS = [14, 15, 21, 27]


class MonteCarlo():
    def __init__(self, env):
        self.env = env
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n
        self.Q_table = np.zeros([self.n_state, self.n_action])  # 初始化Q(s,a)表格
        self.alpha = LR  # 学习率
        self.gamma = GAMMA  # 折扣因子
        self.epsilon = EPSILON  # epsilon-贪婪策略中的参数
        self.V = np.zeros(self.n_state)
        self.samples = []

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])  # 这些Q值中最大值所对应的动作的索引。
        return action

    def save_sample(self, state, action, reward, next_state, next_action):
        self.samples.append([state, action, reward, next_state, next_action])

    def update(self):
        G_t = 0
        for reward in reversed(self.samples):
            state, action, reward, next_state, next_action = reward
            G_t = reward + self.gamma * self.Q_table[next_state, next_action] - self.Q_table[state, action]
            self.Q_table[state, action] += self.alpha * G_t

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        cnt = 0
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
                cnt += 1
        return a, cnt


def print_agent(agent, action_meaning):
    print("策略：")
    for i in range(agent.env.n_height - 1, -1, -1):
        for j in range(agent.env.n_width):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if agent.env._is_end_state(j, i):
                print('EEEE', end=' ')
            elif agent.env.grids.get_dtype(j, i) == 1:  # 墙
                print('****', end=' ')
            else:
                a, cnt = agent.best_action(i * agent.env.n_width + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == "__main__":
    env = MiniWorld()
    env.reset()
    np.random.seed(0)
    agent = MonteCarlo(env)

    num_episodes = 50000  # 智能体在环境中运行的序列的数量
    for episode in range(num_episodes):
        state = env.reset()
        action = agent.take_action(state)
        done = False
        t = 0
        while not done and t < env.max_step:
            # print("episode : ", episode)
            t += 1
            next_state, reward, done, info = env.step(action)
            next_action = agent.take_action(next_state)
            agent.save_sample(state, action, reward, next_state, next_action)
            # print(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            # print(done)
            if done:
                # print("episode : ", episode)
                agent.update()
                agent.samples.clear()
                break

    action_meaning = ['<', '>', '^', 'v']
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
    print_agent(agent, action_meaning)
    env.reset()
    for _ in range(3000):
        env.render()
    env.close()
    print("env closed")
