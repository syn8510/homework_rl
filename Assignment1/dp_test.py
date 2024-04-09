import copy
import numpy as np
from pathlib import Path
import sys
root = Path(__file__).parent.parent
sys.path.append(str(root))
from env.grid_scenarios import MiniWorld
# Hypar-parameters that could be helpful.
GAMMA = 0.9
EPSILON = 0.001
BLOCKS = [14, 15, 21, 27]


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n
        #pi是策略，记录每个状态下不同动作的概率，V是状态价值函数，P是状态转移矩阵
        self.pi = np.ones((self.n_state, self.n_action)) / self.n_action
        self.V = np.zeros(self.n_state)
        self.P = self.createP()

    def createP(self):
        # 初始化P矩阵
        P = [[0 for j in range(4)] for i in range(self.env.n_height * self.env.n_width)]
        #print(P)    
        for j in range(self.env.n_height):
            for i in range(self.env.n_width):
                for a in range(4):
                    # 在终点的情况 or 在墙壁的情况
                    if self.env._is_end_state(i,j) or self.env.grids.get_dtype(i, j) == 1:
                        state = self.env._xy_to_state(i, j)
                        reward = self.env.grids.get_reward(i, j)
                        """
                        在这行代码中, j* self.env.n_width + i是当前的状
                        态, a是采取的动作。[(1,state, reward, True)]是
                        一个列表，其中的元素是一个元组，元组的元素是（转移概
                        率，下一个状态，奖励，是否结束）
                        """
                        #j*6+i是当前状态的编号
                        P[j * self.env.n_width + i][a] = [(1, state, reward, True)]
                        continue
                    # 其他情况
                    new_x, new_y = i, j
                    if a == 0:
                        new_x -= 1  # left
                    elif a == 1:
                        new_x += 1  # right
                    elif a == 2:
                        new_y += 1  # up
                    elif a == 3:
                        new_y -= 1  # down
                    # boundary effect
                    if new_x < 0:
                        new_x = 0
                    if new_x >= self.env.n_width:
                        new_x = self.env.n_width - 1
                    if new_y < 0:
                        new_y = 0
                    if new_y >= self.env.n_height:
                        new_y = self.env.n_height - 1
                    if self.env.grids.get_dtype(new_x, new_y) == 1:
                        new_x, new_y = i, j
                    state = self.env._xy_to_state(new_x, new_y)
                    reward = self.env.grids.get_reward(new_x, new_y)
                    done = self.env._is_end_state(new_x, new_y)
                    P[j * self.env.n_width + i][a] = [(1, state, reward, done)]
        #print(P)
        #print(P[0][1])
        return P
    #状态转移矩阵P的含义为，每个状态下的每个动作都有一个概率分布，这个概率分布是一个列表，列表中的每个元素是一个元组，元组的元素是（转移概率，下一个状态，奖励，是否结束）
    def policy_evaluation(self):
        count = 1
        while True:
            delta = 0
            new_v = np.zeros(self.n_state)
            for s in range(self.n_state):
                qsa_list = []  # 计算状态s下的所有Q(s,a)
                for a in range(self.n_action):
                    qsa = 0
                    for res in self.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + GAMMA * self.V[next_state] * (1 - done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                delta = max(delta, np.abs(new_v[s] - self.V[s]))
            self.V = new_v
            if delta < EPSILON:
                break
            count += 1
        print("策略评估进行%d轮后完成" % count)

    def policy_improvement(self):
        for s in range(self.n_state):
            qsa_list = []
            for a in range(self.n_action):
                qsa = 0
                for res in self.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + GAMMA * self.V[next_state] * (1 - done))
                qsa_list.append(qsa)
            #print(qsa_list)
            maxq = max(qsa_list)
            #print(maxq)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            #print(cntq)
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            old_pi = self.pi.copy()  # 将列表进行深拷贝,方便接下来进行比较
            new_pi = self.policy_improvement()
            if (old_pi == new_pi).all():
                print("策略已收敛")
                break

def print_agent(agent, action_meaning):
    print("最终收敛的策略如下所示：")
    for j in range(agent.env.n_height-1,-1,-1):
        for i in range(agent.env.n_width):
            # 一些特殊的状态,例如终点和陷阱
            if agent.env._is_end_state(i,j):
                print('到终点了', end=' ')
            elif agent.env.grids.get_dtype(i, j) == 1:  # 墙
                print('是一面墙', end=' ')
            else:
                a = agent.pi[j * agent.env.n_width + i]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else '无'
                print(pi_str, end=' ')
        print()

if __name__ == "__main__":
    print("采用策略迭代思想：")
    env = MiniWorld()
    action_meaning = ['左', '右','上', '下', ]
    env.reset()
    agent = PolicyIteration(env)
    agent.policy_iteration()
    for i in range(5,-1,-1):
        for j in range(6):
            print("%.4f" %(agent.V[i*6+j]),end='\t')
        print()
    print(agent.V)
    print_agent(agent, action_meaning)
    env.update_r(agent.V)
    env.reset()
    for _ in range(2000):
        env.render()
    env.close()
    print("env closed")
