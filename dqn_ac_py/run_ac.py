import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
env = gym.make('CartPole-v1').unwrapped
# state_number = env.observation_space.shape[0]
# action_number = env.action_space.n
action_number = 8 * 6
state_number = 8
LR_A = 0.005  # learning rate for actor
LR_C = 0.01  # learning rate for critic
Gamma = 0.9
Switch = 0  # 训练、测试切换标志
'''AC第一部分 设计actor'''
'''第一步.设计actor和critic的网络部分'''


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.in_to_y1 = nn.Linear(state_number, 50)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(50, 20)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(20, action_number)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, inputstate):
        inputstate = self.in_to_y1(inputstate)
        inputstate = F.relu(inputstate)
        inputstate = self.y1_to_y2(inputstate)
        inputstate = torch.sigmoid(inputstate)
        act = self.out(inputstate)
        return F.softmax(act, dim=-1)


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.in_to_y1 = nn.Linear(state_number, 40)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(40, 20)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(20, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, inputstate):
        inputstate = self.in_to_y1(inputstate)
        inputstate = F.relu(inputstate)
        inputstate = self.y1_to_y2(inputstate)
        inputstate = torch.sigmoid(inputstate)
        act = self.out(inputstate)
        return act


class Actor():
    def __init__(self):
        self.actor = ActorNet()
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_A)

    '''第二步.编写actor的选择动作函数'''

    def choose(self, inputstate):
        inputstate = torch.FloatTensor(inputstate)
        probs = self.actor(inputstate).detach().numpy()
        action = np.random.choice(np.arange(action_number), p=probs)
        return action

    '''第四步.根据td-error进行学习，编写公式log(p(s,a))*td_e的代码'''

    def learn(self, s, a, td):
        s = torch.FloatTensor(s)
        prob = self.actor(s)
        log_prob = torch.log(prob)
        actor_loss = -log_prob[a] * td
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()


'''第二部分 Critic部分'''


class Critic():
    def __init__(self):
        self.critic = CriticNet()
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_C)
        self.lossfunc = nn.MSELoss()  # 均方误差（MSE）

    '''第三步.编写td-error的计算代码（V现实减去V估计就是td-error）'''

    def learn(self, s, r, s_):
        '''当前的状态s计算当前的价值，下一个状态s_计算出下一状态的价值v_，然后v_乘以衰减γ再加上r就是v现实'''
        s = torch.FloatTensor(s)
        v = self.critic(s)  # 输入当前状态，有网络得到估计v
        r = torch.FloatTensor([r])  # .unsqueeze(0)#unsqueeze(0)在第一维度增加一个维度
        s_ = torch.FloatTensor(s_)
        reality_v = r + Gamma * self.critic(s_).detach()  # 现实v
        td_e = self.lossfunc(reality_v, v)
        self.optimizer.zero_grad()
        td_e.backward()
        self.optimizer.step()
        advantage = (reality_v - v).detach()
        return advantage  # pytorch框架独有的毛病：返回一定要用reality_v-v，但是误差反向传递一定要用td_e，不然误差传不了，不能收敛


compilation_path = r"D:\Project\HighPerformanceResourceSharingProtocol\HighPerformanceResourceSharingProtocol\target\classes;C:\Users\Administrator\.m2\repository\com\google\code\gson\gson\2.10.1\gson-2.10.1.jar"
compilation_file_name = "sysu.rtsg.analysis.ResponseTimeAnalysisWithVariousPrioritiesForRL"


def stepGO(observation, action, rootpath, step, condition, num):
    done_coarseGrained = False
    done_pureFineGrained = False
    done_greedyFineGrained = False

    s_ = observation.copy()
    index = math.floor(action / 4)
    base = s_[index] + (action % 4 + 1)
    if base <= 5:
        s_[index] = base
    else:
        s_[index] = base % 5
    # print(str(s_))

    # 计算可调度性结果并获取奖励函数
    result = subprocess.run(
        ["java", "-cp", compilation_path, compilation_file_name, rootpath, np.array2string(s_), str(step), str(condition), str(num)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 获取并打印 stdout 输出（将其从字节转换为字符串）
    stdout_output = result.stdout.decode('GBK')
    dict_result = json.loads(stdout_output)

    reward = 0
    if dict_result['is_system_schedulable'] is True:
        reward = 10000000
        done_coarseGrained = True
        # print("reward:" + str(reward))
    elif dict_result['is_pureFineGrained_system_schedulable'] is True:
        reward = 10000000
        done_pureFineGrained = True
        # print("reward:" + str(reward))
    elif dict_result['is_greedyFineGrained_system_schedulable'] is True:
        reward = 10000000
        done_greedyFineGrained = True
        # print("reward:" + str(reward))
    else:
        re_list = dict_result['unschedulable_task_D_R']
        for item in re_list:
            Ri = item[0]
            Di = item[1]
            reward += Ri / (Ri - Di)
        # done_coarseGrained = False
        # done_pureFineGrained = False
        # done_greedyFineGrained = False
        # print("reward:" + str(reward))
    return s_, reward, done_coarseGrained, done_pureFineGrained, done_greedyFineGrained


def main_process(dirPath, resource_size, condition, num, total_step):
    step = 0
    for i in range(1):
        step = 0
        r_totle = []
        # observation, _ = env.reset()  # 环境重置
        # 状态初始化
        observation = np.ones(resource_size, dtype=int)

        while True:
            # print("Step: " + str(step))
            step = step + 1
            action = a.choose(observation)
            # observation_, reward, done, some_bool, info = env.step(action)
            observation_, reward, done_coarseGrained, done_pureFineGrained, done_greedyFineGrained = stepGO(observation, action, dirPath, step, condition, num)
            # if done: reward = -50  # 稍稍修改奖励，让其收敛更快
            td_error = c.learn(observation, reward, observation_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            a.learn(observation, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            observation = observation_
            r_totle.append(reward)
            if done_coarseGrained:
                print(1)
                break
            if done_pureFineGrained:
                print(2)
                break
            if done_greedyFineGrained:
                print(3)
                break
            step += 1
            if step == int(total_step):
                print(0)
                break
    # r_sum = sum(r_totle)
    # print("\r回合数：{} 奖励：{}".format(i, r_sum), end=" ")


if __name__ == '__main__':
    # 通过参数获取资源数量，以及tasks.txt和resources.txt所在的目录位置
    # 读取参数
    resource_size = sys.argv[1]
    resource_size = int(resource_size)
    dirPath = sys.argv[2]
    condition = sys.argv[3]
    num = sys.argv[4]
    step = sys.argv[5]
    # resource_size = 16
    # dirPath = "D:\\Project\\HighPerformanceResourceSharingProtocol\\HighPerformanceResourceSharingProtocol\\scripts"
    # condition = 3
    # num = 2
    # step = 1000



    action_number = int(resource_size) * 4
    state_number = int(resource_size)

    '''训练'''
    if Switch == 0:
        a = Actor()
        c = Critic()
        main_process(dirPath, resource_size, condition, num, step)
        # if i % 50 == 0 and i > 300:  # 保存神经网络参数
        #     save_data = {'net': a.actor.state_dict(), 'opt': a.optimizer.state_dict(), 'i': i}
        #     torch.save(save_data, "D:\model_actor.pth")
        #     save_data = {'net': c.critic.state_dict(), 'opt': c.optimizer.state_dict(), 'i': i}
        #     torch.save(save_data, "D:\model_critic.pth")

    else:
        print('AC测试中...')
        aa = Actor()
        cc = Critic()
        # checkpoint_aa = torch.load("D:\model_actor.pth")
        # aa.actor.load_state_dict(checkpoint_aa['net'])
        # checkpoint_cc = torch.load("D:\model_critic.pth")
        # cc.critic.load_state_dict(checkpoint_cc['net'])
        for j in range(10):
            state = env.reset()
            total_rewards = 0
            while True:
                env.render()
                state = torch.FloatTensor(state)
                action = aa.choose(state)
                new_state, reward, done, info = env.step(action)  # 执行动作
                total_rewards += reward
                if done:
                    print("Score", total_rewards)
                    break
                state = new_state
        env.close()
