import sys

from RL_brain import DeepQNetwork
import numpy as np
import math
import subprocess
import json

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
    # stderr_output = result.stderr.decode('GBK')
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


def run_maze_new(rootpath, resource_size, condition, num, total_step):
    step = 0
    observation = np.ones(resource_size, dtype=int)

    for episode in range(1):
        # print("episode: {}".format(episode))
        while True:
            # print("step: {}".format(step))
            action = RL.choose_action(observation)
            observation_, reward, done_coarseGrained, done_pureFineGrained, done_greedyFineGrained = stepGO(observation, action, rootpath, step, condition, num)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 50) and (step % 5 == 0):
                RL.learn()
            observation = observation_
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


if __name__ == '__main__':
    # 通过参数获取资源数量，以及tasks.txt和resources.txt所在的目录位置
    # 读取参数
    resource_size = sys.argv[1]
    resource_size = int(resource_size)
    dirPath = sys.argv[2]
    condition = sys.argv[3]
    num = sys.argv[4]
    step = sys.argv[5]

    n_actions = int(resource_size) * 4
    n_features = int(resource_size)
    RL = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=50,
                      memory_size=2000
                      )
    run_maze_new(dirPath, resource_size, condition, num , step)
