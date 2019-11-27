---
title: Python 与 Unity mlagents 交互 API
mathjax: false
date: 2018-12-29 21:07:54
categories: Python
tags: python
---

本文基于 [ml-agents](https://github.com/Unity-Technologies/ml-agents) 中与 python 交互的文档  *[Unity ML-Agents Python Interface and Trainers](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md)* ，更加详细地介绍整个交互 API

<!--more-->

# 初始化 unity 环境


```python
import numpy as np
import matplotlib.pyplot as plt
from mlagents.envs import UnityEnvironment

%matplotlib inline
```

初始化环境
`env = UnityEnvironment(file_name="3DBall", worker_id=0, seed=1)`

`file_name` 是 unity 编译生成的二进制可执行环境。

`worker_id` 表示你想用哪个端口来与环境进行交互，可以用来进行并行交互，比如 A3C 。

`seed` 为训练过程中的随机种子，如果想让 untiy 环境每次随机的效果相同，则种子设置为固定值。

若 `file_name=None` 则表示 python 直接与 unity editor 进行交互，等待编辑器中的开始按钮被按下 *Start training by pressing the Play button in the Unity Editor* ，同时这个时候 `worker_id` 必须为 0 才能与编辑器进行连接。


```python
env = UnityEnvironment()
```

    INFO:mlagents.envs:Start training by pressing the Play button in the Unity Editor.
    INFO:mlagents.envs:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 2
            Number of Training Brains : 2
            Reset Parameters :
    		copy -> 1.0
    Unity brain name: Brain1
            Number of Visual Observations (per agent): 2
            Vector Observation space size (per agent): 8
            Number of stacked Vector Observation: 1
            Vector Action space type: continuous
            Vector Action space size (per agent): [2]
            Vector Action descriptions: 3, 3
    Unity brain name: Brain2
            Number of Visual Observations (per agent): 0
            Vector Observation space size (per agent): 8
            Number of stacked Vector Observation: 1
            Vector Action space type: continuous
            Vector Action space size (per agent): [2]
            Vector Action descriptions: 3, 3


# 获取基础 brain 信息


```python
print(env.brains)
print(env.brain_names)

for name in env.brain_names:
    params = env.brains[name]
    print(params)
    print('brain_name: ', end='')
    print(params.brain_name)

    print('num_stacked_vector_observations: ', end='')  # 向量状态栈大小
    print(params.num_stacked_vector_observations)
    print('vector_observation_space_size: ', end='')
    print(params.vector_observation_space_size)  # 向量状态空间

    print('number_visual_observations: ', end='')
    print(params.number_visual_observations)  # 图像状态数量
    print('camera_resolutions: ', end='')
    print(params.camera_resolutions)  # 图像的分辨率

    print('vector_action_space_type: ', end='')
    print(params.vector_action_space_type)  # 行为空间类型（离散或连续）
    print('vector_action_space_size: ', end='')
    print(params.vector_action_space_size)  # 行为空间
    print('vector_action_descriptions: ', end='')
    print(params.vector_action_descriptions)  # 行为描述

    print('---------')
```

    {'Brain1': <mlagents.envs.brain.BrainParameters object at 0x0000023EC4BFB0F0>, 'Brain2': <mlagents.envs.brain.BrainParameters object at 0x0000023EC5327470>}
    ['Brain1', 'Brain2']
    Unity brain name: Brain1
            Number of Visual Observations (per agent): 2
            Vector Observation space size (per agent): 8
            Number of stacked Vector Observation: 1
            Vector Action space type: continuous
            Vector Action space size (per agent): [2]
            Vector Action descriptions: 3, 3
    brain_name: Brain1
    num_stacked_vector_observations: 1
    vector_observation_space_size: 8
    number_visual_observations: 2
    camera_resolutions: [{'height': 450, 'width': 500, 'blackAndWhite': False}, {'height': 550, 'width': 600, 'blackAndWhite': False}]
    vector_action_space_type: continuous
    vector_action_space_size: [2]
    vector_action_descriptions: ['3', '3']
    ---------
    Unity brain name: Brain2
            Number of Visual Observations (per agent): 0
            Vector Observation space size (per agent): 8
            Number of stacked Vector Observation: 1
            Vector Action space type: continuous
            Vector Action space size (per agent): [2]
            Vector Action descriptions: 3, 3
    brain_name: Brain2
    num_stacked_vector_observations: 1
    vector_observation_space_size: 8
    number_visual_observations: 0
    camera_resolutions: []
    vector_action_space_type: continuous
    vector_action_space_size: [2]
    vector_action_descriptions: ['3', '3']
    ---------


# 重置训练环境，开始交互

`train_mode=True` 表示是训练模式，即用的是 *Acadmey* 中的 *Training Configuration* ， `train_mode=False` 为推断模式，即用的是 *Acadmey* 中的 *Inference Configuration* ，unity 中会将每帧都绘制

`config={}` 是 reset 环境时的参数，类型为 `dict` 。需要预先在 unity editor 中定义所有的参数。

返回为一个 `dict` ，包含所有 `brain` 的信息


```python
brain_infos = env.reset(train_mode=True, config={
    'copy': 1
})
```

    INFO:mlagents.envs:Academy reset with parameters: copy -> 1

`vector_observations` 是一个 `numpy` ，维度为 `(智能体数量, 向量状态的长度 * 向量状态栈大小)` ，其中向量状态栈代表有多少状态被储存起来一起作为当前的状态

`visual_observations` 是一个 `list` ，个数为 `Brain` 中设置的摄像机的个数，其中每一个元素为也为一个 `list` ，长度为智能体数量。该 `list` 中的元素为一个三维的 `numpy` ，维度为 `(长，宽，通道数)` 。如每个 `Brain` 要设置左右两台观测的摄像机，则 `visual_observations` 长度为 2 ，分别代表左右两个摄像机。第一个元素为所有智能体的左摄像机的图像集合，第二个元素为所有只能提的右摄像机的图像集合。


```python
for name in env.brain_names:
    params = env.brains[name]
    info = brain_infos[name]
    print(info)
    print('vector_observations: ', end='')  # 向量状态 numpy
    print(info.vector_observations.shape)
    print('visual_observations: ')  # 图像状态 list
    for i, obs_per_camera in enumerate(info.visual_observations):
        print('\t', 'number of agents', len(obs_per_camera))
        for j, ob in enumerate(obs_per_camera):
            print(ob.shape)
            if params.camera_resolutions[i]['blackAndWhite']:
                plt.imshow(ob[:, :, 0], cmap=plt.cm.gray)
            else:
                plt.imshow(ob)
            plt.show()
    print('text_observations: ', end='')  # 文字状态 list
    print(info.text_observations)
    print('rewards: ', end='')  # 奖励 list
    print(info.rewards)
    print('local_done: ', end='')  # 智能体回合是否结束 list
    print(info.local_done)
    print('max_reached: ', end='')  # 智能体回合是否到达最大步数（如果达到最大步数，无论智能体回合是否结束，local_done 也为 True） list
    print(info.max_reached)
    print('previous_vector_actions: ', end='')  # 上一个向量行为 numpy
    print(info.previous_vector_actions.shape)
    print('agents: ', end='')  # 所有智能体 id list
    print(info.agents)
    print('---------')
```

    <mlagents.envs.brain.BrainInfo object at 0x0000020F88ED0FD0>
    vector_observations: (2, 8)
    visual_observations: 
    	 number of agents 2
    	 (450, 500, 3)
    	 (450, 500, 3)
    	 number of agents 2
    	 (550, 600, 3)
    	 (550, 600, 3)
    text_observations: ['', '']
    rewards: [0.0, 0.0]
    local_done: [False, False]
    max_reached: [False, False]
    previous_vector_actions: (2, 2)
    agents: [14714, 14736]
    ---------
    <mlagents.envs.brain.BrainInfo object at 0x0000020F88F69978>
    vector_observations: (1, 8)
    visual_observations: 
    text_observations: ['']
    rewards: [0.0]
    local_done: [False]
    max_reached: [False]
    previous_vector_actions: (1, 2)
    agents: [14658]
    ---------


## 一个最简单的交互方式


```python
env.reset(train_mode=False, config={
    'copy': 1
})
for i in range(200):
    env.step({
        'Brain1': np.random.randn(2, 2),
        'Brain2': np.random.randn(1, 2)
    })
```

    INFO:mlagents.envs:Academy reset with parameters: copy -> 1


## 一个单 brain 的复杂例子（不能直接执行，只是演示）


```python
def simulate(brain_info):
    steps_n = 0
    dones = [False] * len(brain_info.agents)
    trans_all = [[] for _ in range(len(brain_info.agents))]
    rewards_sum = [0] * len(brain_info.agents)
    states = brain_info.vector_observations

    while False in dones and not env.global_done:
        actions = ppo.choose_action(states)
        brain_info = env.step({
            default_brain_name: actions
        })[default_brain_name]
        rewards = brain_info.rewards
        local_dones = brain_info.local_done
        max_reached = brain_info.max_reached
        states_ = brain_info.vector_observations

        for i in range(len(brain_info.agents)):
            trans_all[i].append([states[i],
                                 actions[i],
                                 np.array([rewards[i]]),
                                 local_dones[i],
                                 max_reached[i]])

            if not dones[i]:
                rewards_sum[i] += rewards[i]

            dones[i] = dones[i] or local_dones[i]

        steps_n += 1
        states = states_

    return brain_info, trans_all, rewards_sum


brain_info = env.reset(train_mode=False)[default_brain_name]
for iteration in range(ITER_MAX):
    if env.global_done:
        brain_info = env.reset(train_mode=train_mode)[default_brain_name]
    brain_info, trans_all, rewards_sum = simulate(brain_info)
    mean_reward = sum(rewards_sum) / len(rewards_sum)
```

# 踩过的坑

如果直接给 `step` 传入 `dict` 的话，尽管每个 `brain` 的 action 都是 `numpy` 类型，但在执行完毕后会变为 `list`


```python
env.reset(train_mode=True)
actions = {
    'Brain1': np.random.randn(2, 2),
    'Brain2': np.random.randn(1, 2)
}
print(type(actions['Brain1']), type(actions['Brain2']))
brain_info = env.step(actions)
print(type(actions['Brain1']), type(actions['Brain2']))
```

    <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    <class 'list'> <class 'list'>


# 关闭连接


```python
env.close()
```
