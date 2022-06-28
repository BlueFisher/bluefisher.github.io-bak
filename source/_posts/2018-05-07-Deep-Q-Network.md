---
title: Deep Q-Network
mathjax: true
date: 2018-05-07 21:22:01
categories:
- Reinforcement Learning
tags:
- RL
- DQN
typora-root-url: ..
---

 **Deep Q-Network (DQN)** 算法在 [Q-Learning](https://bluefisher.github.io/2018/05/22/%E6%97%A0%E6%A8%A1%E5%9E%8B%E6%8E%A7%E5%88%B6-Model-Free-Control/)  算法的基础上，利用 **深度卷积神经网络来逼近值函数**，将强化学习与深度学习相结合，估计出最优行为价值函数 (optimal action-value function)



$$
Q^*(s,a) = \max_\pi \mathbb{E}[G_t|s_t=s, a_t=a, \pi]
$$

然而通过深度学习的视角来进行强化学习面临着许多问题，一个是深度学习程序需要大量的人为标记的数据，但强化学习通常需要在大量稀疏的、延迟的、带有噪声的奖励数值中学习。另一个问题是大多数深度学习算法都需要假定数据样本之间是独立的，但强化学习的数据是一连串高度关联的序列。

为了解决这些问题，DQN 算法利用了 **经验回放机制** 和 **设置目标网络与评估网络** 。

有两篇论文提出了 DQN 算法，分别是：

> 1. *Mnih V, Kavukcuoglu K, Silver D, et al. Playing Atari with Deep Reinforcement Learning[J]. Computer Science, 2013.* 
> 2. *Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540):529.* 

后一篇论文比前一篇多了目标、评估网络的特性。

<!--more-->

# Q-Learning

无模型控制 (Model-Free Control) 算法 Q-Learning 即为遵循异策略 (off-policy) 的 Sarsa 算法，下一个行为的选择遵循行为策略 (behavior policy) $A_{t+1} \sim \mu (\cdot|S_t)$，而另一种继任的行为的选择遵循目标策略 (target policy) $A' \sim \pi (\cdot|S_t)$，以此得到更新过程：
$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A') - Q(S_t, A_t))
$$
其中 target policy 对于 $Q(s,a)$ 采用贪婪策略
$$
\DeclareMathOperator*{\argmax}{argmax}
\pi(S_{t+1}) = \argmax_{a'} Q(S_{t+1},a')
$$
behavior pllicy对于 $Q(s,a)$ 采用 *ϵ-greddy* 策略，所以 Q-Learning 的目标函数简化为：
$$
\begin{align*}
R_{t+1} + \gamma Q(S_{t+1}, A') &= R_{t+1} + \gamma Q(S_{t+1}, \argmax_{a'}Q(S_{t+1},a')) \\
&= R_{t+1} + \max_{a'}\gamma Q(S_{t+1},a')
\end{align*}
$$
所以 Q-Learning 控制算法为：
$$
Q(S,A) \leftarrow Q(S,A) + \alpha \left(  R + \gamma \max_{a'}Q(S', a') - Q(S,A)  \right)
$$
伪代码：

![](/images/2018-05-07-Deep-Q-Network/q-learning.png)

# DQN

## 值函数近似 Value Function Approximation

行为价值函数的近似，即使用：
$$
\hat{q}(S,A,w) \approx q_\pi(S,A)
$$
最小化近似行为函数$\hat{q}(S,A,w)$ 和真实值 $q_\pi(S,A)$ 的均方误差：
$$
J(w) = \mathbb{E}_\pi [(q_\pi(S,A) - \hat{q}(S,A,w))^2]
$$
而对于一个 Q-network，在论文中记为：
$$
L_i(\theta_i) = \mathbb{E}_{s,a,r}\left[ \big(y_i-Q(s,a; \theta_i)\big)^2 \right]
$$
其中 $y_i=\mathbb{E}_{s'}[r+\gamma\max_{a'}Q(s',a';\theta_{i-1})]$，所以对于参数 $\theta_i$ 的梯度为：
$$
\nabla_{\theta_i}L_i(\theta_i)=\mathbb{E}_{s,a,r,s'} \left[ \left(r+\gamma\max_{a'}Q(s',a';\theta_{i-1}) -Q(s,a;\theta_i) \right) \nabla_{\theta_i}Q(s,a;\theta_i)  \right]
$$

## 经验回放 Experience Replay

直接从连续的样本中学习效率很低，因为前后样本之间存在着很强的关联性，然而随机抽样则可以打破这些关联以此来减小训练的方差。另外对于同策略学习，现有的参数决定着参数更新所需要的下一个样本，这样就很容易产生一些不想要的事件循环，使得参数更新困在局部最优点甚至无法收敛，但经验回放可以使下一步的行为分布基于之前许许多多的状态而平均化，避免学习结果震荡或发散

保存每一步的“经验” $e_t=(s_t,a_t,r_t,s_{t+1})$ 于集合 $\mathcal{D}_t = \{e_1, \cdots, e_t\}$ 中，每一次更新学习中，都从经验集合里进行随机抽样取出一定数量的样本进行 mini-batch 训练。在实际开发中，只需要在集合中存储最近的 $N​$ 次经验就可以，像队列一样如果集合满了就先进先出，以此可以得到第一篇论文中的算法伪代码：

![](/images/2018-05-07-Deep-Q-Network/dqn.png)

## 目标神经网络 Target Network

为了更进一步增强运用了神经网络的 Q-Learning 方法的稳定性，第二篇论文中提出了用一个分离的神经网络 $\hat{Q}$ 来生成目标 $y_i$ ，成为目标网络 (target network)。在每一次的迭代过程中，将 $r+\gamma\max_{a'} Q^*(s',a')$ 用旧参数 $\theta^-$固定下来 $r+\gamma\max_{a'} \hat{Q}(s',a';\theta_i^-)$ ，经过 $C$ 趟训练之后，将原有更新过后的网络 $Q$ 的参数复制给目标网络 $\hat Q$ ，修改后的伪代码为：

![](/images/2018-05-07-Deep-Q-Network/C6ypxs.png)

# 参考

Sutton, R. & Barto, A. Reinforcement Learning: An Introduction (MIT Press, 1998).

Mnih V, Kavukcuoglu K, Silver D, et al. Playing Atari with Deep Reinforcement Learning[J]. Computer Science, 2013. 

Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540):529. 

<https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-1-A-DQN/>

<https://zhuanlan.zhihu.com/p/28108498>

<https://zhuanlan.zhihu.com/p/28223841>