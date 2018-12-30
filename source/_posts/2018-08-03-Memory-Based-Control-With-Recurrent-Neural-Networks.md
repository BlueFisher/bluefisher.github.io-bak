---
title: Memory-Based Control With Recurrent Neural Networks
mathjax: true
date: 2018-08-03 10:18:24
categories:
- Reinforcement Learning
tags:
- RL
typora-root-url: ..
---

部分可观察环境的马尔可夫决策过程 (partially-observed Markov Decision process POMDP) 是强化学习中的一个非常具有挑战性的部分，在 *Memory-Based Control With Recurrent Neural Networks* 这篇论文中，作者使用了长短时记忆的循环神经网络并扩展了确定性策略梯度 (deterministic policy gradient) 与随机价值梯度 (stochastic value gradient) 两种方法分别称为 RDPG 和 RSVG(0) 来解决 POMDP 问题。

<!--more-->

# POMDP

POMDP 可以被描述为一系列环境状态 $\mathcal{S}$ 和行为动作 $\mathcal{A}$ ，初始状态的概略分布 $p_0(s_0)$ ，状态转移概率 $p(s_{t+1}|s_t,a_t)$ 和价值函数 $r(s_t,a_t)$ 。部分可观察指的是智能体没办法直接观察到完整的状态信息 $s_t$ ，取而代之的只能根据条件概率 $p(o_t|s_t)$ 从 $\mathcal{O}$ 中接受观察到的信息。因此一个智能体能获得的完整的经验回合为 $h_t=(o_1,a_1,o_2,a_2,\cdots,a_{t-1},o_t)$

我们的总目标很简单，就是要去学习一个策略 $\pi(h_t)$ 来最大化衰减过的价值函数。对于随即策略来说，即最大化
$$
J=\mathbb{E}_\tau\left[ \sum_{t=1}^\infty \gamma^{t-1}r(s_t,a_t) \right]
$$
其中轨迹 $\tau=(s_1,o_1,a_1,s_2, \cdots)$ 根据概率分布 $\pi:\ p(s_1)p(o_1|s_1)\pi(a_1|h_1)p(s_2|s_1,a_1)p(o_2|s_2)\pi(a_2|h_2)\cdots$ 产生。对于确定性策略来说，我们用确定性函数 $a_t=\mu(h_t)$ 来代替 $a_t\sim\pi(\cdot|h_t)$ 

 # Recurrent DPG

首先来看确定性策略梯度算法，我们用以 $\theta$ 为参数的 $\mu^\theta$ 表示确定性策略，$Q^\mu$ 表示当前策略的行为价值函数，策略可以用如下方法来更新：
$$
\frac{\partial J(\theta)}{\partial \theta}=\mathbb{E}_{s\sim \rho^\mu}\left[ \frac{\partial Q^\mu(s,a)}{\partial a}\bigg|_{a=\mu^\theta(s)} \frac{\partial \mu^\theta(s)}{\partial \theta} \right]
$$
在部分可观察的情况下，最优策略还有行为价值函数都是关于之前完整经历 $h_t$ 的函数，这是就要用到循环神经网络而不是简单的前馈网络 (feedforward networks) ，我们要将 $\mu(s)$ 和 $Q(s,a)$ 改写成 $\mu(h)$ 和 $Q(h,a)$ 的形式：
$$
\frac{\partial J(\theta)}{\partial \theta}=\mathbb{E}_{\tau}\left[ \sum_t \gamma^{t-1} \frac{\partial Q^\mu(h_t,a)}{\partial a}\bigg|_{a=\mu^\theta(h_t)} \frac{\partial \mu^\theta(h_t)}{\partial \theta} \right]
$$
在实际代码中，通常忽略此处的衰减值 $\gamma^t$ 这一项，$Q^\mu$ 会被替换为可学习的估计值 $Q^w$ ，这也是由参数 $w$ 构成的循环神经网络组建而来。因此，相较于直接考虑整个观察到的经历，我们可以用随时间向后传播 (backpropagation through time BPTT) 的方法来高效的训练循环神经网络。

除此之外，我们仿照 DDPG 中的方法，设置了目标网络，用参数 $\theta'$ 和 $w'$ 表示，并且使用软更新 (soft update) 的方法更新两个网络。具体算法如下：

![](/images/2018-08-03-Memory-Based-Control-With-Recurrent-Neural-Networks/PBEFYQ.md.png)

首先初始化四个网络，每次迭代过程中，先完整地记录一次经历存储到缓冲 $R$ 中，在选择行为时在使用确定性策略算法的基础上增加一点探索的噪声。接着从缓冲中采样 N 个经历构造 $h_t^i$ ，$i$ 代表第 $i$ 个 minibatch ，$t$ 代表每个经历的第 $t$ 个时刻。然后分别更新行为值函数 critic 和策略 actor ，最后使用软更新来更新目标神经网络。

# 参考

Heess, N., Hunt, J. J., Lillicrap, T. P., & Silver, D. (2015). Memory-based control with recurrent neural networks. *arXiv preprint arXiv:1512.04455*. 