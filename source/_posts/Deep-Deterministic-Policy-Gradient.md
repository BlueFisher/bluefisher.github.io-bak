---
title: Deep Deterministic Policy Gradient
mathjax: true
date: 2018-05-16 19:42:28
updated: 2018-05-16 19:42:28
categories: Reinforcement Learning
tags:
- RL
- PG
---

Deep Deterministic Policy Gradient (DDPG) 算法出自论文 *Continuous control with deep reinforcement learning* ，该算法的思想很简单，就是将 [确定性策略梯度 Deterministic Policy Gradient](https://bluefisher.github.io/2018/05/16/%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6-Deterministic-Policy-Gradient/) 与 [Deep Q-Network](https://bluefisher.github.io/2018/05/07/Deep-Q-Network/) 两个算法结合起来，解决了 DQN 只能运用在离散行为空间上的局限，同时借鉴 DQN 的神经网络、经验回放和设置 target 网络使  DPG 中的 Actor-Critic 算法更容易收敛

<!--more-->

# Critic

对于行为价值函数，我们已经使用过非常多次了，即为：
$$
Q^\pi(s_t,a_t) = \mathbb{E}_{s \sim E, a \sim \pi} [R_t|s_t,a_t]
$$
用贝尔曼公式表达为：
$$
Q^\pi(s_t,a_t) = \mathbb{E}_{s_{t+1} \sim E} \big[ r(s_t,a_t) + \gamma \mathbb{E}_{a_{t+1} \sim \pi}[ Q^\pi(s_{t+1},a_{t+1}) ] \big]
$$
如果目标策略为固定策略 $\mu$ 而不是随机策略，则可以去掉内层期望：
$$
Q^\mu(s_t,a_t) = \mathbb{E}_{s_{t+1} \sim E} \big[r(s_t,a_t) + \gamma Q^\mu(s_{t+1},\mu(s_{t+1}))\big]
$$
该公式的期望只与环境有关，所以可以用异策略来学习 $Q^\mu$ ，也就是说用一个不同的随即策略 $\beta$ 来生成状态行为轨迹。

在 Q-Learning 中，确定性策略为贪婪策略 $\mu(s)=\arg\max_a Q(s,a)$ 。而对于以 $\theta^Q$ 为参数的近似函数，最小化以下代价函数：
$$
L(\theta^Q) = \mathbb{E}_{s_t\sim\rho^\beta, a_t\sim\beta} \left[ ( Q(s_t,a_t|\theta^Q) - y_t )^2 \right]
$$
其中：
$$
y_t = r(s_t,a_t)+\gamma Q(s_{t+1},\mu(s_{t+1}| \theta^Q))
$$
尽管这里的 $y_t$ 也依赖于 $\theta^Q$ ，但可以忽略。

一般来说，尽管用大型的、非线性的近似函数来学习行为价值函数会变得非常不稳定，但在这用到了 DQN 中经验回放、设置独立的 target 网络来计算 $y_t$ 让算法稳定。

