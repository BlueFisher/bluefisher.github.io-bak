---
·title: >-
  SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep
  Reinforcement Learning
mathjax: true
typora-root-url: ..
date: 2020-10-29 16:57:06
categories: Reinforcement Learning
tags:为了 SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning 这篇论文
---

为了解决无模型强化学习中稳定优化非线性值函数估计、避免 Q-learning 中由于 target Q 引起的偏差传递及更有效的探索，*SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning* 这篇论文结合了三种方法：1. 随机初始化一系列不同的智能体；2. 带权重的 Bellman backups；3. 在推断阶段使用上置信来选取动作。主要创新在第一点方法上，相当于是 TD3 算法的扩展。

<!--more-->

论文使用 ensemble 方法，结合多个价值函数与策略函数来解决上述的问题。

## Bootstrap with random initialization

对于 N 个 SAC 智能体，即 $\left\{Q_{\theta_{i}}, \pi_{\phi_{i}}\right\}_{i=1}^{N}$ ，其中 $\theta_i$ 和 $\phi_i$ 表示第 $i$ 个 Q 函数与策略，我们首先随机初始化所有模型的参数，以增加模型的多样性，其次再用不同的样本训练每个智能体，每个智能体 $i$ 在每个时间步 $t$ ，我们从伯努利分布中采样出一个二值 mask $m_{t,i}$ ，并将mask 存放到经验池中。在更新模型参数的时候，将这个 mask 乘以目标函数，即 $m_{t, i} \mathcal{L}_{Q}\left(\tau_{t}, \theta_{i}\right)$ 和 $m_{t, i} \mathcal{L}_{\pi}\left(s_{t}, \phi_{i}\right)$ ，以此达到每个智能体可以训练不同样本的目的。

## Weighted Bellman backup

因为 target Q-function $Q_{\bar{\theta}}\left(s_{t+1}, a_{t+1}\right)$ 有一定的偏差，而这个偏差又会传递到 $Q_\theta(s_t,a_t)$ 的更新中。为了缓解这个问题，对于每个智能体 $i$ ，使用 weighted Bellman backup：
$$
\begin{array}{l}
\mathcal{L}_{W Q}\left(\tau_{t}, \theta_{i}\right) \\
=w\left(s_{t+1}, a_{t+1}\right)\left(Q_{\theta_{i}}\left(s_{t}, a_{t}\right)-r_{t}-\gamma\left(Q_{\bar{\theta}_{i}}\left(s_{t+1}, a_{t+1}\right)-\alpha \log \pi_{\phi}\left(a_{t+1} \mid s_{t+1}\right)\right)\right)^{2}
\end{array}
$$
其中 $\tau_{t}=\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$ 是一个 transition，$a_{t+1} \sim \pi_\phi (a|s_t)$ ，$w(s,a)$ 是一个根据 ensemble target Q-functions 产生的置信权重：
$$
w(s, a)=\sigma\left(-\bar{Q}_{\operatorname{std}}(s, a) * T\right)+0.5
$$
其中 $T>0$ 是一个温度参数，$\sigma$ 时 sigmoid 函数，$\bar{Q}_{\operatorname{std}}(s, a)$ 是所有 target Q-functions 的标准差。注意这个置信权重是在 $[0.5,1.0]$ 范围内的。

## UCB exploration

在高效探索方面，ensemble 方法同样可以起到一定的作用。在选取动作的时候，使用如下方法：
$$
a_{t}=\max _{a}\left\{Q_{\operatorname{mean}}\left(s_{t}, a\right)+\lambda Q_{\operatorname{std}}\left(s_{t}, a\right)\right\}
$$
其中 $\lambda$ 是一个超参数。上述公式可以应用于离散动作空间，对于连续动作空间，不能直接最大化 UCB，作者使用了一种简单的估计方法：先从 ensemble 策略中生成 N 个候选动作，再按照上述公式选择能够最大 UCB 的那个动作。

SUNRISE 应用于 SAC 的算法如下：

![](/images/2020-10-29-SUNRISE-A-Simple-Unified-Framework-for-Ensemble-Learning-in-Deep-Reinforcement-Learning/image-20201031110619152.png)

# 参考

Lee, K., Laskin, M., Srinivas, A., & Abbeel, P. (2020). Sunrise: A simple unified framework for ensemble learning in deep reinforcement learning. *arXiv preprint arXiv:2007.04938*.