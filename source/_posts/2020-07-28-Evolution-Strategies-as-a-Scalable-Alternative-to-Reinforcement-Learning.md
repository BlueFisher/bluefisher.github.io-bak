---
title: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
mathjax: true
typora-root-url: ..
date: 2020-07-28 17:05:22
categories: Reinforcement Learning
tags: 
- rl
- es
---

[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864) 将 natural evolution strategies (NES) 替代深度强化学习，通过简单标量数据传递，就可以利用分布式集群的多 CPU 性能，快速扩展训练规模、快速训练智能体。

<!--more-->

论文中介绍的 ES 属于 NES 一类的进化算法。用 $F$ 来表示按照参数 $\theta$ 执行的目标函数，NES 算法用参数 $\theta$ 的分布来表示种群 (population) ，这个分布使用 $\psi$ 来参数化，并且通过随机梯度上升来找到使平均目标价值 $\mathbb{E}_{\theta} \sim p_{\psi} F(\theta)$ 最大化的参数 $\psi$ 。用 REINFORCE 算法来优化可得关于 $\psi$ 的梯度：
$$
\nabla_{\psi} \mathbb{E}_{\theta \sim p_{\psi}} F(\theta)=\mathbb{E}_{\theta \sim p_{\psi}}\left\{F(\theta) \nabla_{\psi} \log p_{\psi}(\theta)\right\}
$$
对于强化学习的任务，这里的 $F(\cdot)$ 就是环境所反馈的随机回报，$\theta$ 是随机或确定性策略 $\pi_\theta$ 的参数。论文中一个独立的多元高斯分布来表示种群 $p_\psi$ ，这个高斯分布由均值 $\psi$ 和固定的协方差 $\sigma^2I$ 来表示。这样 $\mathbb{E}_{\theta} \sim p_{\psi} F(\theta)$ 就可以写成只与均值有关的形式：
$$
\mathbb{E}_{\theta \sim p_{\psi}} F(\theta)=\mathbb{E}_{\epsilon \sim N(0, I)} F(\psi+\sigma \epsilon)
$$
即 $\theta = \psi + \sigma\epsilon$ ，那么关于 $\theta$ 的梯度可以写为：
$$
\begin{align}
& \nabla_\psi \mathbb{E}_{\epsilon \sim N(0, I)} F(\psi+\sigma \epsilon) \\
=& \nabla_\psi\int_\epsilon p(\epsilon) F d\epsilon &&(F\text{ 为标量，不传梯度}) \\
=& \int_\epsilon \nabla_\epsilon p(\epsilon) \nabla_\psi\epsilon Fd\epsilon \\
=& \int_\epsilon p(\epsilon) \nabla_\epsilon \log p(\epsilon) \nabla_\psi\epsilon Fd\epsilon && (\text{REINFORCE trick})\\
=& \mathbb{E}_{\epsilon \sim N(0, I)}\nabla_\epsilon\left(-\frac{1}{2}\epsilon^T\epsilon\right) \nabla_\psi\left(\frac{\theta-\psi}{\sigma}\right)Fd\epsilon && p(\epsilon)=(2\pi)^{-\frac{n}{2}}\exp\left(-\frac{\epsilon^T\epsilon}{2}\right) \\
=& \frac{1}{\sigma} \mathbb{E}_{\epsilon \sim N(0, I)}\{F(\psi+\sigma \epsilon) \epsilon\}
\end{align}
$$
这样就可以直接用随机梯度上升来优化 $\psi$。整个算法可以总结为：

![](/images/2020-07-28-Evolution-Strategies-as-a-Scalable-Alternative-to-Reinforcement-Learning/image-20200728205016411.png)

这里的均值参数用 $\theta$ 表示。算法主要是不断重复两个过程：

1. 随机扰动策略的参数，并在环境中执行一个 episode 并评估收集回报；
2. 将所有的 episode 的结果收集起来，计算梯度并更新参数。

## ES 的规模化与并行化

ES 方法很容易扩展到多个并行的 workers 上，因为它需要建立在完整的 episode 基础上，因此 workers 之间的通讯不频繁，同时如果每个 worker 知道其他 worker 的初始随机种子是什么，那么 workers 之间传递的信息只是一个 episode 的回报标量，非常节省带宽。

简单的并行 ES 版本如下，主要创新点是利用了已知的随机种子：

![](/images/2020-07-28-Evolution-Strategies-as-a-Scalable-Alternative-to-Reinforcement-Learning/image-20200729085726182.png)

在实际中，作者会事先为每个 worker 都生成一堆大量的高斯噪音，在训练的每个迭代中随机挑选一部分噪音加到策略的参数中。尽管这样每次迭代之间的噪音扰动不是完全独立的，但作者并没有发现很大的问题。另外当使用大量的并行 workers 或神经网络结构非常大时，每次可以只扰动一部分的 $\theta$ 。

为了减少方差，作者也使用了 autithetic sampling，也被称作 mirrored sampling，即评估扰动对 $\epsilon, -\epsilon$ 。除此之外，算法也使用了进化学习中的 fitness shaping，给每个回报排个序，加上一个顺序权重。

## Cross Entropy Method

在论文 [Learning Latent Dynamics for Planning from Pixels](http://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf) 中，作者在基于模型的 planning 阶段也使用了一种基于种群的进化学习方法 cross entropy method (CEM) 。CEM 算法非常简单，初始化一个与时间有关的高斯分布表示最优动作序列 $a_{t: t+H} \sim \operatorname{Normal}\left(\mu_{t: t+H}, \sigma_{t: t+H}^{2} \mathbb{I}\right)$ ，其中 $t$ 表示当前 time step，$H$ 是 planning 的长度。从零均值、单位方差开始，重复地采样 $J$ 个动作序列，将他们放入到环境模型中进行评估，然后使用最佳的 $K$ 个动作序列来更新高斯分布。在 $I$ 次迭代过后，返回当前步的高斯的均值 $\mu_t$ 作为动作才真正地交给环境与环境交互。收到下个观测值后，动作的高斯分布重新置为零均值、单位方差，以避免陷入局部最优。整体算法如下：

![](/images/2020-07-28-Evolution-Strategies-as-a-Scalable-Alternative-to-Reinforcement-Learning/image-20200731153201870.png)

这里因为已经有了环境模型，所以可以快速地生成一个完整的 planning 序列，选取 $K$ 个最优的序列是简单地计算每个序列的累积奖励，再进行排序。

## 参考

https://lilianweng.github.io/lil-log/2019/09/05/evolution-strategies.html

Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. *arXiv preprint arXiv:1703.03864*.