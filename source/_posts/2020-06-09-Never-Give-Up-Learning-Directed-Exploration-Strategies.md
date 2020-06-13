---
title: 'Never Give Up: Learning Directed Exploration Strategies'
mathjax: true
typora-root-url: ..
date: 2020-06-09 20:16:00
categories: Reinforcement Learning
tags: rl
---

*Never Give Up: Learning Directed Exploration Strategies* 是 DeepMind 在 ICLR 2020 上发表的一篇论文，主要解决探索与利用问题。这篇论文可以看作是三篇论文的结合：

1. Curiosity-driven Exploration by Self-supervised Prediction
2. Neural Episodic Control
3. Exploration by Random Network Distillation

也是使用内在奖励 (intrinsic reward) 机制，与外部奖励 (external reward) 合起来作为奖励值进行强化学习的训练。

<!--more-->

# Never Give Up

论文中 intrinsic reward 的主要思想很简单，主要有两部分组成：单个 episode 中的新颖程度 (per-episode novelty) 和整个训练阶段的新颖程度 (life-long novelty)。Episode novelty 鼓励智能体能周期性地在不同 episodes 之间访问相同的状态，但不鼓励在同一 episode 中访问相同的状态。而 life-long novelty 则逐渐降低不同 episodes 之间访问相同状态的新颖性。

Episode novelty 主要用论文 *Neural Episodic Control* 中的 episodic memory 来解决，其中的 embedding network 主要用 *Curiosity-driven Exploration by Self-supervised Prediction* 方法，life-long novelty 则使用论文 *Exploration by Random Network Distillation* 中的 Random Network Distillation 来解决。

## Embedding network

$f: \mathcal{O}\rightarrow\mathbb{R}^p$ 将当前的观测值映射到维度是 $p$ 的向量中。映射的训练方法就是 *Curiosity-driven Exploration by Self-supervised Prediction* 中给定 $t$ 与 $t+1$ 时刻的两个观测值，预测观测值转移时所做的动作 $p\left(a | x_{t}, x_{t+1}\right)=h\left(f\left(x_{t}\right), f\left(x_{t+1}\right)\right)$ ，在这个预测的训练过程中训练这个映射。

## Episodic memory and intrinsic reward

Episodic memory $M$ 就是一个动态的储存 embedding 后状态的缓冲区 $\left\{f\left(x_{0}\right), f\left(x_{1}\right), \ldots, f\left(x_{t-1}\right)\right\}$ 。

Intrinsic reward 定义为：

$$
r_{t}^{\text {episodic }}=\frac{1}{\sqrt{n\left(f\left(x_{t}\right)\right)}} \approx \frac{1}{\sqrt{\sum_{f_{i} \in N_{k}} K\left(f\left(x_{t}\right), f_{i}\right)}+c}
$$

$n\left(f\left(x_{t}\right)\right)$ 表示状态 $f\left(x_{t}\right)$ 访问过的次数，我们用核函数 $K: \mathbb{R}^{p} \times \mathbb{R}^{p} \rightarrow \mathbb{R}$ 来表示该状态与其他状态的相似度之和，以此来近似 $n\left(f\left(x_{t}\right)\right)$ 。在实际中，与 *Neural Episodic Control* 相同，用 KNN 来计算 $f(x_t)$ 在 $M$ 中 $k$ 个最近状态，用 $N_{k}=\left\{f_{i}\right\}_{i=1}^{k}$ 表示。$c$ 是一个小常数。$K$ 是一个逆核：
$$
K(x, y)=\frac{\epsilon}{\frac{d^{2}(x, y)}{d_{m}^{2}}+\epsilon}
$$
$\epsilon$ 也是一个小常数，$d$ 是欧拉距离，$d^2_m$ 是 $k$ 个最近状态的平方欧拉距离的平均值。

## Integrating life-long curiosity

Life-long curiosity 即是在 intrinsic reward 的基础上增加了一个动态系数 $\alpha_t$：
$$
r_{t}^{i}=r_{t}^{\text {episodic }} \cdot \min \left\{\max \left\{\alpha_{t}, 1\right\}, L\right\}
$$
这个系数代表当前状态有多新颖，用的是 *Exploration by Random Network Distillation* 方法。该方法也非常简单，任意选定一个随机的、权值固定不变的网络 $g: \mathcal{O} \rightarrow \mathbb{R}^{k}$ 和一个预测网络 $\hat{g}: \mathcal{O} \rightarrow \mathbb{R}^{k}$ 。让这个预测网络尽可能接近随机网络：
$$
\operatorname{err}\left(x_{t}\right)=\left\|\hat{g}\left(x_{t} ; \theta\right)-g\left(x_{t}\right)\right\|^{2}
$$

$$
\alpha_{t}=1+\frac{\operatorname{err}\left(x_{t}\right)-\mu_{e}}{\sigma_{e}}
$$

其中 $\sigma_e, \mu_e$ 是 $\operatorname{err}\left(x_{t}\right)$ 的动态标准差和均值。

# 参考

Badia, A. P., Sprechmann, P., Vitvitskyi, A., Guo, D., Piot, B., Kapturowski, S., ... & Blundell, C. (2020). Never Give Up: Learning Directed Exploration Strategies. *arXiv preprint arXiv:2002.06038*.

Pritzel, A., Uria, B., Srinivasan, S., Badia, A. P., Vinyals, O., Hassabis, D., ... & Blundell, C. (2017, August). Neural episodic control. In *Proceedings of the 34th International Conference on Machine Learning-Volume 70* (pp. 2827-2836). JMLR. org.

Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). Exploration by random network distillation. *arXiv preprint arXiv:1810.12894*.