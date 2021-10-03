---
title: >-
  Efficient Transformers in Reinforcement Learning using Actor-Learner
  Distillation
mathjax: true
typora-root-url: ..
date: 2021-09-30 09:20:45
categories: Reinforcement Learning
tags:
---

本文主要聚焦于 transformer 模型样本利用高（收敛所需的 step 更少），但是训练速度慢 （每个 step 执行的时间长），LSTM 样本利用率低，但是训练速度快的特点，将 transformer 模型蒸馏到 LSTM 中，兼顾样本利用率与训练速度。

<!--more-->

作者通过实验比较了 LSTM 与 transformer 模型的样本利用率与训练使用时间：

![](/images/2021-09-30-Efficient-Transformers-in-Reinforcement-Learning-using-Actor-Learner-Distillation/image-20210930092407254.png)

很明显能看出，LSTM 的样本利用率很低，要跟环境交互非常多次才能收敛，但 LSTM 的训练非常快，尽管需要的样本多，但每个样本的训练时间很短。而 transformer 正好相反，样本利用率高，但每个样本的训练时间很长。

为了兼顾两者特点，在分布式架构中，中心化的 learner 使用 transformer 模型来进行训练，但在分布式的 actor 中，使用 LSTM 来与环境交互，同时将 learner transormer 中的策略蒸馏到 actor LSTM 中，起到兼顾样本利用率与训练速度的目的。

整个架构如下图所示：

![](/images/2021-09-30-Efficient-Transformers-in-Reinforcement-Learning-using-Actor-Learner-Distillation/image-20210930093128097.png)

所有的 Actor 并行地与环境交互收集数据，采集到完整的回合后，放入队列中。Learner Runner 从队列中取出适合训练的一批回合，存到 replay buffer 中，并同时交给 Learner 使用标准的强化学习进行训练。为了把 Learner 中的策略蒸馏到 Actor 中，Distill 模块利用经验池中数据，缩小 Learner 与 Actor 之间的差距。

对于蒸馏的部分，分为策略蒸馏与价值蒸馏两个部分：

$$
L_{A L D}^{\pi}=\mathbb{E}_{s \sim \pi_{A}}\left[\mathcal{D}_{K L}\left(\pi_{A}(\cdot \mid s) \| \pi_{L}(\cdot \mid s)\right)\right]=\mathbb{E}_{s \sim \pi_{A}}\left[\sum_{a \in \mathcal{A}} \pi_{A}(a \mid s) \log \frac{\pi_{L}(a \mid s)}{\pi_{A}(a \mid s)}\right]
$$

$$
L_{A L D}^{V}=\mathbb{E}_{s \sim \pi_{A}}\left[\frac{1}{2}\left(V_{L}^{\pi}(s)-V_{A}^{\pi}(s)\right)^{2}\right]
$$

策略蒸馏时，尽管本意是想将 $\pi_A$ 的策略靠近 $\pi_L$ ，但 $\pi_A$ ，$\pi_L$ 是同时更新的，对两个策略都做了一定的平滑。

价值蒸馏部分，是一个简单的 MSE 损失，此时仅更新 $V_A^\pi$ 。

两部分拼合起来即为：
$$
L_{A L D}=\alpha_{\pi} L_{A L D}^{\pi}+\alpha_{V} L_{A L D}^{V}
$$
从实验效果来看，论文中的方法取得了不错的样本利用率，同时也兼顾了训练速度。

![](/images/2021-09-30-Efficient-Transformers-in-Reinforcement-Learning-using-Actor-Learner-Distillation/image-20210930100851482.png)

