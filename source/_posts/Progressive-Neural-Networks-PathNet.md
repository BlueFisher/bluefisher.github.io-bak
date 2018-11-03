---
title: Progressive Neural Networks & PathNet
mathjax: true
date: 2018-11-03 13:08:39
categories:
- Reinforcement Learning
tags:
- RL
- transfer learning
---

本文简单介绍两篇论文：*[Progressive Neural Networks](https://arxiv.org/pdf/1606.04671)* 和 *[PathNet: Evolution Channels Gradient Descent in Super Neural Networks](https://arxiv.org/pdf/1701.08734)* 。都出自于 Google DeepMind ，主要侧重于修改网络模型来解决针对强化学习中的迁移学习问题。

<!--more-->

# Progressive Neural Networks

微调 (finetuning) 网络模型的方法是迁移学习中的一个常用的方法，一般都是先在源任务中训练好一个神经网络，再在目标任务中利用反向传播微调整个网络模型。但这种微调的方法不大适合在多任务中进行迁移学习，比如我们希望汲取多个任务中的知识然后把这些知识迁移到新任务中，同时这种方法也非常容易使得网络模型忘记之前学习得到的知识。

作者提出了 progressive networks 可以克服以上缺点，它显式地架构了不同任务之间的迁移。整个网络架构十分简单，就一幅图：

![](https://s1.ax1x.com/2018/11/03/i42qiT.png)

Progressive network 一开始就是一个普通的单列神经网络，有 $L$ 层，包含 $h_i^{(1)}\in\mathbb{R}^{n_i}$ 个隐藏层，其中 $n_i$ 是第 $i\le L$ 层的单元数量，用 $\Theta^{(1)}$ 表示训练完毕的第一列神经网络参数。

当切换到第二个任务时， $\Theta^{(1)}$ 被锁住不动，并且整个网络新增加一列， $\Theta^{(2)}$ 开始随机初始化。 $h_i^{(2)}$ 层同时接受来自 $h_{i-1}^{(2)}$ 和 $h_{i-1}^{(1)}$ 的横向连接，推广到 $K$ 个任务用如下公式来表示：
$$
h_i^{(k)}=f\left( W_i^{(k)}h_{i-1}^{(k)} + \sum_{j<k}U_i^{(k:j)}h_{i-1}^{(j)} \right)
$$
强化学习中，每一列可以用来训练一个特定的 MDP ，相当于第 $k$ 列代表一个策略 $\pi^{(k)}(a|s)$ ，输入为状态 $s$ ，$\pi^{(k)}(a|s)=h_L^{(k)}(s)$ 。 

在实践中，我们会修改一下上述公式，增加一个 adapter 也就是图中的灰色 a 方块，它能既增强初始化条件又能降维：
$$
h_i^{(k)}=f\left( W_i^{(k)}h_{i-1}^{(k)} + U_i^{(k:j)}\sigma(V_i^{(k:j)}\alpha_{i-1}^{(<k)}h_{i-1}^{(<k)}) \right)
$$
其中 $h_{i-1}^{(<k)}=[h_{i-1}^{(1)}\cdots h_{i-1}^{(j)} \cdots h_{i-1}^{(k-1)}]$ 。再将之前的横向网络传入当前多层感知机之前，先乘以一个学习标量，用来调整之前横向层的影响程度。$V_i^{(k:j)}$ 是一个射影矩阵，用来进行降维，使得随着 $k$ 越来越多，但横向连接的参数数量与 $|\Theta^{(1)}|$ 相同，在卷积神经网络中，它可以为 $1 \times 1$ 卷积层。

当然 progressive networks 也有一些而问题，其中最大的问题时随着任务数量的增加，神经网络的参数量会越来越多。

# PathNet

