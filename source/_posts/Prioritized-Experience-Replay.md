---
title: Prioritized Experience Replay
mathjax: true
date: 2018-05-25
updated: 2018-05-25
categories:
- Reinforcement Learning
tags:
- RL
- DQN
---


在之前的 [Deep Q-Network](https://bluefisher.github.io/2018/05/07/Deep-Q-Network/) 中，引入了经验回放机制，在经验记忆 (replay memory) 中均匀地选取的经验片段 (replay transition) ，但是这种简单等概率的选取经验片段的方法忽视了每条片段的重要程度。所以在论文 *Prioritized Experience Replay* 中，提出了一种优先经验的方法，将一些比较重要的经验片段回放的频率高一点，从而使得学习更有效率。

<!--more-->

# 优先经验回放

要修改我们的经验记忆可以在两个点上进行改动，一是选择存储哪些经验片段，二是选择回放哪些经验片段。在论文中，只阐述了后者，也就是回放最有价值的片段。

## TD-error

优先回放的一个中心问题就是衡量一个经验片段重要程度的标准。一个理想的标准是在一个片段的当前状态下能够学习到的量的大小，但这个标准无法直接获取到。另一种比较合理的标准就是一个片段的 TD error $\delta$ ，它可以表示某个片段的预测价值与自举 (bootstrap) 的真实价值的偏离程度，或者说这个片段有多“惊人” (surprising) 、多“出乎意料” (unexpected) 。这个标准特别适合在在线 (online) 的强化学习算法比如 SARSA 或者 Q-learning 中使用，因为它们已经计算过了 TD-error ，但在某些情况下 TD-error 也会变成一个比较差的预测量，比如当奖励值有很多的噪声。

## 随机优先 Stochastic Prioritization

使用贪婪法通过比较 TD-error 的大小来优先选取经验片段有许多问题：一些 TD-error 较小的片段可能在很长的一段时间内不能被选取到，对于噪声也非常的敏感，会因为强化算学算法的自举而加剧这种情况出现，同时函数近似的误差也会成为另一种噪声，贪婪法还会只关注经验记忆中的一部分片段，导致误差收敛得很慢。

所以为了解决这些问题，引入了一种随机采样的方法介于贪婪选取与均匀选取两者之间，使得经验片段被选取到的概率随着优先级的递增而单调递增，但同时也保证对于低优先级的片段不至于零概率被选中。具体来说，定义了选取某个片段 $i$ 的概率为：
$$
P(i) = \frac{p_i^\alpha}{\sum_kp_k^\alpha}
$$
其中 $p_i > 0$ 代表某个片段的优先级，指数 $\alpha$ 决定了这个优先级使用多少，如果 $\alpha=0$ 那么就相当于均分采样。

对于优先级，第一种设置优先级的方法是 $p_i=|\delta_i| + \epsilon$ ，其中 $\epsilon$ 是一个小的正常量来防止 TD-error 变为 0 之后就不再被访问。第二种设置方法是 $p_i = \frac{1}{rank(i)}$ ，其中 $rank(i)$ 就是根据 $|\delta_i|$ 进行排序的排位。两种方法都是根据 $|\delta_i|$ 单调的，但后一种方法更加稳定。

在实际实现过程中，可以将所有的排名分为 `batch_size` 个区间，在每个区间内进行均匀采样。我们可以使用 sum-tree 这种数据结构，这样在插入、更新一个片段时，整体数据结构不需要排序，只要 $O(1)$ ，而在采样时，只需要 $O(\log N)$

## 消除偏差

用优先经验的方法代替均匀采样的方法会引入偏差，因为它以一种不受控的形式改变了原来均匀分布，而我们可以用重要性采样的方式来消除偏差。

由于：
$$
\mathbb{E}_{X \sim A}[f(X)] = \mathbb{E}_{X\sim B}\left[ \frac{P_A(X)}{P_B(X)}f(X) \right]
$$
其中 $P_A(X) = \frac{1}{N}$ 也就是均匀采样， $N$ 为整个记忆库的大小， $P_B(X)=P(i)$ ，则重要性权重为：
$$
w_i=\left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta
$$
若 $\beta=1$ 则可以完全补偿非均匀分布所带来的偏差。为了稳定性方面的考虑，我们规格化重要性权重：
$$
w_j= \frac{(N \cdot P(j))^{-\beta}}{\max_i w_i}
$$
这些重要性权重可以应用到 Q-learning 的更新中，用 $w_i\delta_i$ 代替原先的 $\delta_i$

## 算法

![](https://s1.ax1x.com/2018/05/26/CfhJZ8.png)

该算法将 Double DQN 与 本文的 Prioritized Experience Replay 结合起来，最主要的修改在于选取经验片段和用重要性采样来更新参数两个点。

# 参考

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. *arXiv preprint arXiv:1511.05952*.

<https://www.cnblogs.com/wangxiaocvpr/p/5660232.html>

<http://www.meltycriss.com/2018/03/18/paper-prioritized-experience-replay/>