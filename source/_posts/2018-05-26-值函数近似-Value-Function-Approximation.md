---
title: 值函数近似 Value Function Approximation
mathjax: true
date: 2018-05-26
categories:
- Reinforcement Learning
- Course by David Silver
tags:
- RL
---

到目前为止，无论是不是基于模型的强化学习方法，都是通过维护一张价值表 $V(s)$ 或行为价值表 $Q(s,a)$ ，用查表 (lookup table) 的方式来进行学习。但在强化学习中，可能需要解决非常大型的问题，例如西洋双陆棋有 $10^{20}$ 规模的状态空间，围棋甚至高达 $10^{170}$ ，直升飞机的状态空间更是连续的。那么多状态、行为，没有办法通过一张表存储在内存里，学习起来也非常缓慢。为了解决大规模 MDP 的问题，我们可以用某个近似函数来估计真实的价值函数：
$$
\begin{align*}
\hat{v}(s,w) &\approx v_\pi(s) \\ 
\text{or } \hat{q}(s,a,w) &\approx q_\pi(s,a) 
\end{align*}
$$
通过 MC 或 TD 的学习方法来更新近似函数的参数 $w$

<!--more-->

# 近似函数的形式

对于状态价值函数，近似函数的设计很简单，输入状态 $s$ ，输出状态价值函数 $\hat{v}(s,w)$

对于行为价值函数，有两种形式：

1. 输入状态 $s$ 和行为 $a$ ，输出行为价值函数 $\hat{q}(s,a,w)$
2. 只输入状态 $s$ ，输出该状态下的所有行为价值函数 $\hat{q}(s,a_1,w) \cdots \hat{q}(s,a_m,w)$

通常我们会使用第二种形式。那么如何来构造近似函数呢？有许多近似函数的构造方式，我们选取可导的一些近似函数形式，比如线性组合或者神经网络。由于强化学习的训练特殊性，我们最好需要可以适用于非静态的 (non-stationary) 、非独立同分布的 (non-iid) 数据的学习方法。

# 递增方法 Incremental Methods

## 随机梯度下降 Stochastic Gradient Descent

由于引入了近似价值函数，那么我们的首要目标就是最小化近似价值函数与真实价值函数之间的最小均方误差：
$$
J(w) = \mathbb{E}_\pi[( v_\pi(S)-\hat{v}(S,w) )^2]
$$
应用梯度下降能够找到局部最小值：
$$
\begin{align*} \Delta w &= -\frac{1}{2}\alpha \nabla_w J(w) \\ &= \alpha \mathbb{E}_\pi[(v_\pi(S) - \hat{v}(S,w)) \nabla_w \hat{v}(S,w)] \end{align*}
$$
用随机梯度下降来对梯度进行采样
$$
\Delta w = \alpha (v_\pi(S) - \hat{v}(S,w)) \nabla_w \hat{v}(S,w)
$$

## 递增预测算法 Incremental Prediction Algorithms

上文中假定了我们已知真实价值函数 $v_\pi(s)$ 相当于监督学习，但在强化学习中，没有真实标记，只有奖励，所以在实际中，我们用目标值来代替真实价值函数，如果只考虑线性组合的话 $\hat{v}(S,w) = x(S)^\text{T} w$

对于 MC ，目标值是 $G_t$
$$
\Delta w=\alpha(G_t - \hat{v}(S_t,w)) \nabla_w \hat{v}(S_t,w)
$$
对于 TD(0) ，目标值就是 TD target $R_{t+1} + \gamma \hat{v}(S_{t+1},w)$
$$
\Delta w=\alpha(R_{t+1} + \gamma \hat{v}(S_{t+1},w) - \hat{v}(S_t,w)) \nabla_w \hat{v}(S_t,w)
$$
对于 TD(λ) ，前向视角的目标值是 $G_t^\lambda$
$$
\Delta w=\alpha(G_t^\lambda - \hat{v}(S_t,w)) \nabla_w \hat{v}(S_t,w)
$$
反向视角为：
$$
\begin{align*} \delta_t &= R_{t+1} +\gamma \hat{v}(S_{t+1},w) - \hat{v}(S_t,w) \\ E_t &= \gamma \lambda E_{t-1} + x(S_t) \\ \Delta w &= \alpha \delta_t E_t \end{align*}
$$
除了状态价值函数，对于行为价值函数也是同样的道理。

## 收敛性 Convergence

### 预测学习的收敛性 Convergence of Prediction Algorithms

| 同 / 异策略 | 算法 | 查表 | 线性 | 非线性 |
| ----------- | ---- | ---- | ---- | ------ |
| 同策略      | MC   | ✔️    | ✔️    | ✔️      |
| TD(0)       | ✔️    | ✔️    | ❌    |        |
| TD(λ)       | ✔️    | ✔️    | ❌    |        |
| 异策略      | MC   | ✔️    | ✔️    | ✔️      |
| TD(0)       | ✔️    | ❌    | ❌    |        |
| TD(λ)       | ✔️    | ❌    | ❌    |        |

如果不用函数近似，则所有算法都能收敛，使用线性组合来近似，则异策略的 TD 算法无法收敛，非线性近似则除了 MC 都无法收敛。

### 控制学习的收敛性 Convergence of Control Algorithms

| 算法       | 查表 | 线性 | 非线性 |
| ---------- | ---- | ---- | ------ |
| MC         | ✔️    | (✔️)  | ❌      |
| Sarsa      | ✔️    | (✔️)  | ❌      |
| Q-learning | ✔️    | ❌    | ❌      |

(✔️) 表示在最优价值函数附近震荡

对于近似函数来说，都不会是严格收敛的，比较常见的是在最优策略上下震荡，逐渐逼近然后突然来一次发散，再逐渐逼近等。使用非线性函数近似的效果要比近似函数要差很多，实际也是如此。

# 参考

<http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/FA.pdf>

<https://zhuanlan.zhihu.com/p/28223841>