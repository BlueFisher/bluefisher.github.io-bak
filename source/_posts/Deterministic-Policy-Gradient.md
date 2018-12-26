---
title: Deterministic Policy Gradient
mathjax: true
date: 2018-05-16 10:42:32
categories:
- Reinforcement Learning
tags:
- RL
- PG
---

在 *Deterministic Policy Gradient Algorithms* 一文中，DeepMind 在原有随机策略梯度 (Stochastic Policy Gradient) 算法的基础上介绍了确定性策略梯度算法 (Deterministic Policy Gradient Algorithms DPG) 来解决连续性决策问题，是之后 Deep DPG (DDPG) 算法的基础。

传统的策略梯度算法以概率形式 $\pi_\theta(a|s) = \mathbb{P}[a|s; \theta]$ 来表示一个策略，以此来随机的选择行为。但 DPG 用一种确定性的策略形式 $a=\mu_\theta(s)$ 。

DPG 有着比 PG 更简单的形式：行为价值函数的期望，这也使得 DPG 比 PG 更加有效，同时在高维度行为空间中也比 PG 表现得更加好。

<!--more-->

# 随机策略梯度 Stochastic Policy Gradient

在 [策略梯度 Policy Gradient](https://bluefisher.github.io/2018/05/10/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6-Policy-Gradient/) 一文中说过，策略梯度算法根本目的就是要使累计奖赏最大，即让以下期望最大：
$$
\begin{align*}
J(\pi_\theta) &= \int_\mathcal{S} \rho^\pi(s) \int_\mathcal{A} \pi_\theta (s,a)r(s,a) \mathrm{d}a \mathrm{d}s \\
&= \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} [r(s,a)]
\end{align*}
$$

策略梯度的基本思想就是沿着 $\nabla_\theta J(\pi_\theta)$ 方向调整参数：
$$
\begin{align*}
\nabla_\theta J(\pi_\theta) &= \int_\mathcal{S} \rho^\pi(s) \int_\mathcal{A} \nabla_\theta \pi_\theta (s,a) Q^\pi(s,a) \mathrm{d}a \mathrm{d}s \\
&= \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)]
\end{align*}
$$

## 随机 Actor-Critic 算法 Stochastic Actor-Critic Algorithms

即用 critic 来估计行为价值函数 $Q^w(s,a) \approx Q^\pi(s,a)$ 例如时间差分 (TD) 算法。
$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) Q^w(s,a)]
$$
引入了近似函数就可能造成偏差，所以近似函数需要满足

1. $Q^w(s,a) = \nabla_\theta \log \pi_\theta (a|s) ^\mathrm{T} w$
2. $\epsilon^2(w) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} \left[ (Q^w(s,a) - Q^\pi(s,a))^2 \right]$ 最小

## 异策略 Actor-Critic 算法 Off-Policy Actor-Critic

即用不同于行为策略 $π$ 的异策略 $β$ 来选择状态、行为轨迹 (trajectories)
$$
J_\beta (\pi_\theta) = \int_\mathcal{S} \rho^\beta(s) \int_\mathcal{A} \pi_\theta (s,a)r(s,a) \mathrm{d}a \mathrm{d}s
$$

异策略的策略梯度为：
$$
\begin{align*}
\nabla_\theta J_\beta(\pi_\theta) &= \int_\mathcal{S} \rho^\beta(s) \int_\mathcal{A} \nabla_\theta \pi_\theta (s,a) Q^\pi(s,a) \mathrm{d}a \mathrm{d}s \\
&= \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} \left[ \frac{π_θ(a|s)}{β_θ(a|s)} \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a) \right]
\end{align*}
$$

一般来说，critic 会用状态价值函数 $V^v(s) \approx V^\pi(s)$ 来进行估计，所以在这里用 TD-error $\delta_t=r_{t+1} + \gamma V^v(s_{t+1})-V^v(s_t)$ 代替上式中的 $Q^\pi(s,a)$ 。注意到在更新 actor 和 critic 时，都需要用重要性采样比率 $\frac{π_θ(a|s)}{β_θ(a|s)}$ 来进行重要性采样。

# 确定性策略梯度 Gradients of Deterministic Policies

现在考虑确定性策略，即 $a=\mu_\theta(s)$ 。我们从两种角度来导出确定性策略梯度。

## 行为价值函数的梯度 Action-Value Gradients

绝大多数的无模型强化学习算法都是基于策略迭代，也就是策略评估 (policy evaluation) 和策略改善 (policy improvement) 两步。策略评估就是来估计行为价值函数 $Q^\pi(s,a)$ 或 $Q^\mu(s,a)$ ，比如用 MC 或 TD 来进行估计，然后再进行策略改善，最常用的方法是用贪婪法：$\mu^{k+1}(s)=\arg\max_a Q^{\mu^k}(s,a)$

在连续行为空间中，策略改善环节的贪婪法就变得不可行了。我们可以做一点小的改变，就是让策略梯度参数 $θ^{k+1}$ 根据行为价值函数的梯度方向 $\nabla_θ Q^{μ^k}(s,μ_θ(s))$ 来进行更新，即：
$$
\theta^{k+1} = \theta^k + \alpha  \mathbb{E}_{s \sim \rho^{\mu^k}} \left[ \nabla_θ Q^{μ^k}(s,μ_θ(s)) \right]
$$
根据导数的链式法则：
$$
\theta^{k+1} = \theta^k + \alpha  \mathbb{E}_{s \sim \rho^{\mu^k}} \left[ \nabla_θ \mu_\theta(s) \nabla_a Q^{\mu^k} (s,a)|_{a=\mu_\theta(s)} \right]
$$

## 确定性策略梯度定理 Deterministic Policy Gradient Theorem

类比随机策略，因为此时是确定性的策略，所以不需要再对行为 $a$ 做积分求期望，则累计奖励期望为：
$$
\begin{align*}
J(\mu_\theta) &= \int_\mathcal{S} \rho^\mu(s) r(s,\mu_\theta(s)) \mathrm{d}s \\
&= \mathbb{E}_{s \sim \rho^\mu}[r(s,\mu_\theta(s))]
\end{align*}
$$
与随即策略梯度相同，我们使用 $Q$ 值来代替即时奖励，则对于 $J$ 的梯度，即 DPG 为：
$$
\begin{align*}
\nabla_\theta J(\mu_\theta) &= \int_\mathcal{S} \rho^\mu(s) \nabla_θ \mu_\theta(s) \nabla_a Q^{\mu} (s,a)|_{a=\mu_\theta(s)} \mathrm{d}s \\
&= \mathbb{E}_{s \sim \rho^\mu} \left[ \nabla_θ \mu_\theta(s) \nabla_a Q^{\mu} (s,a)|_{a=\mu_\theta(s)} \right] \\
&= \mathbb{E}_{s \sim \rho^\mu} \left[ \nabla_\theta Q^{\mu} (s,\mu_\theta(s)) \right]
\end{align*}
$$
可以发现，与随机策略梯度相比，DPG 少了对行为的积分，多了对行为价值函数的梯度，这也使得 DPG 需要更少的采样却能达到比随机策略梯度更好的效果。

## 随机策略梯度的极限形式 Limit of the Stochastic Policy Gradient

DPG 公式乍一看并不像随机策略梯度公式，但实际上 DPG 是随机策略梯度的一种特例形式。假设定义随机策略的参数为 $\pi_{\mu_\theta, \sigma}$ ，其中 $\sigma$ 为方差参数，也就是说，如果 $\sigma=0$ ，则随机策略等于确定性策略 $\pi_{\mu_\theta, \sigma} \equiv \mu_\theta$ ，所以可以得出策略梯度的极限形式：
$$
\lim_{\sigma \rightarrow 0} \nabla_\theta J(\pi_{\mu_\theta, \sigma}) = \nabla_\theta J(\mu_\theta)
$$

# 确定性 Actor-Critic 算法

与随机 Actor-Critic 算法类似，用一个可导的行为价值函数 $Q^w (s,a)$ 来估计 $Q^\mu(s,a)$

## On-Policy

对于同策略 Actor-Critic ，使用 Sarsa 来估计行为价值函数，算法为：
$$
\begin{align*}
\delta_t &= r_t + \gamma Q^w(s_{t+1},a_{t+1}) - Q^w(s_t,a_t) \tag{11} \\
w_{t+1} &= w_t + \alpha_w \delta_w \nabla_w Q^w(s_t,a_t) \\
\theta_{t+1} &= \theta_t + \alpha_\theta  \nabla_θ \mu_\theta(s) \nabla_a Q^w (s,a)|_{a=\mu_\theta(s)}
 \end{align*}
$$

## Off-Policy

而对于异策略来说，在生成样本轨迹时所用的策略可以是任意的随机行为策略 $\beta(s,a)$，累计奖励 $J$ 变为：
$$
J_\beta (\mu_\theta) = \int_\mathcal{S} \rho^\beta(s) Q^\mu(s,\mu_\theta(s)) \mathrm{d}s
$$
使用 Q-Learning 算法来估计行为价值函数：
$$
\begin{align*}
\delta_t &= r_t + \gamma Q^w(s_{t+1},\mu_\theta(s_{t+1})) - Q^w(s_t,a_t) \tag{16} \\
w_{t+1} &= w_t + \alpha_w \delta_w \nabla_w Q^w(s_t,a_t) \\
\theta_{t+1} &= \theta_t + \alpha_\theta  \nabla_θ \mu_\theta(s) \nabla_a Q^w (s,a)|_{a=\mu_\theta(s)}
 \end{align*}
$$
可以看出同策略与异策略的不同之处在于对 $a_t$ 行为的生成，同策略用的是确定性策略，而异策略则是一个任意的随机策略。对于论文中公式11与公式16处，$a_{t+1}$ 行为的生成方式是相同的，尽管写的不同，都是用确定性策略来生成，这也是我在读论文式比较困惑的一点，还好在 StackExchange 上提问有大神回答了 [链接](https://ai.stackexchange.com/questions/6317/what-is-the-difference-between-onoff-policy-deterministic-actor-critic) 。

论文接下去还有许多内容，由于该论文只对后面的 DDPG 算法打个基础，我也没有非常深入的理解接下去的内容。

# 参考

Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014, June). Deterministic policy gradient algorithms. In *ICML*. 

Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. In *Advances in neural information processing systems* (pp. 1057-1063). 

<https://ai.stackexchange.com/questions/6317/what-is-the-difference-between-onoff-policy-deterministic-actor-critic>

