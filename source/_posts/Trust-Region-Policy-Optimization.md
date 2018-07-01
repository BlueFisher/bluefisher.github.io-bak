---
title: Trust Region Policy Optimization
mathjax: true
date: 2018-06-30 09:37:40
updated: 2018-06-30 09:37:40
categories:
- Reinforcement Learning
tags:
- RL
---

在 *Trust Region Policy Optimization* 论文中，作者提出了一种保证策略迭代优化的过程单调不减地朝更好的方向发展的算法，也就是说每一次策略的改善，都保证改善后的策略比前一个策略要好。在理论的基础上做了一些近似后，得到了能实际运用的算法： Trust Region Policy Optimization (TRPO)

策略梯度的参数更新方程为：
$$
\theta_{new} = \theta_{old} + \alpha \nabla_\theta J
$$
其中 $\alpha$ 为更新步长，然而这个步长很难确定，一个不好的步长会使参数所对应的策略变得不好，而接着又用这个不好的策略来进行采样学习，会使得策略越来越差，所以步长至关重要，如何选取步长使得策略不会变得更差就是 TRPO 要解决的问题。

<!--more-->

# 背景

定义 $\eta(\pi)$ 为衰减后总回报的期望：
$$
\eta(\pi) = \mathbb{E}_{s_0,a_0,\cdots} \left[\sum_{t=0}^{\infty} \gamma^t r(s_t)  \right]
$$
其中， $s_0 \sim \rho_0(s_0),\  a_t \sim \pi(a_t|s_t),\  s_{t+1}\sim P(s_{t+1}|s_t,a_t)$ ，$P$ 为状态转移概率， $\rho_0$ 为初始状态 $s_0$ 的概率分布。我们的目标就是找到一个策略，使得总回报最大。

引言中说过， TRPO 就是在策略迭代的过程中，使得新策略的总回报单调不减，那我们可以把某一个新策略 $\tilde{\pi}$ 的总回报根据 *Kakade & Langford (2002)* 拆分成与旧策略 $\pi$ 有关的两项：
$$
\eta(\tilde{\pi}) = \eta(\pi) + \mathbb{E}_{s_0,a_0,\cdots\sim\tilde{\pi}} \left[ \sum_{t=0}^{\infty} \gamma^t A_\pi(s_t,a_t) \right] \tag{1}
$$
可以看出 $ \mathbb{E}_{s_0,a_0,\cdots\sim\tilde{\pi}}[\cdots]$ 表示了行为的选取符合 $a_t \sim \tilde{\pi}(\cdot |s_t)$ 

定义 $\rho_\pi$ 为衰减过后的状态概率分布：
$$
\rho_\pi(s) = P(s_0=s) + \gamma P(s_1=s) + \gamma ^2 P(s_2=s) + \cdots
$$
我们将 公式 (1) 展开：
$$
\begin{align*}
\eta(\tilde{\pi}) &= \eta(\pi) + \sum_{t=0}^\infty \sum_s P(s_t=s|\tilde{\pi})\sum_a \tilde{\pi}(a|s)\gamma^t A_\pi(s,a) \\
&= \eta(\pi) + \sum_s \rho_{\tilde{\pi}}(s) \sum_a \tilde{\pi}(a|s) A_\pi(s,a) \tag{2}
\end{align*}
$$
这个公式表明，对于任何策略的改善 $\pi \rightarrow \tilde{\pi}$ 在每一个状态下，都会有一个非负的优势函数期望，也就是 $\sum_a \tilde{\pi}(a|s)A_\pi(s,a) \ge 0$ ，以此来保证策略的总回报 $\eta$ 是递增的或者不变（当优势函数等于0），因此，我们可以使用确定性策略 (deterministic policy) 的方法： $\tilde{\pi}=\arg\max_a A(s,a)$ ，如果至少有一个状态、行为对 (state-action pair) 有着正优势函数值 (positive advantage value) ，则策略就会得到改善，否则会收敛到局部最优解当中。

在 公式 (2) 中，依赖于新策略 $\tilde{\pi}$ 的 $\rho_{\tilde{\pi}}(s)$ 很难去直接优化，所以用以下近似函数：
$$
L_\pi(\tilde{\pi}) = \eta(\pi) + \sum_s \rho_{\pi}(s) \sum_a \tilde{\pi}(a|s) A_\pi(s,a)
$$
注意到 $L_\pi$ 使用了访问频次 $\rho_\pi$ 而不是 $\rho_{\tilde{\pi}}$ ，因为如果我们使用参数化的策略，并且 $\pi_\theta(a|s)$ 是可导的，那么新策略与旧策略的参数非常接近时，这种近似是合理的，同时 $L_\pi$ 与 $\eta$ 一阶近似，也就是对于任意参数 $\theta_0$ ，都有：
$$
L_{\pi_{\theta_0}}(\pi_{\theta_0}) = \eta(\pi_{\theta_0})\\
\nabla_\theta L_{\pi_{\theta_0}}(\pi_{\theta}) | _{\theta=\theta_0} = \nabla_\theta \eta(\pi_\theta) | _{\theta=\theta_0}
$$
这表明一个小的步长使 $\pi_{\theta_0} \rightarrow \tilde{\pi}$ 可以改善 $L_{\pi_{\theta_{old}}}$ 同时也会改善 $\eta$ ，但并没有告诉我们这个步长到底有多长。

# 随即策略的单调化改进 Monotonic Improvement Guarantee for General Stochastic Policies

为了解决步长的问题，作者根据之前 Kakade & Langford 的公式推导出了一个重量级不等式。

令 $D_{KL}^{\max} (\pi,\tilde{\pi}) = \max_s D_{KL} (\pi(\cdot|s) \ \|\  \tilde{\pi}(\cdot|s))$ 则：
$$
\begin{align*}
\eta(\tilde{\pi}) &\ge L_\pi(\tilde{\pi}) - CD_{KL}^{\max} (\pi,\tilde{\pi}), \\
& \text{where } C = \frac{4\epsilon\gamma}{(1-\gamma)^2} \\
& \text{and } \epsilon=\max_{s,a}|A_\pi(s,a)|
\end{align*} \tag{3}
$$
以下的 算法1 根据上述 公式 (9) 所表示的策略更新的下界，描述了近似的策略迭代方法：

![](https://s1.ax1x.com/2018/06/30/PFfnFs.png)

公式 (9) 可以保证 算法 1 的策略更新是单调的，也就是说更新的策略序列 $\eta(\pi_0) \le \eta(\pi_1) \le \eta(\pi_2) \le \cdots$ 

证明：令 $M_i(\pi) = L_{\pi_i}(\pi) - CD_{KL}^{\max} (\pi_i,\pi)$ ，根据 公式 (9) 可得 $\eta(\pi_{i+1}) \ge M_i(\pi_{i+1})$ ，由于 $D_{KL}(\pi_i \ \| \ \pi_i) =0$ ，所以 $\eta(\pi_i) = M_i(\pi_i) = L_{\pi_i}(\pi_i)$ ，两式合并一下得到：
$$
\eta(\pi_{i+1}) - \eta(\pi_i) \ge M_i(\pi_{i+1}) - M_i(\pi_i)
$$
因此，如果我们在每一次迭代中，将 $M_i$ 最大化，并且选取 $\pi_{i+1} = \arg\max_\pi [M_i(\pi)]$ ，那么就可以保证总回报 $\eta$ 起码不会变得更差。

# 参数化策略的优化 Optimization of Parameterized Policies

在之前的小节中，我们独立于策略的参数考虑了策略优化问题，现在考虑参数化的策略。首先，我们稍微改变一下符号，用参数 $\theta$ 直接代替 $\pi$ 

之前的小节提出了 $\eta(\theta) \ge L_{\theta_{old}} - CD_{KL}^{\max}(\theta_{old},\theta)$ ，当 $\theta = \theta_{old}$ 时取等号。因此，只要最大化下面的公式，就能保证往好的方向改进总回报：
$$
\max_\theta[L_{\theta_{old}}(\theta) - CD_{KL}^{\max}(\theta_{old},\theta)]
$$
在实际中，如果我们使用上面理论中提到的惩罚 $C$ ，那么步长会变得很小。一种解决方法是约束新旧策略之间的 KL 散度，也就是置信域约束 (trust region constraint) ：
$$
\begin{align*}
&\max_\theta L_{\theta_{old}}(\theta)\\
&\text{subject to }D_{KL}^{\max}(\theta_{old},\theta) \le \delta
\end{align*}
$$
但这种方法并不能应用到实际中，因为状态空间可能会非常大，比如连续状态，导致会有非常多的约束，所以我们使用启发式的近似方法——平均 KL 散度：
$$
\bar{D}_{KL}^\rho(\theta_1,\theta_2) :=  E_{s\sim \rho}[D_{KL}( \pi_{\theta_1}(\cdot | s) \ \|\  \pi_{\theta_2}(\cdot | s) )]
$$
因此，我们通过解决以下优化问题来进行策略更新:
$$
\begin{align*}
&\max_\theta L_{\theta_{old}}(\theta)\\
&\text{subject to } \bar{D}_{KL}^{\rho_{\theta{old}}}(\theta_{old},\theta) \le \delta
\end{align*} \tag{4}
$$

# 基于采样的估计 Sample-Based Estimation of Objective and Constraint

上一小节提出了一个带约束的优化问题，即在每一步更新时，在一个约束下优化总回报的期望。这一小节将说明如何使用蒙特卡罗方法来近似的估计目标函数与约束。

我们先将 公式 (4) 中的 $L_{\theta_{old}}(\theta)$ 展开来表示：
$$
\begin{align*}
&\max_\theta \sum_s \rho_{\theta_{old}}(s) \sum_a \pi_\theta(a|s) A_{\theta_{old}}(s,a) \\
&\text{subject to } \bar{D}_{KL}^{\rho_{\theta{old}}}(\theta_{old},\theta) \le \delta
\end{align*} \tag{5}
$$
首先用 $\frac{1}{1-\gamma}\mathbb{E}_{s\sim\rho_{\theta_{old}}}[\cdots]$ 来代替 $\sum_s \rho_{\theta_{old}}(s)$ ；接着用行为价值函数 $Q_{\theta_{old}}$ 代替优势函数 $A_{\theta_{old}}$ ，因为这只改变了一个常数项；最后我们用重要性采样来代替对行为的求和，用 $q$ 来表示采样的概率分布，则对于某一个状态 $s_n$ 的损失函数为：
$$
\sum_a \pi_\theta(a|s_n) A_{\theta_{old}}(s_n,a) = \mathbb{E}_{a\sim q}\left[ \frac{\pi_\theta(a|s_{n})}{q(a|s_{n})}A_{\theta_{old}}(s_{n},a) \right]
$$
综上三点，公式 (5) 与可以写成下列的期望形式：
$$
\begin{align*}
& \max_{\theta}E_{s\sim \rho_{\theta_{old}}, a\sim q}\left[ \frac{\pi_\theta(a|s)}{q(a|s)}Q_{\theta_{old}}(s,a) \right] \\
&\text{subject to }  E_{s\sim \rho_{\theta_{old}}}[D_{KL}( \pi_{\theta_{old}}(\cdot | s) \ \|\  \pi_{\theta}(\cdot | s) )] \le \delta
\end{align*} \tag{6}
$$
最终的算法可以分为以下三个部分：

1. 采样一系列的状态、行为对，用蒙特卡罗方法估计它们的行为价值函数
2. 通过平均这些样本，建立 公式 (6) 中的目标函数与约束
3. 近似地解决带约束的优化问题来更新策略参数 $\theta$ 。我们使用共轭梯度算法来解决这个问题，此处不再展开

# 参考

Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015, June). Trust region policy optimization. In *International Conference on Machine Learning* (pp. 1889-1897). 

https://zhuanlan.zhihu.com/p/26308073

https://blog.csdn.net/philthinker/article/details/79551892