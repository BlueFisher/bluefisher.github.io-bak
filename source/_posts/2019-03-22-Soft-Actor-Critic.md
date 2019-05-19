---
title: Soft Actor-Critic
mathjax: true
typora-root-url: ..
date: 2019-03-22 10:22:50
categories:
- Reinforcement Learning
tags:
- RL
---

无模型的深度强化学习 (model-free deep reinforcement learning) 算法尽管非常多，效果也不错，但由于异策略采样与神经网络这种高维非线性函数近似的结合，使得 DRL 一直会有两个非常大的问题：采样复杂度大、对超参数非常敏感。在 *[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)* 论文中，伯克利与 Google Brain 联合提出了 Soft Actor-Critic，一种基于最大熵强化学习框架的异策略 actor-critic 算法。SAC 非常的稳定，可以在不同初始权重的情况下得到取得相同的性能。SAC 有三个显著的特点：

1. 策略与值函数分离的 actor-critic 框架
2. 异策略采样可以更有效地复用历史采集到的数据
3. 熵的最大化可以让算法更稳定，同时还能鼓励探索，找到多个都是最优的 near-optimal 行为。

最大熵强化学习可以参考 [Reinforcement Learning with Deep Energy-Based Policies](https://bluefisher.github.io/2018/11/13/Reinforcement-Learning-with-Deep-Energy-Based-Policies/) 。*[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)* 实际上最先提出了 SAC 算法，但本篇论文在此基础上做了一定的修改，尤其是改进了对 temperature hyperparmeter 敏感的问题。

<!--more-->

标准的强化学习目标函数是期望奖励的总和：
$$
\newcommand{\E}{\mathbb{E}}
\newcommand{\st}{s_t}
\newcommand{\at}{a_t}
\newcommand{\sT}{s_T}
\newcommand{\aT}{a_T}
\newcommand{\stp}{s_{t+1}}
\newcommand{\atp}{a_{t+1}}
\sum_t \E_{(\st,\at)\sim\rho_\pi}\left[{r(\st,\at)}\right]
$$

我们的目标就是找到一个策略 $\pi​$ 来最大化该目标函数。而最大熵目标函数增加了一项熵，最优策略则增加了在每一状态下最大化熵：
$$
\newcommand{\ent}{\mathcal{H}}
\pi^* = \arg\max_{\pi} \sum_{t} \E_{(\st, \at) \sim \rho_\pi}\left[{r(\st,\at) + \alpha\ent(\pi(\cdot|\st))}\right]
$$

其中 $\alpha​$ 就是 temperature parameter ，决定了相对于奖励值来说，熵的重要程度。

# Soft Policy Iteration

论文先在表格形式的情况下证明了带有最大化熵的 policy iteration 是能保证收敛。Policy iteration 分为 policy evaluation 和 policy improvement 两个过程。

## Soft Policy Evaluation

soft Q-value 用贝尔曼期望可以迭代地写成：
$$
\mathcal{T}^\pi Q(\st, \at) \triangleq  r(\st, \at) + \gamma \E_{s_{t+1} \sim p}\left[{V(s_{t+1})}\right] \tag{1}
$$
其中
$$
\begin{align*}
V(\st) &= \E_{\at\sim\pi}\left[{Q(\st, \at) - \alpha\log\pi(\at|\st)}\right] \\
&= \E_{\at\sim\pi}\left[{Q(\st, \at) + \alpha\ent(\pi(\cdot|\st))}\right]
\end{align*} \tag{2}
$$
为 soft state value function。可以导出以下soft policy evaluation 定理，即 Q function 一定会收敛：

> Consider the soft Bellman backup operator $\mathcal{T}^\pi​$ in (1) and a mapping $Q^0: \mathcal{S} \times \mathcal{A}\rightarrow \mathbb{R}​$ with $|\mathcal{A}|<\infty​$, and define $Q^{k+1} = \mathcal{T}^\pi Q^k​$. Then the sequence $Q^k​$ will converge to the soft Q-function of $\pi​$ as $k\rightarrow \infty​$.

## Soft Policy Improvement

对于当前策略下的最优 Q function，接下去的策略改进过程则是要找到一个新策略 $\pi'$ 使 $\pi'$ 与 soft Q-function 的 KL 散度最小。关于 soft Q-function 可参见 soft Q-learning (SQL)。
$$
\pi_\mathrm{new} = \arg\underset{\pi'\in \Pi}{\min} D_\mathrm{KL}\left({\pi'(\cdot|\st)} \Bigg{|}\Bigg{|} {\frac{\exp\left(\frac{1}{\alpha}Q^{\pi_\mathrm{old}}(\st, \cdot)\right)}{Z^{\pi_\mathrm{old}}(\st)}}\right) \tag{3}
$$
分母的配分函数 $Z^{\pi_\mathrm{old}}(\st)$ 将 Q 函数分布进行标准化。尽管它在实际训练过程中是无法计算的，但它并不参与到梯度的传播中，所以可以忽略。我们将策略改进正式定义为：

> Let $\pi _\mathrm{old} \in \Pi$ and let $\pi _\mathrm{new}$ be the optimizer of the minimization problem defined in (3). Then $Q^{\pi_\mathrm{new}}(\st, \at) \geq Q^{\pi_\mathrm{old}}(\st, \at)$ for all $(\st, \at) \in \mathcal{S}\times\mathcal{A}$ with $|\mathcal{A}|<\infty$.

将 soft policy evaluation 与 improvement 结合可以得到 soft policy iteration 定理：

> Repeated application of soft policy evaluation and soft policy improvement from any $\pi\in\Pi$ converges to a policy $\pi ^*$ such that $Q^{\pi^*}(\st, \at) \geq Q^{\pi}(\st, \at)$ for all $\pi \in \Pi$ and $(\st, \at) \in \mathcal{S}\times\mathcal{A}$, assuming $|\mathcal{A}|<\infty$.

# Soft Actor-Critic

上节是在表格形式下的理论推导，在实际操作过程中肯定还是需要对 soft Q-function 和 policy 进行值函数近似。我们用参数化的形式将 Q-function 定义为 $Q_\theta(\st,\at)​$ ，policy 为 $\pi_\phi(\at|\st)​$ ，Q-function 可以为一个神经网络结构，policy 可以为由神经网络输出的均值与协方差矩阵。

soft Q-function 的参数可以通过最小化 soft 贝尔曼方程来求得：
$$
J_Q(\theta) = \E_{(\st, \at)\sim\mathcal{D}} \left[{\frac{1}{2}\left(Q_\theta(\st, \at) - \left(r(\st, \at) + \gamma \E_{\stp\sim p}\left[{V_{\bar\theta}(\stp)}\right]\right)\right)^2}\right]
$$
其中 value function 可以用公式 (2) 隐式地进行表示，这 *[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)* 这篇首次提出 SAC 的论文中，value function 是个独立的需要训练的网络，但作者发现原论文中这样做是不必要的。于是梯度为：
$$
\hat \nabla_\theta J_Q(\theta) =  \nabla_\theta Q_\theta(\at, \st) \left(Q_\theta(\st, \at) - \left(r(\st,\at) + \gamma \left(Q_{\bar\theta}(\stp, \atp) - \alpha \log\left(\pi_\phi(\atp|\stp\right)\right)\right)\right) \tag{4}
$$
梯度更新过程中使用了 target soft Q-function ，用参数 $\bar\theta$ 表示。

策略则可以直接用公式 (3) 进行更新（乘以了 $\alpha$ 并且忽略的常量对数配分函数 $Z$）：
$$
\begin{align*}
J_\pi(\phi) &= \alpha \int\pi'(\cdot|\st) \log \left[{\frac{\pi'(\cdot|\st)}{\exp\left(\frac{1}{\alpha}Q^{\pi_\mathrm{old}}(\st, \cdot)\right)/Z}}\right] \mathrm{d}s\mathrm{d}a \\
&= \E_{\st\sim\mathcal{D}}\left[{\E_{\at\sim\pi_\phi}\left[{\alpha \log\left(\pi_\phi(\at|\st)\right) - Q_\theta(\st, \at)}\right]}\right]
\end{align*} \tag{5}
$$
有许多方法来最小化 $J_\pi$ ，比如经典的 REINFORCE 算法，它不需要将梯度在策略与目标概率密度网络中进行反向传播。但在 SAC 中，目标概率密度是用神经网络表示的 Q-function $\exp\left(\frac{1}{\alpha}Q^{\pi_\mathrm{old}}(\st, \cdot)\right)/Z$，它可以求导，所以我们也可以方便地使用重参数方法来降低方差：
$$
\at = f_\phi(\epsilon_t; \st)
$$
其中 $\epsilon_t$ 是输入的噪音向量，从一个固定的分布采样而来，比如高斯分布。我们将公式 (5) 修改为：
$$
J_\pi(\phi) = \E_{\st\sim\mathcal{D},\epsilon_t\sim\mathcal{N}}\left[{\alpha \log \pi_\phi(f_\phi(\epsilon_t;\st)|\st) - Q_\theta(\st, f_\phi(\epsilon_t;\st))}\right]
$$
其中策略 $\pi_\phi$ 隐式地根据 $f_\phi$ 来表示。梯度则为：
$$
\hat\nabla_\phi J_\pi(\phi) = \nabla_\phi \alpha \log\left( \pi_\phi(\at|\st)\right) + 
\big( 
\nabla_{\at} \alpha \log \left(\pi_\phi(\at|\st)\right) - \nabla_{\at} Q(\st, \at)
\big) \nabla_\phi f_\phi(\epsilon_t;\st) \tag{6}
$$
这种更新方式类似于 DDPG，Q 中的 $\at$ 也进行梯度传导。

##  Automating Entropy Adjustment for Maximum Entropy RL

在之前的小节中，我们都使用了固定了 temperature parameter $\alpha$ ，但 $\alpha$ 的选择是很困难的，对于不同的任务可能也需要不同的选择。为了解决这个问题，我们通过一个不同的最大化熵强化学习目标函数来将这个选择的过程自动化，此时的熵被视为一个常量。

我们的目标就是找到一个随机策略来使累积期望回报最大，同时也要满足一个最小的期望熵。也就是说所有状态的熵的期望要大于某个常量，这样不同的状态的熵就可以不同：
$$
\max_{\pi_{0:T}} \E_{\rho_\pi}\left[{\sum_{t=0}^T r(\st,\at)}\right] \text{ s.t. } \E_{(\st, \at)\sim\rho_\pi}\left[{-\log(\pi_t(\at|\st))}\right] \geq \mathcal{H}\ \ \forall t
$$
其中 $\mathcal{H}$ 就是这个最小的期望熵。

因为此时 $t​$ 的策略只会影响到未来的优化目标，所以可以使用动态规划方法，从后往前求解。写成迭代的形式为：
$$
\max_{\pi_0} \left( \E_{}\left[{r(s, a_0)}\right] + \max_{\pi_1} \left( \E_{}\left[{ \ldots }\right] + \max_{\pi_T} \E_{}\left[{r(s_T, a_T)}\right] \right) \right)
$$
从最后一个时刻开始，我们将带约束的最大化问题转变为对偶问题。即约束 $\E{(\sT, \aT)\sim\rho_\pi}{-\log(\pi_T(\sT|\sT))} \geq \mathcal{H}$ 用拉格朗日乘数法变为：
$$
\max_{\pi_T} \E_{(\st, \at)\sim\rho_\pi}\left[{r(s_T, a_T)}\right] = \min_{\alpha_T \geq 0} \max_{\pi_T} \E_{}\left[{r(s_T, a_T) - \alpha_T \log \pi(a_T|s_T)}\right] -\alpha_T \mathcal{H}
$$
其中 $\alpha_T$ 为拉格朗日乘子。这个对偶目标函数与最大化熵的目标函数在策略上有非常大的关系，最优策略实际上就是根据 $\alpha_T$ 而来的最大化熵策略：$\pi_T^*(a_T | s_T; \alpha_T)$ 。我们可以用下式来解得最优拉格朗日乘子 $\alpha_T$ ：
$$
\arg \min_{\alpha_T} \E_{\st, \at\sim \pi_t^*}\left[{- \alpha_T\log\pi_T^*(a_T|s_T; \alpha_T) - \alpha_T \ent}\right]
$$
最后，经过一系列复杂的迭代过程， $a_t^*$ 可由下式得出：
$$
\alpha_t^* = \arg \min_{\alpha_t} \E_{\at\sim \pi_t^*}\left[{- \alpha_t\log\pi_t^*(\at|\st; \alpha_t) - \alpha_t \bar\ent}\right]
$$

## Practical Algorithm

在实际过程中，为了减少在策略改进过程中的偏差，作者还使用了两个 soft Q-functions。在更新公式 (4) 与 (6) 时，使用最小的 Q-function 值。

对于 $\alpha_t$ 的更新，作者使用了近似对偶梯度更新方法：
$$
J(\alpha)  = \E_{\at\sim \pi_t}\left[{ - \alpha\log\pi_t(\at|\st) - \alpha \bar\ent}\right]
$$
最终算法为：

![](/images/2019-03-22-Soft-Actor-Critic/1553519092887.png)

# 参考

Haarnoja, Tuomas, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, et al. “Soft Actor-Critic Algorithms and Applications.” ArXiv:1812.05905 [Cs, Stat], December 12, 2018. http://arxiv.org/abs/1812.05905.

https://zhuanlan.zhihu.com/p/52526801