---
title: High-Dimensional Continuous Control Using Generalized Advantage Estimation
mathjax: true
date: 2018-07-16 17:10:26
categories:
- Reinforcement Learning
tags:
- RL
- PG
---

本文简单介绍一下 *High-Dimensional Continuous Control Using Generalized Advantage Estimation* 这篇论文。该论文主要做了两件事：

1. 使用一种类似于 TD(λ) 的优势函数的估计形式来做到值函数的方差偏差平衡  (bias-variance tradeoff)
2. 将置信域优化算法 (trust region optimization) 同时用在策略与值函数的优化上

<!--more-->

定义策略梯度：
$$
g = \mathbb{E}\left[ \sum_{t=0}^\infty \Psi_t\nabla_\theta \log\pi_\theta(a_t|s_t) \right]
$$
其中 $\Psi_t$ 可以有许多种形式，最常用的是优势函数 $A^\pi(s_t,a_t)$ ：
$$
V^\pi(s_t)=\mathbb{E}_{s_{t+1:\infty},\ a_{t:\infty}}\left[\sum_{l=0}^\infty r_{t+l} \right] \\
Q^\pi(s_t,a_t)=\mathbb{E}_{s_{t+1:\infty},\ a_{t+1:\infty}}\left[\sum_{l=0}^\infty r_{t+l} \right] \\
A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)
$$
为以上的三个价值函数加上衰减 $\gamma$ ：
$$
V^{\pi,\gamma}(s_t)=\mathbb{E}_{s_{t+1:\infty},\ a_{t:\infty}}\left[\sum_{l=0}^\infty \gamma^l r_{t+l} \right] \\
Q^{\pi,\gamma}(s_t,a_t)=\mathbb{E}_{s_{t+1:\infty},\ a_{t+1:\infty}}\left[\sum_{l=0}^\infty \gamma^l r_{t+l} \right] \\
A^{\pi,\gamma}(s_t,a_t) = Q^{\pi,\gamma}(s_t,a_t) - V^{\pi,\gamma}(s_t) \\
$$
则此时的带衰减的策略梯度为：
$$
g^\gamma = \mathbb{E}_{s_{0:\infty},\ a_{0:\infty}}\left[ \sum_{t=0}^\infty A^{\pi,\gamma}(s_t,a_t) \nabla_\theta \log\pi_\theta(a_t|s_t) \right]
$$

# 优势函数的估计 Advantage Function Estimation

我们用 $\hat{A}_t$ 来表示带衰减的优势函数 $A^{\pi,\gamma}(s_t,a_t)$ 的近似估计，则近似的策略梯度估计为：
$$
\hat{g} = \frac{1}{N} \sum_{n=1}^N \sum_{t=0}^\infty \hat{A}_t^n \nabla_\theta \log\pi_\theta(a_t^n|s_t^n)
$$
其中 $n$ 为批处理的索引。

用 $V$ 来表示价值函数的近似估计，定义 TD 误差 ： $\delta_t^V = r_t + \gamma V(s_{t+1})-V(s_t)$ ，在 [策略梯度 Policy Gradient](https://bluefisher.github.io/2018/05/10/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6-Policy-Gradient/) 中也介绍过，TD 误差可以看作是优势函数的无偏估计，即如果 $V=V^{\pi,\gamma}$ 的话， $\mathbb{E}_{s_{t+1}}\left[ \delta_t^{V^{\pi,\gamma}} \right] = A^{\pi,\gamma}(s_t,a_t)$

仿照 TD(λ) ，我们不仅仅让优势函数往前看一步，而是向前看 $k$ 步，那么我们定义 $\hat{A}_t^{(k)}$ ：
$$
\begin{align*}
\hat{A}_t^{(1)} &:= \delta_t^V \\
\hat{A}_t^{(2)} &:= \delta_t^V + \gamma\delta_{t+1}^V \\
\hat{A}_t^{(3)} &:= \delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V \\
\vdots\\
\hat{A}_t^{(k)} &:=\sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = -V(s_t) + r_t+\gamma r_{t+1} + \cdots + \gamma^{k-1}r_{t+k-1} + \gamma^k V(s_{t+k})
\end{align*}
\newcommand{gae}{\text{GAE}}
$$
接着我们将这 $k$ 步取平均，得到本文最关键的生成式优势函数估计 (generalized advantage estimator) $\gae(\gamma,\lambda)$ ：
$$
\begin{align*}
\hat{A}_t^{\gae(\gamma,\lambda)} &:= (1-\lambda) \left( \hat{A}_t^{(1)}+\lambda\hat{A}_t^{(2)}+\lambda^2\hat{A}_t^{(3)} + \cdots \right) \\
&=\sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V
\end{align*}
$$
可以发现，当 $\lambda=0$ 时，$\hat{A}_t:=\delta_t$ ，此时会引入偏差；当 $\lambda=1$ 时，$\hat{A}_t:=\sum_{l=0}^\infty\gamma^l r_{t+l}-V(s_t)$ ，此时会产生很高的方差，所以为了做到偏差方差平衡，$0<\lambda<1$ 很关键。在实验中，作者发现 $\lambda$ 其实比 $\gamma$ 引入的偏差要小得多。

使用 GAE ，我们构建带有偏差的策略梯度估计：
$$
g^\gamma \approx \mathbb{E}\left[ \sum_{t=0}^\infty \nabla_\theta \log\pi_\theta(a_t|s_t) \hat{A}_t^{\gae(\gamma,\lambda)} \right] =  \mathbb{E}\left[ \sum_{t=0}^\infty \nabla_\theta \log\pi_\theta(a_t|s_t) \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V \right]
$$
论文中，还用 reward shaping 的思路解释了 GAE ，此处就不再展开。

# 值函数估计 Value Function Estimation

上一小节相当于在传统的 actor-critic 算法中，对于 critic 部分，做了一些变化，然而这个 critic 也就是优势函数依然需要值函数的估计，所以我们需要解决以下非线性回归问题：
$$
\underset{\phi}{\text{minimize}} \sum_{n=1}^N ||V_\phi(s_n)-\hat{V}_n||^2
$$
而在解决这个问题的过程中，使用了置信域来进行优化，同时可以帮助避免过拟合。首先计算 $\sigma^2=\frac{1}{N}\sum_{n=1}^N ||V_{\phi_{old}}(s_n)-\hat{V}_n||^2$ ，然后解决以下带约束的优化问题：
$$
\begin{align*}
\underset{\phi}{\text{minimize}} &\quad \sum_{n=1}^N ||V_\phi(s_n)-\hat{V}_n||^2 \\
\text{subject to} &\quad \frac{1}{N}\sum_{n=1}^N \frac{||V_{\phi}(s_n)-V_{\phi_{old}}(s_n)||^2}{2\sigma^2} \le \epsilon
\end{align*}
$$
这个约束就是未优化的值函数与优化过的值函数的 KL 散度不能大于 $\epsilon$ 。可以使用共轭梯度算法来解决这个优化问题，此处也不展开，只给出以下结论：
$$
\begin{align*}
\underset{\phi}{\text{minimize}} &\quad g^T(\phi-\phi_{old}) \\
\text{subject to} &\quad \frac{1}{N}\sum_{n=1}^N (\phi-\phi_{old})^TH(\phi-\phi_{old}) \le \epsilon
\end{align*}
$$
其中 $g$ 是目标函数的梯度，$H=\frac{1}{N}\sum_n j_nj_n^T$ ，$j_n=\nabla_\phi V_\phi(s_n)$

# 参考

Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*. 