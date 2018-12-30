---
title: Clipped Action Policy Gradient
mathjax: true
date: 2018-12-26 19:09:23
categories: Reinforcement Learning
tags: RL
---

许多连续行为空间的任务往往都是有边界的行为空间。在 policy gradient 中，如果策略的行为输出超出了边界的话，会在真正做决策之前将行为进行截断 (clip) ，使它控制在边界中，但在策略更新的过程中，其实并不知道策略的输出被截断了。在论文 [Clipped Action Policy Gradient](https://arxiv.org/pdf/1802.07564) 中，作者提出了一种截断行为并且无偏的能减小方差的方法，称之为 clipped action policy gradient (CAPG) 。

<!--more-->

# Preliminaries

强化学习的目标是找到一个策略来最大化累积期望回报：
$$
\newcommand{\E}{\mathbb{E}}
\newcommand{\V}{\mathbb{V}}
\newcommand{\low}{\alpha}
\newcommand{\high}{\beta}
\newcommand{\R}{\mathbb{R}}
\newcommand{\clip}{\mathrm{clip}}
\newcommand{\I}{\mathbb{I}}
\newcommand{\capgpsi}{\bar{\psi}}
\eta(\pi) = \E_{s_0,u_0,\dots} \Big[ \sum_t \gamma^t r(s_t,u_t) \Big| \pi \Big]
$$
其中 $u$ 为行为。policy gradient 的公式为：
$$
\nabla_\theta \eta(\pi_\theta) = \E_s \Big[ \E_u [Q^{\pi_\theta}(s,u) \psi(s,u)|s] \Big]
$$
其中 $\psi(s,u) = \nabla_\theta \log \pi_\theta(u|s)$ 。用蒙特卡洛来估计这个梯度：
$$
\nabla_\theta \eta(\pi_\theta) \approx \frac{1}{N} \sum_i Q^{\pi_\theta}(s^{(i)},u^{(i)}) \psi(s^{(i)},u^{(i)})
$$
尽管蒙特卡洛方法是个无偏估计，但策略梯度最大的问题在于高方差，而本文主要就是解决这个问题。

我们先定义一个随机变量 $Y$ ，使得 $\V[Y] \le \V[X]$ 并且 $\E[Y] = \E[X]$ ，其中 $\V$ 为方差，$\E$ 为期望，$X=Q^{\pi_\theta}(s,u) \psi(s,u)$ 。因为
$$
\begin{align*}
\E[X] &= \E_s[\E_u[X|s]] \tag{1}\\
\V[X] &= \V_s[\E_u[X|s]] + \E_s[\V_u[X|s]] \tag{2}
\end{align*}
$$
这里要提下第二个公式，在文中没有给出证明，实际上非常简单，意思就是假设现在有一个二元密度函数 $p(x,y)$ ，它的期望与方差为：
$$
\begin{align*}
\E[p] &= \E_x[\E_y[p]]\\
\V[p] &= \E[p^2]-[E[p]]^2 = \V_x[\E_y[p]] + \E_x[\V_y[p]]
\end{align*}
$$
可以手动地推导一下，方差中的等式是成立的。

回到刚才的公式 (1) (2) ，这个时候我们可以非常容易地看出：
$$
\begin{align*}
\E_u[Y|s] &= \E_u[X|s] \\
\V_u[Y|s] &\le \V_u[X|s]  
\end{align*}
$$


# Clipped Action Policy Gradient

我们现在考虑智能体的行为被截断成 $[\low, \high] \subset \R^d$ ，也就是说状态转移概率和奖励函数为：
$$
\begin{align*}
  P(s' | s, u) &= P(s' | s,\clip(u, \low, \high)) \\
  r(s, u)     &= r(s, \clip(u, \low, \high))
\end{align*}
$$

## Scalar actions

我们先考虑行为空间为标量的情况，即 $d=1$ 。$Q$ 函数满足：
$$
\begin{align*}
Q^{\pi_\theta}(s,u)
&= Q^{\pi_\theta}(s,\clip(u,\low,\high)) \\
&=
\begin{cases}
    Q^{\pi_\theta}(s,\low) & \text{if } u \le \low\\
    Q^{\pi_\theta}(s,u)    & \text{if } \low < u < \high\\
    Q^{\pi_\theta}(s,\high)& \text{if } \high \le u
\end{cases} \tag{3}
\end{align*}
$$
令 $X$ 为依赖行为 $u$ 的随机变量，$\I_{f(u)}$ 为依据 $f(u)$ 作为条件的指示函数。因为 $X = \I_{u \le \low} X + \I_{\low < u < \high} X + \I_{\high \le u} X$ ，那么 $\E_u[X]$ 可以展开成：
$$
\E_u [X] = \E_u [\I_{u \le \low} X] + \E_u [\I_{\low < u < \high} X] + \E_u [\I_{\high \le u} X] \tag{4}
$$
结合公式 (3) (4) ，我们可以得到：
$$
\begin{align*}
\E_u[Q^{\pi_\theta}(s,u) \psi(s,u)] &= Q^{\pi_\theta}(s,\low) \E_u [\I_{u \le \low}  \nabla_\theta \log \pi_\theta(u|s)] \\
&+\E_u [\I_{\low < u < \high} Q^{\pi_\theta}(s,u) \nabla_\theta \log \pi_\theta(u|s)] \\
&+Q^{\pi_\theta}(s,\high) \E_u [\I_{\high \le u} \nabla_\theta \log \pi_\theta(u|s)].
\end{align*} \tag{5}
$$
同时，如下推论 (1) 成立：

>假设 $\pi_\theta(u|s)$ 是 $u\in\R$ 的 条件 PDF ，其 CDF 为 $\Pi_\theta(u|s)$ ，那么以下等式成立：
>$$
>\begin{align*}
>\E_u[\I_{u \le \low} \nabla_\theta \log \pi_\theta(u|s)]  &= \E_u [\I_{u \le \low} \nabla_\theta \log \Pi_\theta(\low|s)],\\
>\E_u[\I_{\high \le u} \nabla_\theta \log \pi_\theta(u|s)] &= \E_u [\I_{\high \le u} \nabla_\theta \log (1 - \Pi_\theta(\high|s))].
>\end{align*}
>$$
>

证明（只证明第一个式子，第二个式子证法相同）：
$$
\begin{align*}
\E_u[\I_{u \le \low} \nabla_\theta \log \pi_\theta(u|s)] \nonumber
  &= \int_{-\infty}^\low \pi_\theta(u|s) \nabla_\theta \log \pi_\theta(u|s)du \nonumber
\\&= \int_{-\infty}^\low \nabla_\theta \pi_\theta(u|s)du \nonumber
\\&= \nabla_\theta \int_{-\infty}^\low \pi_\theta(u|s)du \nonumber
\\&= \nabla_\theta \Pi_\theta(\low|s) \nonumber
\\&= \Pi_\theta(\low|s) \nabla_\theta \log \Pi_\theta(\low|s) \nonumber
\\&= \E_u[\I_{u \le \low} \nabla_\theta \log \Pi_\theta(\low|s)]. \nonumber
\end{align*}
$$
将这个推论应用到公式 (5) 中，得到：
$$
\begin{align*}
\E_u[Q^{\pi_\theta}(s,u) \psi(s,u)]&=Q^{\pi_\theta}(s,\low) \E_u [\I_{u \le \low} \nabla_\theta \log \Pi_\theta(\low|s)]\\
  &{}+\E_u [\I_{\low < u < \high} Q^{\pi_\theta}(s,u)\nabla_\theta \log \pi_\theta(u|s)]\\
  &{}+Q^{\pi_\theta}(s,\high) \E_u [\I_{\high \le u}\nabla_\theta \log \left(1- \Pi_\theta(\high|s)\right)]\\
&=\E_u [Q^{\pi_\theta}(s,u) \capgpsi(s,u)] \tag{6}
\end{align*}
$$
其中：
$$
\capgpsi(s,u) =
\begin{cases}
    \nabla_\theta \log \Pi_\theta(\low|s)      & \text{if } u \le \low\\
    \nabla_\theta \log \pi_\theta(u|s)         & \text{if } \low < u < \high\\
    \nabla_\theta \log (1-\Pi_\theta(\high|s)) & \text{if } \high \le u
\end{cases}
$$
根据公式 (9) ，我们可以用 policy gradient 中的估计方法来采样估计 $Q^{\pi_\theta}(s,u) \capgpsi(s,u)$ ，我们就称它为 clipped action policy gradient (CAPG) ，CAPG 的估计比标准的 policy gradient 估计更加好，因为无偏且有着更小的方差。

从直觉上来看，因为 $\Pi_\theta(\low|s)$ 和 $1-\Pi_\theta(\high|s)$ 对于状态 $s$ 来说都是确定性的，所以方差会减小。从数学上来说，令 $X$ 为只与 $u$ 有关的随机变量，则方差为：
$$
\begin{align*}
\V_u[X] ={} &\V_u [\I_{u \le \low} X]\! + \!\V_u [\I_{\low < u < \high} X]\! + \!\V_u [\I_{\high \le u} X]\\
 &{}-2\E_u [\I_{u \le \low} X] \E_u [\I_{\low < u < \high} X]\\
 &{}-2\E_u [\I_{\low < u < \high} X] \E_u [\I_{\high \le u} X]\\
 &{}-2\E_u [\I_{\high \le u} X] \E_u [\I_{u \le \low} X].
\end{align*}
$$
我们将 $X=Q^{\pi_\theta}(s,u)\psi(s,u)$ 与 $X=Q^{\pi_\theta}(s,u) \capgpsi(s,u)$ 两个式子进行比较，从上面的推论 (1) 可以看出，方差与 $\V_u [\I_{\low < u < \high} X]$ ， $\E_u [\I_{u \le \low} X], \E_u [\I_{\low < u < \high} X]$ ， $\E_u [\I_{\high \le u} X]$ 都没关系，只与 $\V_u [\I_{u \le \low} X]$ 和 $\V_u [\I_{\high \le u} X]$ 有关。我们可以得到以下推论 (2)：

>假设 $\pi_\theta(u|s)$ 是 $u\in\R$ 的 条件 PDF ，其 CDF 为 $\Pi_\theta(u|s)$ ，那么以下不等式成立：
>$$
>\begin{align*}
>\V_u[\I_{u \le \low} \nabla_\theta \log \pi_\theta(u|s)]
>&\geq \V_u[\I_{u \le \low} \nabla_\theta \log \Pi_\theta(\low|s)] \\
>\V_u[\I_{\high \le u} \nabla_\theta \log \pi_\theta(u|s)]
>&\geq \V_u[\I_{\high \le u} \nabla_\theta \log (1-\Pi_\theta(\high|s))].
>\end{align*}
>$$
>

证明可见论文的附录。同样的，如果乘以一个实值函数 $f(s,u)$ ：
$$
f(s,u) =
\begin{cases}
    f(s,\low) & \text{if } u \le \low\\
    f(s,u)    & \text{if } \low < u < \high\\
    f(s,\high)& \text{if } \high \le u
\end{cases}.
$$
一下等式与不等式仍然成立：
$$
\begin{align*}
  \E_u[f(s,u)\capgpsi(s,u)] =   \E_u[f(s,u)\psi(s,u)],\\
  \V_u[f(s,u)\capgpsi(s,u)] \le \V_u[f(s,u)\psi(s,u)].
\end{align*}
$$
至此，我们已经推导出了在 preliminaries 中所设的 $X$ 与 $Y$ ，其中 $Y=Q^{\pi_\theta}(s,u) \capgpsi(s,u)$ ， $X=Q^{\pi_\theta}(s,u) \psi(s,u)$ 。

## Vector actions

之前一小节对于标量的行为空间可以扩展到向量级别的行为空间，即 $\vec{u} \in \R^d, d \ge 2$ ，但每一个行为之间必须互相独立。令：
$$
\capgpsi^{(i)}(s,u) =
\begin{cases}
    \nabla_\theta \log \Pi_\theta^{(i)}(\low|s)      & \text{if } u \le \low\\
    \nabla_\theta \log \pi_\theta^{(i)}(u|s)         & \text{if } \low < u < \high\\
    \nabla_\theta \log (1-\Pi_\theta^{(i)}(\high|s)) & \text{if } \high \le u
\end{cases}
$$

则以下等式、不等式仍然成立：

$$
\begin{align*}
  \E_{\vec{u}}[f(s,\vec{u})\capgpsi(s,\vec{u})] =   \E_{\vec{u}}[f(s,\vec{u})\psi(s,\vec{u})] \\
  \V_{\vec{u}}[f(s,\vec{u})\capgpsi(s,\vec{u})] \le \V_{\vec{u}}[f(s,\vec{u})\psi(s,\vec{u})]
\end{align*}
$$

# 参考

Fujita, Y., & Maeda, S. I. (2018). Clipped Action Policy Gradient. *arXiv preprint arXiv:1802.07564*.