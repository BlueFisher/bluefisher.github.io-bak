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

强化学习的目标是找到一个策略来最大化累积期望汇报：
$$
\newcommand{\low}{\alpha}
\newcommand{\high}{\beta}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\mathbb{V}}
\newcommand{\V}{\mathbb{V}}
\newcommand{\Cov}{\mathrm{Cov}}
\newcommand{\States}{\mathcal{S}}
\newcommand{\Actions}{\mathcal{U}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\argmax}{\mathrm{argmax}}
\newcommand{\clip}{\mathrm{clip}}
\newcommand{\pitheta}{{\pi_\theta}}
\newcommand{\Borel}{\mathcal{B}}

\eta(\pi) = \E_{s_0,u_0,\dots} \Big[ \sum_t \gamma^t r(s_t,u_t) \Big| \pi \Big],
$$
其中 $u$ 为行为。policy gradient 的公式为：
$$
\nabla_\theta \eta(\pi_\theta) = \E_s \Big[ \E_u [Q^{\pi_\theta}(s,u) \psi(s,u)|s] \Big],
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

