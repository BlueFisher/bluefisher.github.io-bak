---
title: 马尔可夫决策过程(MDP)定义整理
mathjax: true
date: 2018-05-07 09:41:10
updated: 2018-05-07 09:41:10
categories:
- Reinforcement Learning
- Course by David Silver
tags:
- RL
---

马尔可夫决策过程 (Markov decision process, MDP) 对完全可观测的环境进行了正式的描述，也就是说现有的状态完全决定了决策过程中的特征。

几乎所有强化学习的问题都可以转化为MDP，如：

- 针对连续MDP问题的最优决策

- 不完全观测问题也可以转化为MDP

# 马尔可夫过程 Markov Process

## 马尔可夫性 Markov Property

一个状态$S_t$具有马尔可夫性的当且仅当
$$
\mathbb{P}[S_{t+1}|S_t] = \mathbb{P}[S_{t+1}|S_1, \dots , S_t]
$$
一个状态保留了所有历史状态的信息，而一旦一个状态确定了，历史状态就不再重要，也就是说一个状态完全可以决定未来状态

状态转移概率：

$$
\mathcal{P}_{ss'}=\mathbb{P}[S_{t+1}=s'|S_t=s]
$$

状态转移矩阵，每行之和为1
$$
\mathcal{P}=
\begin{bmatrix}
\mathcal{P}_{11} & \cdots & \mathcal{P}_{1n} \\
\vdots \\
\mathcal{P}_{n1} & \cdots & \mathcal{P}_{nn}
 \end{bmatrix}
$$

<!--more-->

## 马尔可夫过程（链） Markov Process (Chain)

$<\mathcal{S}, \mathcal{P}>$

$\mathcal{S}$是有限的状态集，$\mathcal{P}$是状态转义矩阵

# 马尔可夫奖励过程 Markov Reward Process (MRP)

$<\mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma>$

$\mathcal{R}$是奖励函数，$\mathcal{R}_s=\mathbb{E}[R_{t+1}|S_t=s]$

$\gamma$是衰减因子，$\gamma \in [0,1]$

## 回报 Return

从$t$时刻开始，$t+1$时刻的Reward加上$t+2$时刻的Reward乘衰减因子……
$$
G_t=R_{t+1}+\gamma R_{t+2} + \dots = \sum_{k=0}^{\infty}{\gamma^k R_{t+k+1}}
$$

## 价值函数 Value Function

某一状态的长期价值
$$
v(s) = \mathbb{E}[G_t|s_t=s]
$$

## 贝尔曼方程 (Bellman Equation) 形式的MRP

$$
\begin{align*}
 v(s) &= \mathbb{E}[G_t|s_t=s] \\
 &= \mathbb{E}[R_{t+1} + \gamma v(S_{t+1})|s_t=s] \\
 &= \mathcal{R}_s + \gamma \sum_{s' \in \mathcal{S}}{\mathcal{P}_{ss'}v(s')}
\end{align*}
$$

矩阵形式的贝尔曼方程：

$$
\begin{align*}
    v &= \mathcal{R}+\gamma \mathcal{P} v \\
    &= 
    \begin{bmatrix}
        {v(1)} \\
        \vdots \\
        {v(n)}
    \end{bmatrix} 
    = 
    \begin{bmatrix}
        \mathcal{R}_1\\
        \vdots \\
        \mathcal{R}_n
    \end{bmatrix}
    +
    \begin{bmatrix}
        \mathcal{P}_{11} & \cdots & \mathcal{P}_{1n} \\
        \vdots \\
        \mathcal{P}_{n1} & \cdots & \mathcal{P}_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    {v(1)} \\
    \vdots \\
    {v(n)}
    \end{bmatrix} 
\end{align*}
$$

# 马尔可夫决策过程 Markov Decision Process (MDP)

$<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma>$
$\mathcal{S}$是有限的状态集
$\mathcal{A}$是有限的动作集
$\mathcal{P}_{ss'}^a=\mathbb{P}[S_{t+1}=s'|S_t=s, A_t=a]$
$\mathcal{R}_s^a=\mathbb{E}[R_{t+1}|S_t=s, A_t=a]$
$\gamma$是衰减因子，$\gamma \in [0,1]$

## 策略 Policy

$$
\pi(a|s) = \mathbb{P}[A_t=a | S_t=s]
$$

策略完整定义了智能体的动作行为，MDP策略仅依赖于当前状态而不是历史状态，也就是说策略是静态的，与时间无关，$A_t \sim \pi(\cdot|S_t), \forall t>0$

给定一个MDP，$\mathcal{M}=<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma>$和一个策略$\pi$

状态序列$S_1, S_2, \cdots$是一个马尔可夫过程$<\mathcal{S}, \mathcal{P}^\pi>$

状态奖励序列$S_1, R_2, S_2, \cdots$是一个马尔可夫奖励过程$<\mathcal{S}, \mathcal{P}^\pi, \mathcal{R}^\pi, \gamma>$

其中：
$$
\begin{align*}
    \mathcal{P}_{s,s'}^\pi &= \sum_{a \in \mathcal{A}}{\pi(a|s)\mathcal{P}_{ss'}^a} \\
    \mathcal{R}_{s}^\pi &= \sum_{a \in \mathcal{A}}{\pi(a|s)\mathcal{R}_{s}^a}
\end{align*}
$$

## 价值函数 Value Function

状态价值函数 state-value function

$$
v_\pi(s)=\mathbb{E}_\pi[G_t|S_t=s]
$$

行为价值函数 action-value function

$$
q_\pi(s,a)=\mathbb{E}_\pi[G_t|S_t=s,A_t=a]
$$

## 贝尔曼期望方程 Bellman Expectation Equation
$$
\begin{align*}
v_\pi(s) &= \mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s] \\
         &= \sum_{a\in \mathcal{A}}{\pi(a|s)q_\pi(s,a)} \\
         &= \sum_{a\in \mathcal{A}} {\pi(a|s)
         \left(
            \mathcal{R}_s^a+\gamma \sum_{s'\in \mathcal{S}}{P_{ss'}^a{v_\pi(s')}}
         \right)}
\end{align*}
$$
$$
\begin{align*}
q_\pi(s,a) &= \mathbb{E}_\pi[R_{t+1} + \gamma q_\pi(S_{t+1},A_{t+1})|S_t=s,A_t=a] \\
&= \mathcal{R}_s^a + \gamma \sum_{s'\in \mathcal{S}}{\mathcal{P}_{ss'}^a v_\pi(s')} \\
&= \mathcal{R}_s^a + \gamma \sum_{s'\in \mathcal{S}}{\mathcal{P}_{ss'}^a \sum_{a'\in \mathcal{A}}{\pi(a'|s')q_\pi(s',a')}}
\end{align*}
$$

## 最优价值函数 Optimal Value Function
最优状态价值函数
$$
v_*(s) = \max_\pi v_\pi(s)
$$
最优行为价值函数
$$
q_*(s,a) = \max_\pi q_\pi(s,a)
$$

## 贝尔曼最优方程 Bellman Optimality Equation
$$
\begin{align*}
v_*(s) &= \max_a q_*(s,a) \\
&= \max_a \mathcal{R}_s^a + \gamma \sum_{s'\in \mathcal{S}}{\mathcal{P}_{ss'}^a v_*(s')} \\
\end{align*}
$$
$$
\begin{align*}
q_*(s,a) &= \mathcal{R}_s^a + \gamma \sum_{s'\in \mathcal{S}}{\mathcal{P}_{ss'}^a v_*(s')} \\
&= \mathcal{R}_s^a + \gamma \sum_{s'\in \mathcal{S}}{\mathcal{P}_{ss'}^a \max_a' q_*(s', a')}
\end{align*}
$$

# 参考
<http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf>

<https://zhuanlan.zhihu.com/p/28084942>
