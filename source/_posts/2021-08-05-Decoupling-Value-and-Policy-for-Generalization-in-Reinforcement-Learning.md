---
title: Decoupling Value and Policy for Generalization in Reinforcement Learning
mathjax: true
typora-root-url: ..
date: 2021-08-05 20:28:22
categories:
tags:
---

本文主要针对强化学习中的泛化问题，提出了要分离价值网络与策略网络，来解决一个共享的表征导致的过拟合表征问题和价值函数估计不准确的问题，同时利用对抗网络，来鼓励学习一种与任务无关的表征，从这两点提高强化学习的泛化性。

<!--more-->

作者通过一个 Ninja 游戏中的两个关卡表明共享策略与价值函数可能会造成的问题：

![](/images/2021-08-05-Decoupling-Value-and-Policy-for-Generalization-in-Reinforcement-Learning/image-20210805221015802.png)

关卡1与关卡2的初始观测值从语义上来说都是相同的，包括障碍物、智能体的位置，但是图像上不一样，关卡1的背景是黑色的，关卡2的背景是蓝色的，且关卡1的最大步长会比关卡2来的短。假设现在已经通过 PPO 算法在这两个关卡上训练并得到了最优策略，由于有 discounted reward 的存在，且两个关卡的长度不一样，所以训练出的最优状态价值 $V$​​ 对于第一个观测值来说是不同的，很明显由于关卡1长度小，$V_1(s_0)>V_2(s_0)$​​ 。

但尽管如此，理论上第一帧对应的语义信息是完全相同的，也就意味着理论上第一帧的价值是相同的，因为无论第一帧的颜色如何变化，但是语义没有改变，对应的最优动作应该是不变的。

但正是因为真正学到的最优价值函数不同，导致策略与价值函数共享的表征网络必然要学习到除了智能体位置、障碍物位置等语义信息以外的一点额外东西来区分两种不同关卡的第一帧，只有这样才能让价值函数的输出不一样。而就是这点额外的东西，可能是背景的颜色，导致了过拟合的问题，然事实上背景颜色是不应该影响状态价值的。

这样训练出来的策略尽管能很好地完成这两个关卡，但是在其他新的关卡中泛化性就会降低。所以作者提出了 Invariant Decoupled Advantage Actor-Critic (IDAAC)。

# Invariant Decoupled Advantage Actor-Critic
首先一个简单的思想就是分离策略与价值函数。但如果只是在 PPO 的基础上把原本共享的表征层拆分，那么策略网络的梯度接受不到来自于价值函数的梯度。如果只是单纯的靠策略目标函数来更新策略网络，不引入价值梯度，也就是不引入任何任务的奖励信息，策略网络训练会不好。

因此作者在分离的策略网络的输出中，又分离输出优势函数：

![](/images/2021-08-05-Decoupling-Value-and-Policy-for-Generalization-in-Reinforcement-Learning/image-20210806095841009.png)

现在策略网络既输出动作概率，也输出状态动作对的优势，策略的目标函数为：
$$
J_{\mathrm{DAAC}}(\theta)=J_{\pi}(\theta)+\alpha_{\mathrm{s}} \mathrm{S}_{\pi}(\theta)-\alpha_{\mathrm{a}} L_{\mathrm{A}}(\theta)
$$
其中 $J_\pi(\theta)$ 是常规的 PPO 目标函数
$$
J_{\pi}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]
$$
$S_\pi(\theta)$ 为策略熵的正则项，$L_A(\theta)$​ 为优势函数损失：
$$
L_{\mathrm{A}}(\theta)=\hat{\mathbb{E}}_{t}\left[\left(A_{\theta}\left(s_{t}, a_{t}\right)-\hat{A}_{t}\right)^{2}\right]
$$
其中 $\hat{A}_t$​​ 是 标准GAE，由价值网络产生，即为： $\hat{A}_{t}=\sum_{k=t}^{T}(\gamma \lambda)^{k-t} \big(r_{t}+\gamma V_{\phi}\left(s_{t+1}\right)-V_{\phi}\left(s_{t}\right)\big)$​​ 。​​

价值网络损失就是标准的均方误差：
$$
L_{\mathrm{V}}(\phi)=\hat{\mathbb{E}}_{t}\left[\left(V_{\phi}\left(s_{t}\right)-\hat{V}_{t}\right)^{2}\right]
$$

# Learning Instance-Invariant Features

从泛化性的角度看，一个好的策略表征是要学习到最少的特征来做出最优的决策。为了避免 Ninja 游戏中出现的智能体可能会去记住每个任务对应的总 episode 长度的信息，作者使用了一个对抗网络，让判别器无法分辨某两帧图像哪个在前哪个在后，即强制让策略只依靠学习到的特征来决策，而不是判断当前的 step。

作者借鉴了视频表征中的方法，$E_\theta$​ 为表征模型，将观测 $s$​ 输入编码为向量 $f$​ 。那么判别器 $D_\psi$​ 以同一 episode 中的两帧的表征 $f_i,\  f_j$​​ 为输入（有前后顺序），输出一个 0 到 1 的值表示该两帧图像是否是 $s_i$ 在前 $s_j$ 在后。 判别器在论文中的损失是交叉熵损失：
$$
\begin{aligned}
L_{\mathrm{D}}(\psi)=&-\log \left[\mathrm{D}_{\psi}\left(\mathrm{E}_{\theta}\left(s_{i}\right), \mathrm{E}_{\theta}\left(s_{j}\right)\right)\right] \\
&-\log \left[1-\mathrm{D}_{\psi}\left(\mathrm{E}_{\theta}\left(s_{i}\right), \mathrm{E}_{\theta}\left(s_{j}\right)\right)\right]
\end{aligned}
$$
但该公式可能没有表述清楚，后一项应该为 $-\log \left[1-\mathrm{D}_{\psi}\left(\mathrm{E}_{\theta}\left(s_{j}\right), \mathrm{E}_{\theta}\left(s_{i}\right)\right)\right]$​ 会更好。

而生成器也就是表征模型的 $E_\theta$ 的损失即不让判别器分辨出来，尽可能达到 50% 的概率：

$$
\begin{aligned}
L_{\mathrm{E}}(\theta)=&-\frac{1}{2} \log \left[\mathrm{D}_{\psi}\left(\mathrm{E}_{\theta}\left(s_{i}\right), \mathrm{E}_{\theta}\left(s_{j}\right)\right)\right] \\
&-\frac{1}{2} \log \left[1-\mathrm{D}_{\psi}\left(\mathrm{E}_{\theta}\left(s_{i}\right), \mathrm{E}_{\theta}\left(s_{j}\right)\right)\right]
\end{aligned}
$$
![](/images/2021-08-05-Decoupling-Value-and-Policy-for-Generalization-in-Reinforcement-Learning/image-20210806101643407.png)

此时策略网络的目标函数需要加上生成器的损失项：
$$
J_{\mathrm{IDAAC}}(\theta)=J_{\mathrm{DAAC}}(\theta)-\alpha_{i} L_{\mathrm{E}}(\theta)
$$
整个 IDAAC 的算法伪代码为：

![](/images/2021-08-05-Decoupling-Value-and-Policy-for-Generalization-in-Reinforcement-Learning/image-20210806101819621.png)

