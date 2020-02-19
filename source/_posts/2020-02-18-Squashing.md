---
title: Squashing
mathjax: true
typora-root-url: ..
date: 2020-02-18 21:16:18
categories: Reinforcement Learning
tags: 
- rl
- sac
---

一般环境的动作空间是有界的，但由于策略是个分布，如高斯分布，是无界的，所以我们需要对动作进行 squashing 挤压。$\tanh$ 是一个可逆函数，值域在 $(-1,1)$ 之间，非常适合用来达到我们的目的。

换句话说，一个 squashed 后的与状态独立的高斯策略应该是 $\mathbf{a}=\tanh \left(\mathbf{b}_{\phi}(\mathbf{s})+\mathbf{A}_{\phi}(\mathbf{s}) \epsilon\right)$ ，其中 $\epsilon \sim \mathcal{N}(0, I)$ ， $\mathbf{b}$ 是可训练的偏差，$\mathbf{A}$ 是可训练的满秩矩阵，一般来说是对角全为正数的对角矩阵。我们可以将两个函数的转换简写成：$\mathbf{a}=\left(f_{2} \circ f_{1}\right)(\epsilon)$ ，其中 $\mathbf{z}=f_{1}(\epsilon) \triangleq b(\mathbf{s})+A(\mathbf{s}) \epsilon$ ， $\mathbf{a}=f_{2}(\mathbf{z}) \triangleq \tanh (\mathbf{z})$ 。

Soft actor-critic 需要我们计算动作的 log-likelihood 值，因为 $f_1$ 和 $f_2$ 都是可逆的，所以我们可以应用如下定理：对于任意可逆函数 $\mathbf{z}^{(i)}=f_{i}\left(\mathbf{z}^{(i-1)}\right)$ ，我们有

$$
\begin{align}
\mathrm{z}^{(N)}=\left(f_{N} \circ \cdots \circ f_{1}\right)\left(\mathrm{z}^{0}\right) & \Leftrightarrow \\
\log p\left(\mathrm{z}^{(N)}\right) &= \log p\left(\mathrm{z}^{(0)}\right)-\sum_{i=1}^{N} \log \left|\operatorname{det}\left(\frac{d f_{i}\left(\mathrm{z}^{(i-1)}\right)}{d \mathrm{z}^{(i-1)}}\right)\right|
\end{align}
$$

其中 $\frac{d f_{i}(\mathbf{z})}{d \mathbf{z}}$ 是 $f_i$ 的雅可比矩阵。

在实际开发中，对于 $\tanh$ ，雅可比矩阵是对角为 $\frac{d \tanh \left(z_{i}\right)}{d z_{i}}=1-\tanh ^{2}\left(z_{i}\right)$ 的对角矩阵，因此我们有：
$$
\log \left|\operatorname{det}\left(\frac{d f_{2}(\mathbf{z})}{d \mathbf{z}}\right)\right|=\sum_{i=1}^{|\mathcal{A}|} \log \left(1-\tanh ^{2}\left(z_{i}\right)\right)
$$
即：
$$
\pi(\mathbf{a} | \mathbf{s})=\mu(\mathbf{z} | \mathbf{s})\left|\operatorname{det}\left(\frac{\mathrm{da}}{\mathrm{dz}}\right)\right|^{-1}
$$

$$
\log \pi(\mathbf{a} | \mathbf{s})=\log \mu(\mathbf{z} | \mathbf{s})-\sum_{i=1}^{|\mathcal{A}|} \log \left(1-\tanh ^{2}\left(z_{i}\right)\right)
$$