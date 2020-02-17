---
title: 理解 Variational Lower Bound
mathjax: true
typora-root-url: ..
date: 2020-02-06 19:27:51
categories: Machine Learnine
tags: ml
---

本文翻译自 [Understanding the Variational Lower Bound](http://users.umiacs.umd.edu/~xyang35/files/understanding-variational-lower.pdf), Xitong Yang, September 13, 2017


变分贝叶斯（Variational Bayesian (VB)）是一类非常受欢迎的统计类机器学习方法。VB 非常有用的一个特性是推断优化的二元性：我们可以将统计推断问题（从一个随机变量的值推断出另一种随机变量的值）作为优化问题（找到参变量的值来最小化某些目标函数）。另外，**variational lower bound** ，也被称作 **evidence lower bound** (ELBO)，在 VB 的推导中起了非常重要的作用。在这篇文章中，我们主要介绍有关 variational lower bound 的最基础的知识，有助于理解与 “hard attention” 机制有关的论文。

<!--more-->

# Variational Lower Bound

## 问题设置

假设 $X$ 是观测值（数据），$Z$ 是隐变量。注意隐变量也可能包含参数。这两种变量之间的关系可以用如下图模型表示：

![](/images/2020-02-06-理解-Variational-Lower-Bound/image-20200206195028885.png)

另外，大写的 $P(X)$ 表示某个变量的概率分布，小写的 $p(X)$ 表示 $X$ 分布的概率密度函数。隐变量的后验概率可以用贝叶斯定理写为：
$$
p(Z | X)=\frac{p(X | Z) p(Z)}{p(X)}=\frac{p(X | Z) p(Z)}{\int_{Z} p(X, Z)}
$$

## 第一个推导：琴生不等式

从观测值 $X$ 的 log 概率（$X$ 的边缘分布）出发，我们有：
$$
\begin{align}
\log p(X) &=\log \int_{Z} p(X, Z) \tag{1}\\
&=\log \int_{Z} p(X, Z) \frac{q(Z)}{q(Z)} \tag{2}\\
&=\log \left(\mathbb{E}_{q}\left[\frac{p(X, Z)}{q(Z)}\right]\right) \tag{3}\\
& \geq \mathbb{E}_{q}\left[\log \frac{p(X, Z)}{q(Z)}\right] \tag{4}\\
&=\mathbb{E}_{q}[\log p(X, Z)]+H[Z] \tag{5}
\end{align}
$$
公式 (5) 就是 **variational lower bound**，也被称为 ELBO。

公式 (2) 中的 $q(Z)$ 是我们在 VB 中后验概率 $p(Z|X)$ 的估计概率。这里我们暂且将它视为任意一种分布，推导也依然成立。公式 (4) 对凸函数 log 运用了琴生不等式 $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$ 。$H[Z]=-\mathbb{E}_{q}[\log q(Z)]$ 是香农熵。

我们做如下标记：
$$
L=\mathbb{E}_{q}[\log p(X, Z)]+H[Z]
$$
很明显 $L$ 就是观测变量的 log 概率的一个 lower bound。也就是说，如果我们想要去最大化边缘分布，我们可以转而最大化它的 variational lower bound $L$ 。

## 第二个推导：KL 散度

在上一小节中，我们没有过多关注于分布 $p(Z)$ ，然而 $p(Z)$ 确是 VB 的核心动机。在很多情况下，后验概率 $p(Z|X)$ 的计算是十分困难的，比如我们可能需要对所有的隐变量做积分（求和）来计算分母。

变分方法的主要思想就是找一个估计的概率分布 $q(Z)$ 来尽可能地接近后验概率 $p(Z|X)$ 。这些估计的概率分布可以有它们独有的*变分参数（variational parameters）*：$q(Z|\theta)$ ，所以我们想要去寻找这些参数来使 $q$ 尽可能接近后验概率。当然 $q(Z)$ 的分布肯定要在推断中相对来说更加简单好求一些。

为了衡量两个分布 $q(Z)$ 和 $p(Z|X)$ ，一个常用的标准就是就是 Kullback-Leibler (KL) 散度。变分推断的 KL 散度为：
$$
\begin{align} 
K L[q(Z) \| p(Z | X)] &=\int_{Z} q(Z) \log \frac{q(Z)}{p(Z | X)} \tag{6}\\ 
&=-\int_{Z} q(Z) \log \frac{p(Z | X)}{q(Z)} \tag{7}\\ 
&=-\left(\int_{Z} q(Z) \log \frac{p(X, Z)}{q(Z)}-\int_{Z} q(Z) \log p(X)\right) \tag{8}\\ 
&=-\int_{Z} q(Z) \log \frac{p(X, Z)}{q(Z)}+\log p(X) \int_{Z} q(Z) \tag{9}\\ 
&=-L+\log p(X) \tag{10}
\end{align}
$$
其中 $L$ 就是在上一小节定义的 **variational lower bound** 。公式 (10) 是归一化常量 $\int_Z q(Z)=1$ 而推导得来。重新整理可以得到：
$$
L=\log p(X)-K L[q(Z) \| p(Z | X)]
$$
 因为 KL 散度永远是 $\geq 0$ 的，所以，再一次，我们得到了 $L\leq \log p(X)$ 是观测变量的分布的一个 lower bound。同时我们也知道了它们之间的区别就在于估计分布和真实分布之间的 KL 散度。换句话说，如果估计分布与真实后验分布完美接近，那么 lower bound $L$ 就等于 log 概率。

# 例子

## 视觉注意力中的多目标识别问题

现在，我们可以使用我们对 variational lower bound 的理解来推导一下实际问题中的学习方法。我们使用 *Multiple object recognition with visual attention* 这篇论文来做例子。

首先，我们想要做的是最大化类别标签的 log 似然：$\log p(y|I,W)$ ，其中 $I$ 是图像，$W$ 是模型参数，$y$ 是类别标签，$y$ 也可以被看成是观测变量。目标函数就可以被重写成含有边缘化隐变量 $l$ （对 $l$ 积分）的形式：
$$
\log p(y | I, W)=\log \sum_{l} p(l | I, W) p(y | l, I, W)
$$
我们可以直接使用琴生不等式来得到该目标函数的 lower bound ：
$$
\log \sum_{l} p(l | I, W) p(y | l, I, W) \geq \sum_{l} p(l | I, W) \log p(y | l, I, W)
$$
当然我们也可以用观测变量的边缘分布的 variational lower bound，即设置 $q(l)=p(l|I,W)$ 来得到相同的结果：
$$
\log p(y | I, W) \geq \sum_{l} q(l) \log \frac{p(y, l | I, W)}{q(l)}=\sum_{l} p(l | I, W) \log p(y | l, I, W)
$$