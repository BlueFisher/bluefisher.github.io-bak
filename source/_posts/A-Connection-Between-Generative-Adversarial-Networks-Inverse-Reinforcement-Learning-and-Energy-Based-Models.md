---
title: >-
  A Connection Between Generative Adversarial Networks, Inverse Reinforcement
  Learning, and Energy-Based Models
mathjax: true
date: 2018-11-12 15:00:56
categories: Reinforcement Learning
tags: 
- RL
- IRL
---

之前在学习 Berkeley 的 CS 294: Deep Reinforcement Learning 课程时，对于逆强化学习 (inverse reinforcement learning IRL) 中的最大熵逆强化学习 (MaxEnt) 有点云里雾里，可能一开始受 [Maximum Entropy Inverse Reinforcement Learning](http://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf) 和 [Maximum entropy deep inverse reinforcement learning](https://arxiv.org/abs/1507.04888) 两篇论文的影响，基于概率图模型，使用了逆最优控制问题 (inverse optimal control IOC) 方法，使得整个算法的推导、求解十分复杂，以至于到后来的 [Guided cost learning: Deep inverse optimal control via policy optimization](http://www.jmlr.org/proceedings/papers/v48/finn16.pdf) 论文就更是不知道在说什么。

然而 [A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models](https://arxiv.org/abs/1611.03852) 这篇论文将前面这些方法结合起来并与生成式对抗网络 (generative adversarial networks GAN) 做了对比，比较详细的说明了这些基于 MaxEnt 的 IRL 算法到底在做一件什么事。本文也简单介绍一下这篇论文。

<!--more-->

# Background

## Generative Adversarial Networks

GAN 是一种比较新的生成模型，一般被用来学习一种很难去表示的概率分布，当这种概率模型很容易就能表示时，我们直接用极大似然估计就可以了。

它同时学习生成器 (generator) $G$ 和判别器 (discriminator) $D$。Discriminator 用来分类，判断它的输入是来自 generator 还是采样自真实的 data distribution $p(x)$ 。Generator 的目标是尽量让 discriminator 将 generator 输出的样本分类到来自于真实分布。也就是说 Generator 接受一个噪音输入，输出样本 $x\sim G$ ，discriminator 接受一个样本 $x$ 输入，输出该样本是否来自于真实分布的概率 $D(x)$ 。

Discriminator 的损失函数为：
$$
\newcommand{\L}{\mathcal{L}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\dis}{\text{discriminator}}
\newcommand{\gen}{\text{generator}}
\newcommand{\cost}{\text{cost}}
\newcommand{\samp}{\text{sampler}}
\newcommand{\p}{\frac{1}{Z}\exp(-c_\theta(\tau))}
\L_{\dis}(D)=\E_{x\sim p}[-\log D(x)]+\E_{x\sim G}[-\log (1-D(x))]
$$

如果样本来自于真实分布 $p$ ，则增加 $D(x)$ 概率，如果来自 generator 则减少概率。

Generator 的损失函数为：
$$
\L_{\gen}(G)=\E_{x\sim G}[-\log D(x)]+\E_{x\sim G}[\log(1-D(x))]
$$
Generator 想要混淆 discriminator ，使得判别它生成出来的样本时，尽量提高 $D(x)$ 概率。

## Energy-Based Models

Energy-based models 与 softmax 有点相似，它将样本 $x$ 用能量函数 (energy function) 封装了下 $E_\theta(x)$ ，用 Bolzmann distribution 来建模：
$$
p_\theta(x)=\frac{1}{Z}e^{-E_\theta(x)}
$$
Energy function 的参数 $\theta$ 即是用来将样本的似然值最大化。看似很简单用极大似然估计就可以求解，然而在优化中最大的问题是这个归一化的配分函数 partition function $Z$ ，在绝大多数高维环境问题中， partition function 不可能进行求和或者求积。一些比较常用的用来估计 $Z$ 的方法需要在学习时内部循环采样，比如使用 Markov chain Monte Carlo (MCMC) 方法，但当分布 $p_\theta(x)$ 有很多不同的模型时，MCMC 方法就需要花大量时间来进行采样。还有一种 approximate infernce 方法，但如果 energy function 很小时，则该方法可能无法捕捉到它们。

## Inverse Reinfocement Learning

IRL 或者说 IOC 的目标就是根据示例行为 (demonstrated behavior) 找出 cost function ，一般假设这些示例来自于在某个未知 cost function 下的接近最优的专家行为。由于 IRL 最早来自于最优控制论，所以习惯于用 cost function 而不是 reward function ，用 $x,u$ 来表示强化学习中的状态行为 $s,a$ 。在论文中，作者主要讨论了 MaxEnt IRL 和 guided cost learning 的 MaxEnt IRL 算法。

## Maximum entropy inverse reinforcement learning

MaxEnt IRL 用 Boltzmann distribution 建模了 demonstrations 的分布，energy function 由 cost function $c_\theta$ 得到：
$$
p_\theta(\tau)=\frac{1}{Z} \exp(-c_\theta(\tau))
$$
其中 $\tau=\{x_1,u_1,\cdots,x_T,u_T\}$ ，$c_\theta(\tau)=\sum_tc_\theta(x_t,u_t)$ ，partition function $Z$ 是从动态环境得来的所有轨迹的 $\exp(-c_\theta(\tau))$ 的积分。

在这个模型下，最优轨迹应该有着最高的似然概率，而专家可以根据一定概率生成次最优轨迹，如果这个生成的轨迹导致 cost function 很大，则这个概率应该指数级地减小。要求得 $\theta$ 来最大化 demonstrations 的似然值就像上文 energy-based models 说的一样，看起来比较简单用极大似然估计就可以，但最大的难点在于估计 partition function ，berkeley 的课程中详细介绍过 [Maximum Entropy Inverse Reinforcement Learning](http://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf) 的解决方法，然而这种方法的局限非常大，只在离散的比较小的状态空间并且环境模型已知的情况下才可行。

## Guided cost learning

Guided cost learning 提出了一种基于采样的迭代方法来估计 $Z$ ，并且可以在高维的状态、行为空间和非线性 cost function 的情况下解决问题。该算法训练一个新的采样分布 $q(\tau)$ 并且用重要性采样来估计 $Z$ ：
$$
\begin{align*}
\L_\cost(\theta) &= \E_{\tau\sim p}[-\log p_\theta(\tau)]=\E_{\tau\sim p}[c_\theta(\tau)]+\log Z \\
&= \E_{\tau\sim p}[c_\theta(\tau)] + \log\left( \E_{\tau\sim q} \left[ \frac{\exp(-c_\theta(\tau))}{q(\tau)} \right] \right) \\
&= \frac{1}{N}\sum_{\tau_i\in\mathcal{D}_\text{demo}}c_\theta(\tau_i) + \log\frac{1}{M}\sum_{\tau_j\in\mathcal{D}_\text{sample}} \frac{\exp(-c_\theta(\tau_j))}{q(\tau_j)}
\end{align*}
$$

$$
\frac{d\L_\cost(\theta)}{d\theta}=\frac{1}{N}\sum_{\tau_i\in\mathcal{D}_\text{demo}}\frac{dc_\theta(\tau_i)}{d\theta}-\frac{1}{\sum_j w_j}\sum_{\tau_j\in\mathcal{D}_\text{sample}} w_j\frac{dc_\theta(\tau_i)}{d\theta} \\\quad \text{where } w_j=\frac{exp(-c_\theta(\tau_j))}{q(\tau_j)}
$$

可以看出 guided cost learning 算法既优化 cost function 同时也要优化 $q(\tau)$ 来减少重要性的采样带来的方差问题。

对于 partition function $\int\exp(-c_\theta(\tau))$ 的最优重要性采样分布为 $q(\tau) \propto |\exp(-c_\theta(\tau))|=\exp(-c_\theta(\tau))$ ，在学习过程中， $q(\tau)$ 应该要尽量靠近 $p(\tau)$ 分布，所以作者使用了最小化 $q(\tau)$ 和 $\exp(-c_\theta(\tau))$ 之间 KL 散度的方法：
$$
\L_\samp(q)=-\int q(\tau)\log\frac{\exp(-c_\theta(\tau))}{q(\tau)}d\tau
$$
这个公式也可以写成等价的最小化 cost function 和最大化熵的形式：
$$
\L_\samp(q)=\E_{\tau\sim q}[c_\theta(\tau)]+\E_{\tau\sim q}[\log q(\tau)] \tag{1}
$$
理论上来说，这个最优化的采样分布就应该是最优 cost function 下的 demonstration distribution 。因此，guided cost learning 在训练过程既学习 cost function 来表示 demonstration distribution ，同时也要学习策略 $q(\tau)$ 以此来生成从 demonstration distribution 中产生的样本。

重要性采样的估计方法可能会产生高方差的问题，作者用了一种混合 data 样本与 generated 样本的方法：令 $\mu=\frac{1}{2}p+\frac{1}{2}q$ 为混合分布，同时令 $\tilde{p}(\tau)$ 为一个比较粗略的 demonstrations 的分布（比如当前的模型 $p_\theta$ 或用另一种方法训练一个更简单的模型），guided cost learning 现在用 $\mu$ 来做重要性采样：
$$
\L_\cost(\theta)=\E_{\tau\sim p}[c_\theta(\tau)] + \log\left( \E_{\tau\sim \mu} \left[ \frac{\exp(-c_\theta(\tau))}{\frac{1}{2}\tilde{p}(\tau)+\frac{1}{2}q(\tau)} \right] \right)
$$

# GANs 与 IRL

## 一个特殊形式的 discriminator

对于一个固定的 generator 和一个一般未知的概率分布 $q(\tau)$ ，最优 discriminator 为：
$$
D^*(\tau)=\frac{p(\tau)}{p(\tau)+q(\tau)} \tag{2}
$$
其中 $p(\tau)$ 为真实数据的概率分布。

传统的 GAN 算法训练直接输出这个值。当 generator 的概率密度可以被衡量时，我们可以修改一下传统的 GAN ，我们不再去直接估计公式 (1) ，而是去估计 $p(\tau)$ ，在这种情况下，新的 discriminator $D_\theta$ 为：
$$
D_\theta(\tau)=\frac{\tilde{p}(\tau)}{\tilde{p}(\tau)+q(\tau)}
$$
现在我们与 MaxEnt IRL 关联起来，用 Boltzmann distribution 代替这个估计的真实数据的概率分布：
$$
D_\theta(\tau)=\frac{\p}{\p)+q(\tau)}
$$
可以发现最优 discriminator 与 generator 完全独立，也就是说当 discriminator 是最优时，$\frac{1}{Z}\exp(-c_\theta(\tau))=p(\tau)$ ，这可以使训练更加稳定。

## GAN 与 guided cost learning 的等价性

我们将 GAN 与 MaxEnt IRL 的几个公式结合起来再写一遍：

Discriminator 的损失函数为：
$$
\begin{align*}
\L_{\dis}(D_\theta) &= \E_{\tau\sim p}[-\log D_\theta(\tau)]+\E_{\tau\sim q}[-\log (1-D_\theta(\tau))] \\
&= \E_{\tau\sim p}\left[-\log \frac{\p}{\p+q(\tau)}\right]+\E_{\tau\sim q}\left[-\log \frac{q(\tau)}{\p+q(\tau)}\right]
\end{align*}
$$

MaxEnt IRL 的 log 似然函数为：
$$
\begin{align*}
\L_\cost(\theta) &= \E_{\tau\sim p}[c_\theta(\tau)] + \log\left( \E_{\tau\sim \frac{1}{2}p+\frac{1}{2}q} \left[ \frac{\exp(-c_\theta(\tau))}{\frac{1}{2}\tilde{p}(\tau)+\frac{1}{2}q(\tau)} \right] \right) \tag{3} \\
&= \E_{\tau\sim p}[c_\theta(\tau)] + \log\left( \E_{\tau\sim \mu} \left[ \frac{\exp(-c_\theta(\tau))}{\frac{1}{2Z}\exp(-c_\theta(\tau))+\frac{1}{2}q(\tau)} \right] \right)
\end{align*}
$$
注意这里我们使用了 $\tilde{p}(\tau)=p_\theta(\tau)=\p$ ，也就是说用当前的模型来估计重要性权重。

我们重新记 $\tilde{\mu}(\tau)=\frac{1}{2Z}\exp(-c_\theta(\tau))+\frac{1}{2}q(\tau)$ ，注意到当 $\theta$ 与 $Z$ 为最优时， $\p$ 即为 $p(\tau)$ 的最优估计，所以 $\tilde{\mu}(\tau)$ 为 $\mu$ 的最优估计。

接下来作者通过三个等式，说明了 GAN 与 MaxEnt IRL 实际上做了同一件事情，本文不做过多的公式推导，只展示结果。

### 1. $Z$ 为 partition function 的估计

我们用 $\L_\dis$ 对 partition function $Z$ 求偏导：
$$
\begin{align*}
\partial_Z\L_\dis(D_\theta)&=0 \\
Z&=\E_{\tau\sim\mu}\left[ \frac{\exp(-c_\theta(\tau))}{\tilde{\mu}(\tau)} \right]
\end{align*}
$$
也就是说最小化的 $Z$ 实际上就是公式 (3) 中的最小化的重要性采样估计。

### 2. $c_\theta$ 优化 IRL 的目标函数

我们现在根据上一结论的 $Z$ 对 $\theta$ 求偏导，可以得到：
$$
\partial_\theta\L_\cost(\theta) = \partial_\theta\L_\dis(D_\theta)
$$
可以看出 MaxEnt IRL 的 discriminator 的损失函数与 GAN 的目标函数对 $\theta$ 的偏导相等。

### 3. Generator 优化 MaxEnt IRL 的目标函数

$$
\L_\gen(q) = \log Z+\L_\samp(q)
$$

$\log Z$ 这一项为 discriminator 的参数，在更新 generator 时这项时固定不动的，所以 generator 的损失函数与公式 (1) 中 sampler 的损失函数相等。

结合以上三点，我们就可以把 GAN 看作是一种基于采样的解决 MaxEnt IRL 问题的算法。

总结一下 IRL 里的轨迹 $\tau$ 相当于 GAN 中的样本 $x$ ，采样策略 $q(\tau)$ 相当于 generator $G$ ，cost function $c$ 相当于 discriminator $D$ 。

# 用 GANs 的方法来训练 EBMs

有了以上 GAN 与 IRL 的基础，我们可以直接将 GAN 与 EBM 联系起来。实际上已经有一些论文 [Deep directed generative models with energy-based probability estimation](https://arxiv.org/abs/1606.03439)、[Energy-based generative adversarial network](https://arxiv.org/abs/1609.03126) 在尝试用 GAN 的方法解决 EBM 的 partition function 问题。这些方法都是交替地训练 generator 来产生小能量样本 $E_\theta(x)$ 还要用样本来优化 energy function 的参数来估计 partition function 。

与上文一样，我们可以推导出无偏的 partition function 估计为：
$$
Z=\E_{x\sim\mu}\left[ \frac{\exp(-E_\theta(x))}{\frac{1}{2}\tilde{p}(x)+\frac{1}{2}q(x)} \right]
$$
所以损失函数为：
$$
\begin{align*}
\L_{\text{energy}}(\theta)&=\E_{x\sim p}[-\log p_\theta(x)] \\
&=\E_{x\sim p}[-E_\theta(x)]-\log\left( \E_{x\sim\mu}\left[ \frac{\exp(-E_\theta(x))}{\frac{1}{2}\tilde{p}(x)+\frac{1}{2}q(x)} \right] \right)
\end{align*}
$$
而 generator 则最小化 energy function 并且最大化熵：
$$
\L_\gen(q)=\E_{x\sim q}[E_\theta(x)]+\E_{x\sim q}[\log q(x)]
$$

# 参考

Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008, July). Maximum Entropy Inverse Reinforcement Learning. In *AAAI* (Vol. 8, pp. 1433-1438).

Wulfmeier, M., Ondruska, P., & Posner, I. (2015). Maximum entropy deep inverse reinforcement learning. *arXiv preprint arXiv:1507.04888*.

Finn, C., Levine, S., & Abbeel, P. (2016, June). Guided cost learning: Deep inverse optimal control via policy optimization. In *International Conference on Machine Learning* (pp. 49-58).

Finn, C., Christiano, P., Abbeel, P., & Levine, S. (2016). A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. *arXiv preprint arXiv:1611.03852*.

http://rail.eecs.berkeley.edu/deeprlcourse-fa17/index.html