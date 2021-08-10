---
title: >-
  Learning Invariant Representations For ReinForcement Learning Without
  Reconstruction
mathjax: false
typora-root-url: ..
date: 2021-08-08 10:03:07
categories:
tags: 
---

本文的基本思想是要通过表征学习，将图像的观测输入映射为向量化状态，再交由强化学习训练。但该表征学习不需要额外的领域知识，也不是 reconstruction 重建式的表征，它是通过衡量两个状态之间的 bisimulation metrics 来学习到一种不包含无关信息、对任务没有任何帮助的表征。

<!--more-->

目前很多的表征方法都是通过 reconstruction，将隐状态重建恢复为原始观测值，达到表征的目的，这种 autoencoders 的表征尽管可以将原始图像降维至向量给强化学习智能体训练，但是表征的隐状态会包含许多与任务无关的特征，从而干扰强化学习训练。如下图中的自动驾驶例子，蓝色原始图像中的房子、云和树对于汽车的驾驶任务来说没有关系，在隐状态空间中应该映射为同一个点，而橙色图片中的汽车这一关键元素与蓝色图片相比是不一样的，那么对应的在隐状态空间中，应该映射至不同的点。

![image-20210808151256670](/images/2021-08-08-Learning-Invariant-Representations-For-ReinForcement-Learning-Without-Reconstruction/image-20210808151256670.png)

那么如何衡量两个不一样的原始状态是否在隐状态空间中是相同的呢？作者借用了 Bisimulation 来衡量状态的等效性：



若两个状态 $s_i,\ s_j$ 在执行动作 $a$ 后得到的奖励相同，并且转移到的下一个状态之间也是等效的，那么就称 $s_i,\ s_j$​ 等效。

当然因为状态动作都为连续的，只是根据定义通过数值上的相等来判断等效是不可行的，所以作者根据奖励之间的距离与状态转移概率分布之间的距离来衡量状态之间的等效程度，即 Bisimulation Metric：



其中 $c\in [0,1)$，$W_1$​ 表示两个状态转移概率分布的 Wasserstein 距离。

那么对于强化学习中的表征模型而言，两个原始状态观测 $s_i,\ s_j$ 经过表征后的隐状态 $z_i=\phi(s_i),\ z_j=\phi(s_j)$​​ 的距离要尽量与该两个隐状态的 Bisimulation Metric 相似。所以构造如下的均方差损失函数：



其中 $\bar{z}$ 表示不传梯度的 $\phi(s)$ ，因为这个损失是要让隐状态的距离更能反应 Bisimulation Metric，所以此时的 Bisimulation Metric不传梯度。另外作者对于奖励的距离直接使用环境所返回的奖励来表示，对于下一状态的转移概率分布，作者使用了一个高斯分布来近似，而两个转移概率分布之间的 Wasserstein 距离，作者则使用了 2-Wasserstein 距离，因为此时两个高斯分布的 2-Wasserstein 有个非常好的性质可以直接计算：



下图为 encoder 部分的网络架构

![image-20210808212841730](/images/2021-08-08-Learning-Invariant-Representations-For-ReinForcement-Learning-Without-Reconstruction/image-20210808212841730.png)

首先 encoder 的输出即向量化的隐状态 $\phi(s)$ 作为标准强化学习的输入进行训练，如 SAC、DQN等。

然后 endoer 本身则使用上述的损失来进行更新，注意上图网络结构中的 Reward Model 不参与 encoder 的训练，因为根据损失函数，只采用真时交互的 reward 数据，损失中的状态转移概率分布即为图中 Dynamics Model 的输出，为一高斯分布，同时因为损失中是要比较两个状态之间距离，在如何生成这两个状态的问题上，作者简单的将从经验池中取到的一个 batch 数据复制一份并打乱，再将打乱前与打乱后的状态之间一一比较，构造损失，从而更新 encoder。

最后通过下一个 step 的隐状态与真实奖励，以一种监督学习的方式，更新 Reward Model 与 Dynamics Model，注意这里的 Reward Model 仅仅是辅助训练 Dynamics Model，效果会更好。

下列算法的 7、8、9 行即为上述关键的三步。

![image-20210808223219390](/images/2021-08-08-Learning-Invariant-Representations-For-ReinForcement-Learning-Without-Reconstruction/image-20210808223219390.png)



