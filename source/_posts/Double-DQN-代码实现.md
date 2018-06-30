---
title: Double DQN & 代码实现
mathjax: true
date: 2018-05-21
updated: 2018-05-21
categories:
- Reinforcement Learning
tags:
- RL
- python
---

在 *Deep Reinforcement Learning with Double Q-Learning* 论文中，Google DeepMind 将 *Double Q-learning* 论文中提到的 Double Q-Learning 与 DQN 相结合，在 DQN 的基础上做了改进，提出了 Double DQN 算法。本文也在 [Deep Q-Network](https://bluefisher.github.io/2018/05/07/Deep-Q-Network/) 和 [DQN 代码实现](https://bluefisher.github.io/2018/05/08/DQN-%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0/) 两篇博文的基础上介绍 Double DQN 同时实现代码 。

<!--more-->

# Double DQN

Q-learning 的更新为：
$$
\theta_{t+1} = \theta_t + \alpha(Y^Q_t-Q(S_t,A_t;\theta_t))\nabla_{\theta_t}Q(S_t,A_t;\theta_t)
$$
其中 $Y_t^Q$ 为：
$$
Y_t^Q \equiv R_{t+1} + \gamma \max_a Q(S_{t+1},a;\theta_t)
$$
在 DQN 中，由于设置了另一个 target 网络，参数为 $\theta^-$，所以：
$$
Y_t^{DQN} \equiv R_{t+1} + \gamma \max_a Q(S_{t+1},a; \theta_t^-)
$$
在 Q-learning 和 DQN 中的 `max` 操作同时用来进行选择并且评估一个行为，这很有可能去选择一个过估计 (overestimated) 的值，导致值函数过优化 (overoptimistic)

而 Double DQN 则对 $Y_t^{DQN}$ 进行了一个改动：
$$
Y_t^{DoubleQ} \equiv R_{t+1} + \gamma Q(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta_t);\theta^-_t)
$$
产生过优化的原因在于在估计时的误差，即使这个估计是无偏的，但在使用 `max` 操作时会一步步扩大误差。论文中的大致意思是，假设某个状态的所有最优行为价值等于状态价值 $Q_*(s,a) = V_*(s)$ ，令 $Q_t$ 为任意的无偏的行为价值函数的近似估计，即 $\sum_a(Q_t(s,a-V_*(s))=0$ ，当然仍有误差 $\frac{1}{m}\sum_a(Q_t(s,a-V_*(s))^2 = C$ ，其中 $m$为当前状态的行为总数且 $C>0,\,m\ge2$ ，那么在这些条件下， $\max_a Q_t(s,a) \ge V_*(s) + \sqrt{\frac{C}{m-1}}$ ，也就是说采用 `max` 操作以后会有一个固定的误差下界，但使用 Double DQN 后的估计误差下界为 0（证明使用了反证法）。尽管使用 Double DQN 会减少过优化问题，但也有可能产生欠估计 (underestimated)

# 代码改进

与 DQN 比较， Double DQN 只改动了一处 直接看 [所有代码](https://github.com/BlueFisher/Reinforcement-Learning/tree/master/Deep_Q_Network/Double_DQN)

DQN：

```python
# Y = R + gamma * max(Q^)
q_target = self.r + self.gamma \
    * tf.reduce_max(self.q_target_z, axis=1, keepdims=True) * (1 - self.done)
```

Double DQN：

```python
 # argmax(Q)
max_a = tf.argmax(self.q_eval_z, axis=1)
one_hot_max_a = tf.one_hot(max_a, self.a_dim)

# Y = R + gamma * Q^(S, argmax(Q))
q_target = self.r + self.gamma \
    * tf.reduce_sum(one_hot_max_a * self.q_target_z, axis=1, keepdims=True) * (1 - self.done)
q_target = tf.stop_gradient(q_target)
```

# 参考

Van Hasselt, H., Guez, A., & Silver, D. (2016, February). Deep Reinforcement Learning with Double Q-Learning. In *AAAI* (Vol. 16, pp. 2094-2100).