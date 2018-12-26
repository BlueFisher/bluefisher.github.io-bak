---
title: Proximal Policy Optimization 代码实现
mathjax: true
date: 2018-07-06 10:49:07
updated: 2018-07-06 10:49:07
categories:
- Reinforcement Learning
tags:
- RL
- PG
- Python
---

在 [Proximal Policy Optimization Algorithms](https://bluefisher.github.io/2018/07/03/Proximal-Policy-Optimization-Algorithms/) 一文的基础上，可以看出来 PPO 比 TRPO 算法实现起来方便得多，相比于 Actor-Critic 算法，最重要的改动在于把目标函数进行了替换 (surrogate objective) ，同时在更新这个替代的目标函数时对它加上了一定更新幅度的限制。在实际的代码实现中，我们根据论文中的说明，将 Actor 和 Critic 合并起来，共用一套神经网络参数，只用一个损失函数来进行优化。直接看[完整代码](https://github.com/BlueFisher/Reinforcement-Learning/tree/master/Proximal_Policy_Optimization)

<!--more-->

# PPO

最重要的 PPO 类架构为：

```python
class PPO(object):
    def __init__(self, s_dim, a_bound, c1, c2, epsilon, lr, K):
    def _build_net(self, scope, trainable):
    def get_v(self, s):
    def choose_action(self, s):
    def train(self, s, a, discounted_r):
```

## _build_net 神经网络构建函数

```python
with tf.variable_scope(scope):
    l1 = tf.layers.dense(self.s, 100, tf.nn.relu, trainable=trainable)
    mu = tf.layers.dense(l1, 1, tf.nn.tanh, trainable=trainable)
    sigma = tf.layers.dense(l1, 1, tf.nn.softplus, trainable=trainable)

    # 状态价值函数 v 与策略 π 共享同一套神经网络参数
    v = tf.layers.dense(l1, 1, trainable=trainable)

    mu, sigma = mu * self.a_bound, sigma + 1

    norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

params = tf.global_variables(scope)
return norm_dist, v, params
```

## \__init__

首先是一些变量的定义，同时构建神经网络，获取到策略 $\pi$ 与状态价值函数 $V$ ：

```python
self.sess = tf.Session()

self.a_bound = a_bound
self.K = K

self.s = tf.placeholder(tf.float32, shape=(None, s_dim), name='s_t')

pi, self.v, params = self._build_net('network', True)
old_pi, old_v, old_params = self._build_net('old_network', False)
```

然后是一个轨迹采样完成之后得到的衰减过后的奖励，这一步的计算在主函数中事先进行：
$$
r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t+1}  + \gamma^{T-t} V(s_T)
$$

```python
self.discounted_r = tf.placeholder(tf.float32, shape=(None, 1), name='discounted_r')
```

构建优势函数：
$$
\hat{A}_t = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t+1}  + \gamma^{T-t} V(s_T)
$$

```python
advantage = self.discounted_r - old_v
```

最关键的替代的目标函数：
$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[ \min( r_t(\theta)\hat{A}_t, \text{clip}( r_t(\theta),1-\epsilon,1+\epsilon )\hat{A}_t ) \right]
$$

```python
self.a = tf.placeholder(tf.float32, shape=(None, 1), name='a_t')
ratio = pi.prob(self.a) / old_pi.prob(self.a)

L_clip = tf.reduce_mean(tf.minimum(
    ratio * advantage,  # 替代的目标函数 surrogate objective
    tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage
))
```

再与 Critic 结合，同时加上信息熵：
$$
L^{CLIP+VF+S}_t(\theta) = \hat{\mathbb{E}}_t[L^{CLIP}_t(\theta) -c_1L^{VF}_t + c_2S[\pi_\theta](s_t)]
$$

```python
L_vf = tf.reduce_mean(tf.square(self.discounted_r - self.v))
S = tf.reduce_mean(pi.entropy())
L = L_clip - c1 * L_vf + c2 * S
```

至此基本完成整个算法的构造，其余细节可以在完整代码中查看。

# 主函数

```python
def simulate():
    s = env.reset()
    r_sum = 0
    trans = []
    for step in range(T_TIMESTEPS):
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        trans.append([s, a, (r + 8) / 8])
        s = s_
        r_sum += r

    v_s_ = ppo.get_v(s_)
    for tran in trans[::-1]:
        v_s_ = tran[2] + GAMMA * v_s_
        tran[2] = v_s_

    return r_sum, trans


for i_iteration in range(ITER_MAX):
    futures = [executor.submit(simulate) for _ in range(ACTOR_NUM)]
    concurrent.futures.wait(futures)

    trans_with_discounted_r = []
    r_sums = []
    for f in futures:
        r_sum, trans = f.result()
        r_sums.append(r_sum)
        trans_with_discounted_r += trans

    print(i_iteration, r_sums)

    for i in range(0, len(trans_with_discounted_r), BATCH_SIZE):
        batch = trans_with_discounted_r[i:i + BATCH_SIZE]
        s, a, discounted_r = [np.array(e) for e in zip(*trans_with_discounted_r)]
        ppo.train(s, a, discounted_r[:, np.newaxis])
```

根据论文中的说明，我们在采样时，用了多个 Actor 并行地采样，由于规模比较小，采样上的时间差别不大，主要耗时在训练过程中。

# 参考

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*. 

https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py