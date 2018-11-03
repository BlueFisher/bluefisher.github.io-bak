---
title: Actor-Critic Softmax & Gaussian Policy 代码实现
mathjax: false
date: 2018-05-10 22:44:04
updated: 2018-05-11 21:34:04
categories:
- Reinforcement Learning
tags:
- RL
- python
- PG
---

在 [策略梯度 Policy Gradient](https://bluefisher.github.io/2018/05/10/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6-Policy-Gradient/) 一文的理论基础上，实践了一下基于离散行为 Softmax Policy 与基于连续行为 Gaussian Policy 的 Actor-Critic 算法。[直接看所有代码](https://github.com/BlueFisher/Reinforcement-Learning/tree/master/Actor_Critic)

程序框架很简单，就为 Actor 和 Critic 两个类：

```python
class Actor(object):
    def __init__(self, sess, s_dim, a_dim, lr):

    def choose_action(self, s):  # 根据softmax所输出的概率选择行为

    def learn(self, s, a, td_error):

class Critic(object):
    def __init__(self, sess, s_dim, gamma, lr):
        
    def learn(self, s, r, s_):
```

<!--more-->

# Softmax Policy

## Actor

```python
class Actor(object):
    def __init__(self, sess, s_dim, a_dim, lr):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.s = tf.placeholder(tf.float32, shape=(1, s_dim), name='s')
        self.a = tf.placeholder(tf.float32, shape=(a_dim,), name='a')
        self.td_error = tf.placeholder(tf.float32, shape=(), name='td_error')

        l = tf.layers.dense(
            self.s, a_dim, **initializer_helper
        )

        self.a_prob_z = tf.nn.softmax(l)  # 每个行为所对应的概率

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.squeeze(l), labels=self.a) * self.td_error
        # 与以下形式相同，这里用softmax交叉熵代替
        # loss = tf.reduce_sum(-tf.log(self.a_prob_z) * self.a) * self.td_error
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    def choose_action(self, s):  # 根据softmax所输出的概率选择行为
        a_prob_z = self.sess.run(self.a_prob_z, feed_dict={
            self.s: s[np.newaxis, :]
        })

        action = np.random.choice(range(self.a_dim), p=a_prob_z.ravel())
        return action

    def learn(self, s, a, td_error):
        one_hot_action = np.zeros(self.a_dim)
        one_hot_action[a] = 1
        self.sess.run(self.optimizer, feed_dict={
            self.s: s[np.newaxis, :],
            self.a: one_hot_action,
            self.td_error: td_error
        })
```

## Critic

```python
class Critic(object):
    def __init__(self, sess, s_dim, gamma, lr):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, shape=(1, s_dim), name='s')
        self.v_ = tf.placeholder(tf.float32, shape=(), name='v_')
        self.r = tf.placeholder(tf.float32, shape=(), name='r')

        l = tf.layers.dense(
            inputs=self.s, units=30,
            activation=tf.nn.relu, **initializer_helper
        )
        self.v = tf.layers.dense(
            inputs=l, units=1, **initializer_helper
        )

        self.td_error = self.r + gamma * self.v_ - self.v
        loss = tf.square(self.td_error)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.v, {
            self.s: s_[np.newaxis, :]
        })

        td_error, _ = self.sess.run([self.td_error, self.train_op], {
            self.s: s[np.newaxis, :],
            self.v_: v_.squeeze(),
            self.r: r
        })

        return td_error.squeeze()
```

## 主函数

```python
env = gym.make('CartPole-v0')
env = env.unwrapped

with tf.Session() as sess:
    actor = Actor(
        sess=sess,
        s_dim=4,
        a_dim=2,
        lr=0.01
    )

    critic = Critic(
        sess=sess,
        s_dim=4,
        gamma=0.99,
        lr=0.001
    )

    tf.global_variables_initializer().run()

    for i_episode in range(10000):
        s = env.reset()
        n_step = 0
        while True:
            a = actor.choose_action(s)
            s_, r, done, _ = env.step(a)

            if done:
                r = -20

            td_error = critic.learn(s, r, s_)
            actor.learn(s, a, td_error)

            n_step += 1
            if done:
                print(i_episode, n_step)
                break

            s = s_
```

# Gaussian Policy

# Actor

```python
class Actor(object):
    def __init__(self, sess, s_dim, a_bound, lr):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, shape=(1, s_dim), name='s')
        self.a = tf.placeholder(tf.float32, shape=(), name='a')
        self.td_error = tf.placeholder(tf.float32, shape=(), name='td_error')

        l1 = tf.layers.dense(inputs=self.s, units=30, activation=tf.nn.relu, **initializer_helper)

        # 均值
        mu = tf.layers.dense(inputs=l1, units=1, activation=tf.nn.tanh, **initializer_helper)
        # 方差
        sigma = tf.layers.dense(inputs=l1, units=1, activation=tf.nn.softplus, **initializer_helper)

        # 均值控制在(-2, 2) 方差控制在(0, 2)
        mu, sigma = tf.squeeze(mu * a_bound), tf.squeeze(sigma + 1)

        self.normal_dist = tf.distributions.Normal(mu, sigma)
        self.action = tf.clip_by_value(self.normal_dist.sample(1), -a_bound, a_bound)

        loss = self.normal_dist.log_prob(self.a) * self.td_error

        # 最大化 J，即最小化 -loss
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(-loss)

    def learn(self, s, a, td_error):
        self.sess.run(self.optimizer, feed_dict={
            self.s: s[np.newaxis, :],
            self.a: a,
            self.td_error: td_error
        })

    def choose_action(self, s):
        return self.sess.run(self.action, {
            self.s: s[np.newaxis, :]
        }).squeeze()
```

## 主函数

```python
env = gym.make('Pendulum-v0')
env = env.unwrapped

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

with tf.Session() as sess:
    actor = Actor(
        sess,
        state_dim,
        action_bound,
        0.001
    )
    critic = Critic(
        sess,
        state_dim,
        0.9,
        0.001
    )

    tf.global_variables_initializer().run()

    for i_episode in range(1000):
        s = env.reset()

        ep_reward = 0
        for j in range(200):
            # env.render()
            a = actor.choose_action(s)
            s_, r, done, _ = env.step([a])
            r /= 10
            ep_reward += r

            td_error = critic.learn(s, r, s_)
            actor.learn(s, a, td_error)
            if j == 200 - 1:
                print(i_episode, int(ep_reward))
                break

            s = s_
```

---

由于 Critic 在更新时难以收敛，导致 Actor 更新时更难收敛，所以以上两个程序都无法收敛，会呈现震荡的状态。

# 参考

<https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/run_CartPole.py>

<https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20Actor%20Critic%20Solution.ipynb>

<https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb>