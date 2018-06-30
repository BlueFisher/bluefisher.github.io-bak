---
title: DDPG 代码实现
mathjax: false
date: 2018-05-17 00:00:01
updated: 2018-05-17 00:00:01
categories:
- Reinforcement Learning
tags:
- RL
- python
---

根据 [Deep Deterministic Policy Gradient](https://bluefisher.github.io/2018/05/16/Deep-Deterministic-Policy-Gradient/) ，尽管 DPPG 算法的思路简单，就是将 DPG 与 DQN 的几个特性结合起来，但因为设置了4个神经网络，之间还因求导链式法则而相互关联，所以程序设计稍微复杂一点。[直接看所有代码](https://github.com/BlueFisher/Reinforcement-Learning/tree/master/Deep_Deterministic_Policy_Gradient)。

整体框架：

```python
class Memory(object):

class Actor(object):
    def __init__(self, sess, s_dim, a_bound, lr, tau):

    def _build_net(self, s, scope, trainable):
    # 选择确定性行为
    def choose_action(self, s):
    # 根据链式法则，生成 Actor 的梯度
    def generate_gradients(self, Q_a_gradients):
        
    def learn(self, s):


class Critic(object):
    def __init__(self, sess, s_dim, s, s_, a, a_, gamma, lr, tau):

    def _build_net(self, s, a, scope, trainable):
    # 生成 Q 对 a 的导数，交给 actor
    def get_gradients(self):

    def learn(self, s, a, r, s_):
```

<!--more-->

# Memory

首先是最简单的状态行为轨迹的记忆库，与 DQN 中的相同：

```python
class Memory(object):
    def __init__(self, batch_size, max_size):
        self.batch_size = batch_size
        self.max_size = max_size
        self.isFull = False
        self._transition_store = collections.deque()

    def store_transition(self, s, a, r, s_):
        if len(self._transition_store) == self.max_size:
            self._transition_store.popleft()

        self._transition_store.append((s, a, r, s_))
        if len(self._transition_store) == self.max_size:
            self.isFull = True

    def get_mini_batches(self):
        n_sample = self.batch_size if len(self._transition_store) >= self.batch_size else len(self._transition_store)
        t = random.sample(self._transition_store, k=n_sample)
        t = list(zip(*t))

        return tuple(np.array(e) for e in t)
```

# Critic

critic 相对简单一点，与 DQN 算法基本相同，目标是最小化 $\mathbb{E}\left[ ( Q(s_t,a_t|\theta^Q) - y_t )^2 \right]$ ，在最小化时，可以阻断也可以不阻断 $y_t$ 处的梯度，效果都差不多。

```python
class Critic(object):
    def __init__(self, sess, s_dim, s, s_, a, a_, gamma, lr, tau):
        self.sess = sess
        self.s_dim = s_dim
        self.s = s
        self.s_ = s_
        self.a = a

        with tf.variable_scope('critic'):
            self.r = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
            self.q = self._build_net(s, a, 'eval', True)
            self.q_ = self._build_net(s_, a_, 'target', False)

        param_eval = tf.global_variables('critic/eval')
        param_target = tf.global_variables('critic/target')
        # soft update
        self.target_replace_ops = [tf.assign(t, tau * e + (1 - tau) * t)
                                   for t, e in zip(param_target, param_eval)]

        # y_t
        target_q = self.r + gamma * self.q_
        # 可以保留或忽略 target_q 的梯度
        target_q = tf.stop_gradient(target_q)

        loss = tf.reduce_mean(tf.squared_difference(target_q, self.q))
        self.train_ops = tf.train.AdamOptimizer(lr).minimize(loss, var_list=param_eval)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            # ls = tf.layers.dense(
            #     s, 30, name='layer_s', trainable=trainable, **initializer_helper
            # )
            # la = tf.layers.dense(
            #     a, 30, name='layer_a', trainable=trainable, **initializer_helper
            # )
            # l = tf.nn.relu(ls + la)

            l = tf.concat([s, a], 1)
            l = tf.layers.dense(l, 30, activation=tf.nn.relu, trainable=trainable, **initializer_helper)

            with tf.variable_scope('Q'):
                q = tf.layers.dense(l, 1, name='q', trainable=trainable, **initializer_helper)
        return q

    # 生成 Q 对 a 的导数，交给 actor
    def get_gradients(self):
        return tf.gradients(self.q, self.a)[0]

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_ops, {
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
        })
        self.sess.run(self.target_replace_ops)
```

这里尝试了两种神经网络架构，效果都不错。

# Actor

actor 最关键的一步就是 `generate_gradients` 函数，它先计算 $\nabla_{\theta^\mu} \mu(s|\theta^\mu)$ 在根据链式法则获取 critic 传来的 $\nabla_a Q(s,a|\theta^Q)$ 并相乘，得到最终的梯度。

```python
class Actor(object):
    def __init__(self, sess, s_dim, a_bound, lr, tau):
        self.sess = sess 
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.lr = lr

        with tf.variable_scope('actor'):
            self.s = tf.placeholder(tf.float32, shape=(None, s_dim), name='state')
            self.s_ = tf.placeholder(tf.float32, shape=(None, s_dim), name='state_')

            self.a = self._build_net(self.s, 'eval', True)
            self.a_ = self._build_net(self.s_, 'target', False)

        self.param_eval = tf.global_variables('actor/eval')
        self.param_target = tf.global_variables('actor/target')

        # soft update
        self.target_replace_ops = [tf.assign(t, tau * e + (1 - tau) * t) for t, e in zip(self.param_target, self.param_eval)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(
                s, 30, activation=tf.nn.relu,
                name='layer', trainable=trainable, **initializer_helper
            )
            with tf.variable_scope('action'):
                a = tf.layers.dense(
                    l, 1, activation=tf.nn.tanh,
                    name='action', trainable=trainable, **initializer_helper
                )
                a = a * self.a_bound
        return a

    def choose_action(self, s):
        a = self.sess.run(self.a, {
            self.s: s[np.newaxis, :]
        })

        return a[0]

    def generate_gradients(self, Q_a_gradients):
        # 根据链式法则，生成 Actor 的梯度
        grads = tf.gradients(self.a, self.param_eval, Q_a_gradients)
        optimizer = tf.train.AdamOptimizer(-self.lr)
        self.train_ops = optimizer.apply_gradients(zip(grads, self.param_eval))

    def learn(self, s):
        self.sess.run(self.train_ops, {
            self.s: s
        })
        self.sess.run(self.target_replace_ops)
```

# 主函数

在主函数中可以体现出异策略的形式，即在计算 $a_t$ 时，是在确定性策略的基础上增加了噪声来形成随机的异策略探索。

```python
env = gym.make('Pendulum-v0')
env = env.unwrapped

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

var = 3.

with tf.Session() as sess:
    memory = Memory(32, 10000)
    actor = Actor(sess, state_dim, action_bound, lr=0.01, tau=0.01)
    critic = Critic(sess, state_dim, actor.s, actor.s_, actor.a, actor.a_, gamma=0.9, lr=0.001, tau=0.01)
    t = critic.get_gradients()

    actor.generate_gradients(t)

    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        s = env.reset()
        r_episode = 0
        for j in range(200):
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), -action_bound, action_bound)  # 异策略探索
            s_, r, done, info = env.step(a)

            memory.store_transition(s, a, [r / 10], s_)

            if memory.isFull:
                var *= 0.9995
                b_s, b_a, b_r, b_s_ = memory.get_mini_batches()
                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            r_episode += r
            s = s_

            if(j == 200 - 1):
                print('episode {}\treward {:.2f}\tvar {:.2f}'.format(i, r_episode, var))
                break
```

# 参考

<https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG>