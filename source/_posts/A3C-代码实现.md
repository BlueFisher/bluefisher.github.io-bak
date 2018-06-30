---
title: A3C 代码实现
mathjax: false
date: 2018-05-18
updated: 2018-05-18
categories:
- Reinforcement Learning
tags:
- RL
- python
---

在 [Asynchronous Methods for Deep Reinforcement Learning](https://bluefisher.github.io/2018/05/17/Asynchronous-Methods-for-Deep-Reinforcement-Learning/) 一文中，将异步的强化学习框架套给了四种强化学习算法，我们主要实现了最后一种 Asynchronous Advantage Actor-Critic (A3C) ，用来解决连续行为空间的问题。[直接看所有代码](https://github.com/BlueFisher/Reinforcement-Learning/blob/master/Actor_Critic/a3c.py)

整体框架：

```python
# 全局网络
class Global_net(object):
    def __init__(self, sess, s_dim, a_bound, gamma, actor_lr, critic_lr, 
                 max_global_ep, max_ep_steps, update_iter):
    
    # 与子线程共用的构建神经网络函数
    def build_net(self, scope):


# 独立线程，actor critic 合并
class Worker_net(object):
    def __init__(self, global_net, name):
       
    def _choose_action(self, s):

    # 从全局下载参数替换子线程中的参数
    def _sync(self):

    # 在子线程中进行学习，并将子线程的参数更新到全局
    def _update(self, done, transition):

    # 子线程模拟自己独有的环境
    def run(self):
```

<!--more-->

# 全局网络

全局网络比较简单，只是生成了 actor 和 critic 两个神经网络，另外保存了一些全局参数。

```python
class Global_net(object):
    def __init__(self, sess, s_dim, a_bound, gamma, actor_lr, 
                 critic_lr, max_global_ep, max_ep_steps, update_iter):
        self.sess = sess
        self.a_bound = a_bound
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.global_ep = 0
        self.max_global_ep = max_global_ep
        self.max_ep_steps = max_ep_steps
        self.update_iter = update_iter

        self.s = tf.placeholder(tf.float32, shape=(None, s_dim), name='s')

        *_, self.a_params, self.c_params = self.build_net('global')

    def build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope(scope):
            with tf.variable_scope('actor'):
                l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
                mu = tf.layers.dense(l_a, 1, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_a, 1, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
            with tf.variable_scope('critic'):
                l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')

            mu, sigma, v = tf.squeeze(mu * self.a_bound), tf.squeeze(sigma + 1e-4), tf.squeeze(v)

            a_params = tf.global_variables(scope + '/actor')
            c_params = tf.global_variables(scope + '/critic')
            return mu, sigma, v, a_params, c_params
```

# 子线程

子线程稍微复杂点，不仅需要生成自己独立的两个神经网络，同时还要独立地进行环境模拟并与全局网络进行更新、同步。

## 构造函数 & 行为选择函数

```python
def __init__(self, global_net, name):
    self.g = global_net
    self.name = name

    self.a = tf.placeholder(tf.float32, shape=(None,), name='a')
    self.R = tf.placeholder(tf.float32, shape=(None,), name='R')

    mu, sigma, self.v, self.a_params, self.c_params = global_net.build_net(name)

    td = self.R - self.v
    critic_loss = tf.reduce_mean(tf.square(td))

    self.normal_dist = tf.distributions.Normal(mu, sigma)
    obj = self.normal_dist.log_prob(self.a) * tf.stop_gradient(td)
    # 加上策略的熵增加探索空间，避免过早进入局部最优
    obj = obj + self.normal_dist.entropy()
    actor_loss = -tf.reduce_mean(self.normal_dist.log_prob(self.a) * tf.stop_gradient(td))

    self._choose_a_ops = tf.squeeze(tf.clip_by_value(self.normal_dist.sample(1),
                                                        -global_net.a_bound, global_net.a_bound))

    self.a_grads = tf.gradients(actor_loss, self.a_params)
    self.c_grads = tf.gradients(critic_loss, self.c_params)

    # 用自己的梯度来更新全局参数
    actor_optimizer = tf.train.RMSPropOptimizer(self.g.actor_lr)
    critic_optimizer = tf.train.RMSPropOptimizer(self.g.critic_lr)
    self.update_a_op = actor_optimizer.apply_gradients(zip(self.a_grads, self.g.a_params))
    self.update_c_op = critic_optimizer.apply_gradients(zip(self.c_grads, self.g.c_params))

    self.sync_a_ops = [tf.assign(l, g) for l, g in zip(self.a_params, self.g.a_params)]
    self.sync_c_ops = [tf.assign(l, g) for l, g in zip(self.c_params, self.g.c_params)]

def _choose_action(self, s):
    return self.g.sess.run(self._choose_a_ops, {
        self.g.s: s[np.newaxis, :]
    })
```

在 tensorfllow 中，我们不需要在子线程中累积梯度，然后再更新到全局网络中，直接用 `apply_gradients` 函数，将梯度更新到全局参数即可。

## 与全局网络的交互

```python
def _sync(self):
    self.g.sess.run(self.sync_a_ops)
    self.g.sess.run(self.sync_c_ops)

# 在子线程中进行学习，并将子线程的参数更新到全局
def _update(self, done, transition):
    if done:
        R = 0
    else:
        s_ = transition[-1][2]
        R = self.g.sess.run(self.v, {
            self.g.s: s_[np.newaxis, :]
        }).squeeze()

    buffer_s, buffer_a, _, buffer_r = zip(*transition)
    buffer_R = []
    for r in buffer_r[::-1]:
        R = r + self.g.gamma * R
        buffer_R.append(R)

    buffer_R.reverse()

    buffer_s, buffer_a, buffer_R = np.vstack(buffer_s), np.array(buffer_a), np.array(buffer_R)

    self.g.sess.run([self.update_a_op, self.update_c_op], {
        self.g.s: buffer_s,
        self.a: buffer_a,
        self.R: buffer_R
    })
```

## 环境模拟

```python
def run(self):
    env = gym.make('Pendulum-v0')
    env = env.unwrapped

    self._sync()
    total_step = 1
    transition = []

    while self.g.global_ep <= self.g.max_global_ep:
        s = env.reset()
        ep_rewards = 0
        for ep_i in range(self.g.max_ep_steps):
            a = self._choose_action(s)
            s_, r, *_ = env.step([a])
            done = ep_i == self.g.max_ep_steps - 1

            ep_rewards += r
            transition.append((s, a, s_, r / 10))

            if total_step % self.g.update_iter == 0 or done:
                self._update(done, transition)
                self._sync()
                transition = []

            s = s_
            total_step += 1

        self.g.global_ep += 1
        print(self.g.global_ep, self.name, int(ep_rewards))
```

与普通的 Actor-Critic 算法基本一样，只不过此处的环境是在每个子线程中模拟的，再每隔一段时间更新到全局中。

# 主函数

```python
sess = tf.Session()


global_net = Global_net(
    sess=sess,
    s_dim=3,
    a_bound=2,
    gamma=0.9,
    actor_lr=0.0001,
    critic_lr=0.001,
    max_global_ep=1000,
    max_ep_steps=200,
    update_iter=10
)

THREAD_N = 4

workers = [Worker_net(global_net, 'w' + str(i)) for i in range(THREAD_N)]

sess.run(tf.global_variables_initializer())

executor = concurrent.futures.ThreadPoolExecutor(THREAD_N)
futures = [executor.submit(w.run) for w in workers]
concurrent.futures.wait(futures)
for f in futures:
    f.result()
```

这里用了 python 的 `concurrent.futures` 包，也可以直接使用 `threading.Thread` 并配合 tensorflow 的 `tf.train.Coordinator()` 来同步线程。

# 参考

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., … & Kavukcuoglu, K. (2016, June). Asynchronous methods for deep reinforcement learning. In *International Conference on Machine Learning* (pp. 1928-1937).

<https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/10_A3C/A3C_continuous_action.py>