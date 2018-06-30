---
title: Tensorflow - Optimizer & Gradients
mathjax: true
date: 2018-05-09 20:15:48
updated: 2018-05-09 20:15:48
categories:
- Python
- Tensorflow
tags:
- tensorflow
- python
---

之前遇到的优化问题，在 Tensorflow 中，都是直接用 Optimizer 的 `minimize` 方法将损失函数最小化即可，但最近在写 DDPG Actor 的代码时，无法直接写出一个代价函数进行最小化，而是需要用到 Critic 的梯度利用链式法则合并出代价函数的梯度来进行最小化，所以借此机会简单研究了下 Tensorflow 的梯度计算。

<!--more-->

# Optimizer

先看最简单的用 optimizer 来最小化代价函数：$L=Xw^2$ ，其中 `w` 为待优化参数

```python
import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, shape=(None, 1))
w = tf.get_variable('w', initializer=10.)
L = X * w**2
grads = tf.gradients(L, w)
print(grads)
optimizer = tf.train.GradientDescentOptimizer(1).minimize(L)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run([grads], {
    X: np.array([[1.], [2.]])
}))

sess.run(optimizer, {
    X: np.array([[1.],
                 [2.]])
})

print(sess.run(w))
```

输出：

```
[<tf.Tensor 'gradients/pow_grad/Reshape:0' shape=() dtype=float32>]
[[60.0]]
-50.0
```

可以看出 Tensorflow 对于 `w` 的梯度实际上就是一个 `Tensor` 。在手动计算梯度时，由于这里的 `X` 是个矩阵，所以 $\frac{\partial L}{\partial w} = 2Xw$ ，但 Tensorflow 在这里并没有将梯度的计算变为矩阵 $\begin{bmatrix} 20 \\ 40 \end{bmatrix}$ ，而是直接将梯度矩阵加了起来，因为在实际梯度下降的过程中，执行的步骤就是
$$
w = w - \alpha (2x_1w) \\
w = w - \alpha (2x_2w)
$$
 即为：
$$
w = w-\alpha \sum_{i=1}^2{2x_iw}
$$
由于这里 $\alpha=1$ ，所以最后一行输出优化过后的 `w` 为 $10-60=-50$。

# apply_gradients

上一个例子里，Tensorflow 自动求出损失函数 `L` 每一个可训练 (trianable) 的参数的梯度，然后依次进行梯度下降优化，我们可以手动的应用梯度，并实现梯度的链式法则。先将上面的例子中的 `optimizer = tf.train.GradientDescentOptimizer(1).minimize(L)` 修改为：

```python
optimizer = tf.train.GradientDescentOptimizer(1).apply_gradients([(grads[0], w)])
```

实际效果一样的，`apply_gradients` 函数接收计算完梯度后的 `Tensor` 和 该梯度对应待优化参数的的元组对，若损失函数里包含的待优化参数非常多，也不需要写很多条 `tf.gradients` ，可以修改为：

```python
optimizer = tf.train.GradientDescentOptimizer(1)
grads_and_var = optimizer.compute_gradients(L)
optimizer = optimizer.apply_gradients(grads_and_var)
```

也是相同的效果，只不过自动计算了代价函数的梯度、参数元祖对。

再看一个应用链式法则的梯度计算例子：参数为 `w` ，定义 $g(x)=x-10$ ，$f(x) = x^2$ ，现在的目的很简单就是求 $f(g(w)) = (w-10)^2$ 的最小值，但通过链式求导法则来手动计算梯度，即：
$$
\frac{\partial f}{\partial w} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial w} = 2g(w)\cdot1=2(w-10)
$$

代码如下：

```python
import tensorflow as tf

w = tf.get_variable('w', initializer=5.)
g = w - 10
f = g**2

sess = tf.Session()

grads_f_g = tf.gradients(f, g)
grads_g_w = tf.gradients(g, w, grads_f_g)
opt = tf.train.AdamOptimizer(0.1).apply_gradients([(grads_g_w[0], w)])

sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(opt)

print(sess.run([w, f]))
```

可以看出 `tf.gradients` 函数接受第三个参数 `grad_ys` 来进行梯度的合并

> `grad_ys` is a list of tensors of **the same length as `ys` ** that holds the initial gradients for each y in `ys`. When `grad_ys`is None, we fill in a tensor of '1's of the shape of y for each y in `ys`. A user can provide their own initial `grad_ys` to compute the derivatives using a different initial gradient for each y (e.g., if one wanted to weight the gradient differently for each value in each y). 

输出：

```python
[10.0, 0.0]
```

即 `w` 为10，损失函数最小为0

# 参考

<https://www.tensorflow.org/api_docs/python/tf/gradients>

<https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer>