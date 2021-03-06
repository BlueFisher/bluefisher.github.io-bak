---
title: Tensorflow RNN
mathjax: false
date: 2018-07-12 09:41:29
categories:
- Python
- Tensorflow
tags:
- python
- tensorflow
typora-root-url: ..
---

![RNN](/images/2018-07-12-Tensorflow-RNN/Pw5xCn.md.png) 

![LSTM](/images/2018-07-12-Tensorflow-RNN/Pw5z3q.png) 

循环神经网络 (Recurrent Neural Networks RNN) 作为一种序列模型非常适合用在处理数据之间有着时间顺序的问题，比如自然语言处理 (NLP) ，Tensorflow 将 RNN 中循环的网络封装为一个细胞 (cell) ，本文简单介绍一下如何构建一个长短期记忆 (Long Short-Term Memory LSTM) 网络。

<!--more-->

# 单层 LSTM

以自然语言处理为例，一句话可以看成一个序列，如果我们使用批处理，那输入的数据应该为多个句子。假设有 `batch_size` 个句子，每个句子有 `num_steps` 个单词，相当于 RNN 的时间序列长度为 `num_steps` ，每个单词以 one-hot 形式储存，总共有 `input_size` 个单词，那么每个单词就代表长度为 `input_size` 的向量中的一个元素，也就是图中 `x` 向量的长度。可以看出，Tensorflow 中，RNN 输入的张量应该是一个三维张量 `(batch_size, num_steps, input_size)`

```python
inputs = tf.placeholder(tf.float32, shape=(batch_size, num_steps, input_size))
```

首先，先定义一个 LSTM 细胞，它的大小为输出的向量长度 `lstm_size` ，也就是图中 `h` 向量的长度：

```python
rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
```

设置初始化的状态，该状态是一个长度为 N 的元组，具体参考再下一行代码的 `state`：

```python
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
```

定义 RNN ：

```python
outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                   inputs=inputs,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
```

其中 `outputs` 为 RNN 的输出，维度为三维，前两维是输入的 `batch_size` 与 `num_steps` ，第三维是传入 `cell` 的大小，这里的大小为 `lstm_size` 所以此时的输出维度与输入一样。 `state` 为 LSTM 循环中间产生的记忆状态，是一个长度为 N 的元组，要看整个 RNN 网络有几层组成，在上述代码中，只有一层 LSTM ，而一层 LSTM 会产生两个记忆状态，所以此时的 N 为 2，维度都为为 `(batch_size, lstm_size)` ，可以传递到下一次的循环中，成为下一次的初始状态。

# 多层 LSTM

```python
rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [lstm_size1, lstm_size2]]
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
initial_state = multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=inputs,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
```

这里我们定义了双层 LSTM 网络，第一层的输出长度为 `lstm_size1` ，接着传给第二层输出 `lstm_size2` ，使用 `MultiRNNCell` 函数拼接  。此时 `outputs` 的维度为 `(batch_size, num_steps, lstm_size2)` ，由于是两层 LSTM ，每层 LSTM 会产生两个状态，所以 `state` 为一个四元组，前两个维度为 `(batch_size, lstm_size1)` ，后两个为 `(batch_size, lstm_size2)`