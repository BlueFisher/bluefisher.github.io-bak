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
---

循环神经网络 (Recurrent Neural Networks RNN) 作为一种序列模型非常适合用在处理数据之间有着时间顺序的问题，比如自然语言处理 (NLP) ，Tensorflow 将 RNN 中循环的网络封装为一个细胞 (cell) ，本文简单介绍一下如何构建一个长短期记忆 (Long Short-Term Memory LSTM) 网络。

以自然语言处理为例，一句话可以看成一个序列，如果我们使用批处理，那输入的数据应该为多个句子。假设有 `batch_size` 个句子，每个句子有 `num_steps` 个单词，相当于 RNN 的时间序列长度为 `num_steps` ，每个单词以 one-hot 形式储存，总共有 `lstm_size` 个单词，那么每个单词就代表长度为 `lstm_size` 的向量中的一个元素。可以看出，Tensorflow 中，RNN 输入的张量应该是一个三维张量 `(batch_size, num_steps, lstm_size)`

```python
inputs = tf.placeholder(tf.float32, shape=(batch_size, num_steps, lstm_size))
```

首先，先定义一个 LSTM 细胞，它的大小为 `lstm_size` ：

```python
cell = tf.nn.rnn_cell.BasicRNNCell(lstm_size)
```

定义 RNN ：

```python
outputs, states = tf.nn.dynamic_rnn(cell, inputs=inputs, dtype=tf.float32)
```

其中 `outputs` 为 RNN 的输出，维度为三维，前两维是输入的 `batch_size` 与 `num_steps` ，第三维是传入 `cell` 的大小，这里的大小为 `lstm_size` 所以此时的输出维度与输入一样。 `states` 为 LSTM 循环中间产生的记忆状态，维度为 `(batch_size, num_steps)` ，可以传递到下一次的循环中。
