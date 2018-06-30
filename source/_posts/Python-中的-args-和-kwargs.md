---
title: Python 中的 *args 和 **kwargs
mathjax: false
date: 2018-05-06 20:59:22
updated: 2018-05-06 20:59:22
categories:
- Python
tags:
- python
- tricks
---

在编写函数时，  `*args` 和 `**kwargs` 可以使我们向函数传递任意数量的参数(Arbitrary Argument Lists)

`*args` 将函数的普通参数打包成元组的形式：

```python
>>> def foo(*args):
...     print(type(args))
...     for a in args:
...         print(a)
...
>>>
>>> foo(1, 'a', [1, 2, 3])
<class 'tuple'>
1
a
[1, 2, 3]
```

`**kwargs` 将带有关键字的参数打包成字典的形式：

```python
>>> def bar(**kwargs):
...     print(type(kwargs))
...     for a in kwargs:
...         print(a, kwargs[a])
...
>>>
>>> bar(a=1, b='a', c=[1, 2, 3])
<class 'dict'>
a 1
b a
c [1, 2, 3]
```

普通参数、`*args` 、 `**kwargs` 都可以混合使用

```python
def foo(kind, *args, **kwargs):
    pass
```

<!--more-->

另外， `*` 与 `**` 还可以起到解包的作用，将元组、列表、字典等解包，作为参数传递给函数

`*` 可以将列表解包成普通函数参数的形式：

```python
def foo(bar, lee):
    pass


l = [1, 2]
foo(*l)
```

例如在 `zip` 函数中：

```python
>>> a = [1, 2, 3]
>>> b = [4, 5, 6]
>>> zipped = list(zip(a, b))
>>> print(zipped)
[(1, 4), (2, 5), (3, 6)]
>>> unzipped = list(zip(*zipped))
>>> print(unzipped)
[(1, 2, 3), (4, 5, 6)]
```

`**` 可以将字典解包成带关键字的函数参数形式，比如在tensorflow中：

```python
initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.1),
    'bias_initializer': tf.constant_initializer(.1)
}

tf.layers.dense(foo, bar, **initializer_helper)
```

相当于是：

```python
tf.layers.dense(
    foo, bar,
    kernel_initializer=tf.random_normal_initializer(0., 0.1),
    bias_initializer=tf.constant_initializer(.1)
)
```

可以省去每次写一堆重复关键字参数的代码

---

> 如果你爱Python这门语言，就少用一些你觉得爽但是别人会mmp的语法了。告诉我，当你看到别人写的函数有**kwargs这样的参数，你不想问候他祖宗十八代吗？ 

😂 [知乎：如何看待知乎、饿了么后端的招聘纷纷由 Python 渐渐转向 Java？](https://www.zhihu.com/question/56468869/answer/293589878)

# 参考

<https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters>

<https://docs.python.org/dev/tutorial/controlflow.html#more-on-defining-functions>
