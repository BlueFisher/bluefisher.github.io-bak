---
title: Python Tricks
mathjax: true
date: 2018-05-12 10:08:29
updated: 2018-06-19 15:00:30
categories:
- Python
tags:
- python
- tricks
---

- numpy 一维数组与二维数组相加
- class 中的 self
- global & nonlocal
- 默认参数陷阱
- 交换变量
- 行内 if 语句
- 带索引的列表迭代
- *… 持续更新中*

<!--more-->

# numpy 一维数组与二维数组相加

在写 tensorflow 时，`dense` 层的输出是一个二维数组，假设为 $n \times 1$ 的 `Tensor` ，而此时与另一个一维 `Tensor` 相加，则会出现意想不到的问题。用 numpy 的矩阵举个例子：

```python
>>> import numpy as np
>>>
>>> a = np.array([1, 2, 3])
>>> b = np.array([[1], [2], [3]])
>>> print(a + b)
[[2 3 4]
 [3 4 5]
 [4 5 6]]
```

结果是个 $3 \times 3$ 的矩阵，难怪程序不收敛。所以在实际写程序的过程中，尽量设置相同 `Tensor` 的维数。

# class 中的 self

类中所有函数无论有没有参数，在定义成员函数时必须接受一个 `self` 参数：

```python
class foo(object):
    def bar(self):
        pass

    def lee(self, *args, **kwargs):
        pass
```

# global & nonlocal

## global

当在局部作用域中要使用全局变量时，要使用 `global` 来修饰全局变量，如果不需要修改全局变量，也可以不使用 `global` 关键字。

```python
g = 0


def foo():
    g = 10
    print('local', g)


print('global', g)
foo()
print('global', g)
```

输出：

```
global 0
local 10
global 0
```

这时第5行的 `g` 为局部变量，尽管与痊愈变量 `g` 的名字相同，但是两个不同的变量。如果要修改全局变量，则要加上 `global` ：

```python
def foo():
    global g
    g = 10
    print('local', g)
```

输出：

```
global 0
local 10
global 10
```

## nonlocal

`nonlocal` 与 `global` 类似，`nonlocal` 用来修饰外层（非全局）变量。

```python
def foo():
    g = 0

    def bar():
        g = 10
        print('bar', g)

    bar()
    print('foo', g)


foo()
```

输出：

```
bar 10
foo 0
```

如果要修改外部变量 `g` 则要加上 `nonlocal` 关键字：

```python
def bar():
    nonlocal g
    g = 10
    print('bar', g)
```

输出：

```
bar 10
foo 10
```

# 默认参数陷阱

```python
def foo(arr=[], el=None):
    arr.append(el)
    print(arr)


foo(el=1)
foo(el=2)
```

对于这样一个函数，其中有默认参数 `arr=[]` ，如果是 javascript 或其他编程语言的话，每次调用函数都是默认把一个空 `list` 赋值给 `arr` ，但 python 不是这么做的，它的输出为：

```
[1]
[1, 2]
```

因为 python 函数的参数默认值，是在编译阶段就确定的，之后所有的函数调用，如果参数不显示的赋值，默认参数都是指向在编译时就确定的对象指针。

对于普通的不可变变量 `int` , `string` , `float` , `tuple`，在函数体内如果修改了该参数，那么参数就会重新指向另一个新的不可变值。

但对于可变对象 `list` , `dict` ，所有对默认参数的修改实际上都是对编译时已经确定的那个对象的修改。

# 交换变量

```python
>>> x = 1
>>> y = 2
>>>
>>> x, y = y, x
>>> print(x, y)
2 1
```

# 行内 if 语句

```python
>>> print("Hello" if True else "World")
Hello
>>> print("Hello") if False else print("World")
World
```

# 带索引的列表迭代

```python
>>> arr = ['foo', 'bar', 'lee']
>>> for index, el in enumerate(arr):
...     print(index, el)
...
0 foo
1 bar
2 lee
```