---
title: tracing in tf.function
mathjax: false
typora-root-url: ..
date: 2020-03-25 13:06:25
categories:
- Python
- Tensorflow
tags: 
- tf
---

```python
import tensorflow as tf
import numpy as np
import time
```

## 比较动态图与静态图的执行速度


```python
def test1(a, b):
    a = tf.pow(a, 2)
    b = tf.pow(b, 2)
    a = a * b
    return a, b


@tf.function
def test2(a, b):
    a = tf.pow(a, 2)
    b = tf.pow(b, 2)
    a = a * b
    return a, b
```


```python
a = np.random.randn(1024, 1024).astype(np.float32)
b = np.random.randn(1024, 1024).astype(np.float32)
```


```python
t = time.time()
for _ in range(100):
    test1(a, b)
print('动态图', time.time() - t)

t = time.time()
test2(a, b)
print('静态图第一次构造', time.time() - t)

t = time.time()
for _ in range(100):
    test2(a, b)
print('静态图', time.time() - t)
```

    动态图 3.0627214908599854
    静态图第一次构造 0.5964057445526123
    静态图 0.6522562503814697


<!--more-->

## tf.function多次trace问题


```python
def test1(a, b):
    print('动态图执行')
    a = tf.pow(a, 2)
    b = tf.pow(b, 2)
    a = a * b
    return a, b

@tf.function
def test2(a, b):
    print('静态图trace')
    a = tf.pow(a, 2)
    b = tf.pow(b, 2)
    a = a * b
    return a, b
```


```python
for _ in range(10):
    test1(a, b)

for _ in range(10):
    test2(a, b)
```

    动态图执行
    动态图执行
    动态图执行
    动态图执行
    动态图执行
    动态图执行
    动态图执行
    动态图执行
    动态图执行
    动态图执行
    静态图trace



```python
@tf.function
def test(a, b):
    print('trace', a, b)
    return a, b
```


```python
for batch in range(1, 10):
    test(np.zeros([batch, 2], dtype=np.float32),
         np.zeros([batch, 3], dtype=np.float32))
    
    None 2  None 3
```

    trace Tensor("a:0", shape=(1, 2), dtype=float32) Tensor("b:0", shape=(1, 3), dtype=float32)
    trace Tensor("a:0", shape=(2, 2), dtype=float32) Tensor("b:0", shape=(2, 3), dtype=float32)
    trace Tensor("a:0", shape=(3, 2), dtype=float32) Tensor("b:0", shape=(3, 3), dtype=float32)
    trace Tensor("a:0", shape=(4, 2), dtype=float32) Tensor("b:0", shape=(4, 3), dtype=float32)
    trace Tensor("a:0", shape=(5, 2), dtype=float32) Tensor("b:0", shape=(5, 3), dtype=float32)
    WARNING:tensorflow:5 out of the last 5 calls to <function test at 0x000002514EB061F8> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.
    trace Tensor("a:0", shape=(6, 2), dtype=float32) Tensor("b:0", shape=(6, 3), dtype=float32)
    WARNING:tensorflow:6 out of the last 6 calls to <function test at 0x000002514EB061F8> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.
    trace Tensor("a:0", shape=(7, 2), dtype=float32) Tensor("b:0", shape=(7, 3), dtype=float32)
    WARNING:tensorflow:7 out of the last 7 calls to <function test at 0x000002514EB061F8> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.
    trace Tensor("a:0", shape=(8, 2), dtype=float32) Tensor("b:0", shape=(8, 3), dtype=float32)
    WARNING:tensorflow:8 out of the last 8 calls to <function test at 0x000002514EB061F8> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.
    trace Tensor("a:0", shape=(9, 2), dtype=float32) Tensor("b:0", shape=(9, 3), dtype=float32)
    WARNING:tensorflow:9 out of the last 9 calls to <function test at 0x000002514EB061F8> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.


## 使用 `input_signature`


```python
@tf.function(input_signature=(tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 3], dtype=tf.float32)))
def test(a, b):
    print('trace', a, b)
    return a, b
```


```python
for batch in range(10):
    test(np.zeros([batch, 2], dtype=np.float32), np.zeros([batch, 3], dtype=np.float32))
```

    trace Tensor("a:0", shape=(None, 2), dtype=float32) Tensor("b:0", shape=(None, 3), dtype=float32)


## 参数为None的问题


```python
@tf.function(input_signature=(tf.TensorSpec(shape=[None, 2], dtype=tf.float32), 
                              tf.TensorSpec(shape=[None, 3], dtype=tf.float32)))
def test(a, b=None):
    print(a, b)
    return a, b
```


```python
for batch in range(10):
    test(np.zeros([batch, 2], dtype=np.float32), np.zeros([batch, 3], dtype=np.float32))
```

    Tensor("a:0", shape=(None, 2), dtype=float32) Tensor("b:0", shape=(None, 3), dtype=float32)



```python
for batch in range(10):
    test(np.zeros([batch, 2], dtype=np.float32))
```


    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in _convert_inputs_to_signature(inputs, input_signature, flat_input_signature)
       2239         flatten_inputs[index] = ops.convert_to_tensor(
    -> 2240             value, dtype_hint=spec.dtype)
       2241         need_packing = True


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\framework\ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
       1313     if ret is None:
    -> 1314       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
       1315 


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\framework\constant_op.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
        316   _ = as_ref
    --> 317   return constant(v, dtype=dtype, name=name)
        318 


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\framework\constant_op.py in constant(value, dtype, shape, name)
        257   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    --> 258                         allow_broadcast=True)
        259 


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\framework\constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
        265   if ctx.executing_eagerly():
    --> 266     t = convert_to_eager_tensor(value, ctx, dtype)
        267     if shape is None:


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\framework\constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
         95   ctx.ensure_initialized()
    ---> 96   return ops.EagerTensor(value, ctx.device_name, dtype)
         97 


    ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.


​    
    During handling of the above exception, another exception occurred:


    ValueError                                Traceback (most recent call last)
    
    <ipython-input-14-ea9826667fb9> in <module>
          1 for batch in range(10):
    ----> 2     test(np.zeros([batch, 2], dtype=np.float32), None)


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\def_function.py in __call__(self, *args, **kwds)
        566         xla_context.Exit()
        567     else:
    --> 568       result = self._call(*args, **kwds)
        569 
        570     if tracing_count == self._get_tracing_count():


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\def_function.py in _call(self, *args, **kwds)
        604       # In this case we have not created variables on the first call. So we can
        605       # run the first trace but we should fail if variables are created.
    --> 606       results = self._stateful_fn(*args, **kwds)
        607       if self._created_variables:
        608         raise ValueError("Creating variables on a non-first call to a function"


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in __call__(self, *args, **kwargs)
       2360     """Calls a graph function specialized to the inputs."""
       2361     with self._lock:
    -> 2362       graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
       2363     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
       2364 


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in _maybe_define_function(self, args, kwargs)
       2659     if self.input_signature is None or args is not None or kwargs is not None:
       2660       args, kwargs = self._function_spec.canonicalize_function_inputs(
    -> 2661           *args, **kwargs)
       2662 
       2663     cache_key = self._cache_key(args, kwargs)


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in canonicalize_function_inputs(self, *args, **kwargs)
       2183           inputs,
       2184           self._input_signature,
    -> 2185           self._flat_input_signature)
       2186       return inputs, {}
       2187 


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in _convert_inputs_to_signature(inputs, input_signature, flat_input_signature)
       2244                          "the Python function must be convertible to "
       2245                          "tensors:\n%s" %
    -> 2246                          format_error_message(inputs, input_signature))
       2247 
       2248   if any(not spec.is_compatible_with(other) for spec, other in zip(


    ValueError: When input_signature is provided, all inputs to the Python function must be convertible to tensors:
      inputs: (
        [],
        None)
      input_signature: (
        TensorSpec(shape=(None, 2), dtype=tf.float32, name=None),
        TensorSpec(shape=(None, 3), dtype=tf.float32, name=None))



```python
@tf.function(input_signature=(tf.TensorSpec(shape=[None, 2], dtype=tf.float32), None))
def test(a, b=None):
    print(a, b)
    return a, b
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-18-c5e14b2bdf68> in <module>
    ----> 1 @tf.function(input_signature=(tf.TensorSpec(shape=[None, 2], dtype=tf.float32), None))
          2 def test(a, b=None):
          3     print(a, b)
          4     return a, b


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\def_function.py in function(func, input_signature, autograph, experimental_implements, experimental_autograph_options, experimental_relax_shapes, experimental_compile)
       1172   """
       1173   if input_signature is not None:
    -> 1174     function_lib.validate_signature(input_signature)
       1175 
       1176   def decorated(inner_function):


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in validate_signature(signature)
       2738     raise TypeError("Invalid input_signature {}; input_signature must be "
       2739                     "a possibly nested sequence of TensorSpec objects."
    -> 2740                     .format(signature))
       2741 
       2742 


    TypeError: Invalid input_signature (TensorSpec(shape=(None, 2), dtype=tf.float32, name=None), None); input_signature must be a possibly nested sequence of TensorSpec objects.



```python
# 可以不传参数为None的input_signature
@tf.function(input_signature=(tf.TensorSpec(shape=[None, 2], dtype=tf.float32),))
def test(a, b=None):
    print(a, b)
    return a, b

# 在调用时也不能传递为None的参数
test(np.zeros([16, 2], dtype=np.float32))
```

    Tensor("a:0", shape=(None, 2), dtype=float32) None





    (<tf.Tensor: shape=(16, 2), dtype=float32, numpy=
     array([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]], dtype=float32)>, None)



且在类中，input_signature中不能使用self关键字


```python
class A:
    def __init__(self):
        self.test_dim = 3
        self.test = self.test.get_concrete_function(a=tf.TensorSpec(shape=[None, self.test_dim], dtype=tf.float32))

    @tf.function
    def test(a):
        print(a)
        return a

A()
```


    ---------------------------------------------------------------------------
    
    NameError                                 Traceback (most recent call last)
    
    <ipython-input-11-db5c0dc01b62> in <module>
    ----> 1 class A:
          2     def __init__(self):
          3         self.test_dim = 3
          4 
          5     @tf.function(input_signature=(tf.TensorSpec(shape=[None, self.test_dim], dtype=tf.float32)))


    <ipython-input-11-db5c0dc01b62> in A()
          3         self.test_dim = 3
          4 
    ----> 5     @tf.function(input_signature=(tf.TensorSpec(shape=[None, self.test_dim], dtype=tf.float32)))
          6     def test(a):
          7         print(a)


    NameError: name 'self' is not defined


## 使用 `concrete function`


```python
@tf.function
def test(a, b=None):
    print(a, b)
    return a, b


test_c = test.get_concrete_function(a=tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                                    b=tf.TensorSpec(shape=[None, 3], dtype=tf.float32))
```

    Tensor("a:0", shape=(None, 2), dtype=float32) Tensor("b:0", shape=(None, 3), dtype=float32)



```python
batch = 16
# 参数不能为numpy类型
test_c(np.zeros([batch, 2], dtype=np.float32), 
       np.zeros([batch, 3], dtype=np.float32))
```


    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-20-7773b92fc8ce> in <module>
          2 # 参数不能为numpy类型
          3 test_c(np.zeros([batch, 2], dtype=np.float32), 
    ----> 4        np.zeros([batch, 3], dtype=np.float32))


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in __call__(self, *args, **kwargs)
       1549       TypeError: For invalid positional/keyword argument combinations.
       1550     """
    -> 1551     return self._call_impl(args, kwargs)
       1552 
       1553   def _call_impl(self, args, kwargs, cancellation_manager=None):


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in _call_impl(self, args, kwargs, cancellation_manager)
       1589       raise TypeError("Keyword arguments {} unknown. Expected {}.".format(
       1590           list(kwargs.keys()), list(self._arg_keywords)))
    -> 1591     return self._call_flat(args, self.captured_inputs, cancellation_manager)
       1592 
       1593   def _filtered_call(self, args, kwargs):


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1682         raise ValueError("All inputs to `ConcreteFunction`s must be Tensors; "
       1683                          "on invocation of %s, the %d-th input (%s) was not a "
    -> 1684                          "Tensor." % (self._func_graph.name, i, str(arg)))
       1685     args = tensor_inputs + captured_inputs
       1686     possible_gradient_type = (


    ValueError: All inputs to `ConcreteFunction`s must be Tensors; on invocation of test, the 0-th input ([[0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]
     [0. 0.]]) was not a Tensor.



```python
# 参数只能为Tensor
test_c(tf.zeros([batch, 2]), 
       tf.zeros([batch, 3]))
```




    (<tf.Tensor: shape=(16, 2), dtype=float32, numpy=
     array([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]], dtype=float32)>,
     <tf.Tensor: shape=(16, 3), dtype=float32, numpy=
     array([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]], dtype=float32)>)




```python
# 若concrete_function参数中没有None，则不能传递None
test_c(tf.zeros([batch, 2]), None)
```


    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-22-46602737c871> in <module>
          1 # 若concrete_function参数中没有None，则不能传递None
    ----> 2 test_c(tf.zeros([batch, 2]), None)


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in __call__(self, *args, **kwargs)
       1549       TypeError: For invalid positional/keyword argument combinations.
       1550     """
    -> 1551     return self._call_impl(args, kwargs)
       1552 
       1553   def _call_impl(self, args, kwargs, cancellation_manager=None):


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in _call_impl(self, args, kwargs, cancellation_manager)
       1589       raise TypeError("Keyword arguments {} unknown. Expected {}.".format(
       1590           list(kwargs.keys()), list(self._arg_keywords)))
    -> 1591     return self._call_flat(args, self.captured_inputs, cancellation_manager)
       1592 
       1593   def _filtered_call(self, args, kwargs):


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1682         raise ValueError("All inputs to `ConcreteFunction`s must be Tensors; "
       1683                          "on invocation of %s, the %d-th input (%s) was not a "
    -> 1684                          "Tensor." % (self._func_graph.name, i, str(arg)))
       1685     args = tensor_inputs + captured_inputs
       1686     possible_gradient_type = (


    ValueError: All inputs to `ConcreteFunction`s must be Tensors; on invocation of test, the 1-th input (None) was not a Tensor.



```python
# 若concrete_function参数中没有None，且指定了Tensor格式，则也不能不传
test_c(tf.zeros([batch, 2]))
```


    ---------------------------------------------------------------------------
    
    KeyError                                  Traceback (most recent call last)
    
    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in _call_impl(self, args, kwargs, cancellation_manager)
       1573       try:
    -> 1574         args.append(kwargs.pop(compat.as_str(keyword)))
       1575       except KeyError:


    KeyError: 'b'


​    
    During handling of the above exception, another exception occurred:


    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-24-c93942248dfa> in <module>
          1 # 若concrete_function参数中没有None，且指定了Tensor格式，则也不能不传
    ----> 2 test_c(tf.zeros([batch, 2]))


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in __call__(self, *args, **kwargs)
       1549       TypeError: For invalid positional/keyword argument combinations.
       1550     """
    -> 1551     return self._call_impl(args, kwargs)
       1552 
       1553   def _call_impl(self, args, kwargs, cancellation_manager=None):


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in _call_impl(self, args, kwargs, cancellation_manager)
       1581                 list(self._arg_keywords),
       1582                 specified_keywords,
    -> 1583                 list(set(self._arg_keywords) - set(specified_keywords))))
       1584     if kwargs:
       1585       positional_arg_keywords = set(self._arg_keywords[:len(args)])


    TypeError: Expected argument names ['a', 'b'] but got values for ['a']. Missing: ['b'].



```python
# 若concrete_function参数中没有指定某参数，则可以不传
test_c = test.get_concrete_function(a=tf.TensorSpec(shape=[None, 2], dtype=tf.float32))
test_c(tf.zeros([batch, 2], dtype=tf.float32))
```

    Tensor("a:0", shape=(None, 2), dtype=float32) None





    (<tf.Tensor: shape=(16, 2), dtype=float32, numpy=
     array([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]], dtype=float32)>, None)




```python
# 但不能传None
test_c(tf.zeros([batch, 2], dtype=tf.float32), None)
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-26-b9e2079756e4> in <module>
          1 # 但不能传None
    ----> 2 test_c(tf.zeros([batch, 2], dtype=tf.float32), None)


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in __call__(self, *args, **kwargs)
       1549       TypeError: For invalid positional/keyword argument combinations.
       1550     """
    -> 1551     return self._call_impl(args, kwargs)
       1552 
       1553   def _call_impl(self, args, kwargs, cancellation_manager=None):


    ~\Miniconda3\envs\tf\lib\site-packages\tensorflow_core\python\eager\function.py in _call_impl(self, args, kwargs, cancellation_manager)
       1568            "of {}), got {}. When calling a concrete function, positional "
       1569            "arguments may not be bound to Tensors within nested structures."
    -> 1570           ).format(self._num_positional_args, self._arg_keywords, args))
       1571     args = list(args)
       1572     for keyword in self._arg_keywords[len(args):]:


    TypeError: Expected at most 1 positional arguments (and the rest keywords, of ['a']), got (<tf.Tensor: shape=(16, 2), dtype=float32, numpy=
    array([[0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.]], dtype=float32)>, None). When calling a concrete function, positional arguments may not be bound to Tensors within nested structures.



```python
# 将numpy类型转为Tensor，同时去除None参数
def np_to_tensor(fn):
    def c(*args, **kwargs):
        return fn(*[tf.constant(k) if not isinstance(k, tf.Tensor) else k for k in args if k is not None],
                  **{k: tf.constant(v) if not isinstance(v, tf.Tensor) else v for k, v in kwargs.items() if v is not None})

    return c


@tf.function
def test(a, b=None):
    print(a, b)
    return a, b


test_c = test.get_concrete_function(a=tf.TensorSpec(shape=[None, 2], dtype=tf.float32))
test = np_to_tensor(test_c)
```

    Tensor("a:0", shape=(None, 2), dtype=float32) None



```python
test(np.zeros([batch, 2], dtype=np.float32))
```




    (<tf.Tensor: shape=(16, 2), dtype=float32, numpy=
     array([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]], dtype=float32)>, None)




```python
test(a=np.zeros([batch, 2], dtype=np.float32), b=None)
```




    (<tf.Tensor: shape=(16, 2), dtype=float32, numpy=
     array([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]], dtype=float32)>, None)



## 参数为`list`
参考 [https://github.com/tensorflow/tensorflow/issues/37778]()


```python
@tf.function
def test(a, b, c):
    pass

def _list_arg_to_concrete_arg(name, arg_list):
    concrete_arg = {
        name: arg_list[0]
    }
    for i, arg in enumerate(arg_list[1:]):
        concrete_arg[f'{name}_{i+1}'] = arg

    return concrete_arg

c_test = test.get_concrete_function(a=[tf.TensorSpec((None, 1)),
                                       tf.TensorSpec((None, 1))],
                                    b=tf.TensorSpec((None, 2)),
                                    c=[tf.TensorSpec((None, 3)),
                                       tf.TensorSpec((None, 3))])

c_test(**_list_arg_to_concrete_arg('a', [tf.random.normal([2, 1]), tf.random.normal([2, 1])]),
       b=tf.random.normal([2, 2]),
       **_list_arg_to_concrete_arg('c', [tf.random.normal([3, 3]), tf.random.normal([3, 3])]))
```
