import tensorflow as tf
import numpy as np

#将numpy数据转换成tensor
#np.ones([2,3]) ones填充为1 [2,3]两行三列的数据
a = tf.convert_to_tensor(np.ones([2,3]))
print(a)
# tf.Tensor(
# [[1. 1. 1.]
#  [1. 1. 1.]], shape=(2, 3), dtype=float64)

b = tf.convert_to_tensor(np.zeros([2,3]))
print(b)
# tf.Tensor(
# [[0. 0. 0.]
#  [0. 0. 0.]], shape=(2, 3), dtype=float64)

c = tf.convert_to_tensor([1,2])
print(c)
# tf.Tensor([1 2], shape=(2,), dtype=int32)


d = tf.convert_to_tensor([1,2.])
print(d)
#tf.Tensor([1. 2.], shape=(2,), dtype=float32)


e = tf.convert_to_tensor([[1],[2.]])
print(e)
# tf.Tensor(
# [[1.]
#  [2.]], shape=(2, 1), dtype=float32)


#转换数据类型
bb = tf.cast(b,tf.float32)
print(bb)
# tf.Tensor(
# [[0. 0. 0.]
#  [0. 0. 0.]], shape=(2, 3), dtype=float32)