import tensorflow as tf
import numpy as np

## tf.convert_to_tensor(data) tf.constant(data)  两个方法等同
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



## tf.zeros(shape)  tf.ones(shape) tf.fill(shape,num) tf.random.normal(shape,mean,stddev)
aaa = tf.zeros([])
print(aaa)
#tf.Tensor(0.0, shape=(), dtype=float32)

bbb = tf.zeros([1])
print(bbb)
#tf.Tensor([0.], shape=(1,), dtype=float32)

ccc = tf.zeros([3,2])
print(ccc)
# tf.Tensor(
# [[0. 0.]
#  [0. 0.]
#  [0. 0.]], shape=(3, 2), dtype=float32)

ddd = tf.zeros([2,3,3])
print(ccc)
# tf.Tensor(
# [[0. 0.]
#  [0. 0.]
#  [0. 0.]], shape=(3, 2), dtype=float32)

eee = tf.fill([2,3],3)
print(eee)
# tf.Tensor(
# [[3 3 3]
#  [3 3 3]], shape=(2, 3), dtype=int32)


##正态分布
#normal 正态分布  mean均值  stddev方差
fff = tf.random.normal([2,3],mean = 1,stddev = 1)
print(fff)
# tf.Tensor(
# [[0.88460416 1.341759   1.3003764 ]
#  [2.865214   2.0417223  1.7424684 ]], shape=(2, 3), dtype=float32)

#截断的正态分布
ggg = tf.random.truncated_normal([2,3],mean = 0,stddev = 1)
print(ggg)
# tf.Tensor(
# [[ 1.4676268  -0.2924926   0.79790026]
#  [ 0.29715794  0.34682703 -0.220721  ]], shape=(2, 3), dtype=float32)


##均匀分布
hhh = tf.random.uniform([2,3],minval = 0,maxval = 1)
print(hhh)
# tf.Tensor(
# [[0.44952643 0.227602   0.3751372 ]
#  [0.8394946  0.65379524 0.36792338]], shape=(2, 3), dtype=float32)


### 在某一个维度打散 [23,128,128,3]  在23（图片张数的维度）打散  数据和标签的打散也需要同步

idx = tf.range(10) #生成[0-(num-1)]的数据
idx = tf.random.shuffle(idx)
print(idx)
#tf.Tensor([ 2  1  6  5  8  9  3  0 10  7  4], shape=(11,), dtype=int32)

a1 = tf.random.normal([10,784])
b1 = tf.random.uniform([10],maxval = 10, dtype=tf.int32)
print(a1)
# tf.Tensor(
# [[-1.6404355  -1.2437458  -0.5601016  ...  1.1270251   1.505386
#   -1.7553782 ]
#  [-0.5719522   1.1967027  -0.6265038  ... -2.0629847  -0.42430753
#    0.7923556 ]
#  [-1.2378675   1.0013101  -0.68806636 ...  0.17599575 -0.25089052
#   -1.3089185 ]
#  ...
#  [ 0.6588213   0.1704336  -0.6893655  ... -1.3371702   0.8469205
#    0.4589822 ]
#  [-2.1922586  -1.3143234   0.0766689  ...  0.18827249  0.3793327
#   -0.02671175]
#  [-0.03306737 -0.17829102  0.33649486 ... -0.2923438  -1.6222512
#    0.02916134]], shape=(10, 784), dtype=float32)
print(b1)
# tf.Tensor([6 5 3 2 4 4 3 0 3 8], shape=(10,), dtype=int32)

aa1 = tf.gather(a1,idx)
bb1 = tf.gather(b1,idx)
print(aa1)
# tf.Tensor(
# [[ 3.5350916e-01  1.1860502e+00 -1.2749006e+00 ... -1.4599077e+00
#    7.1220255e-01  5.4348505e-01]
#  [ 6.5952212e-01  8.9758961e-04 -1.1430790e+00 ...  2.8275824e-01
#    1.8281003e+00  8.9035499e-01]
#  [ 2.2291248e+00 -8.3986139e-01  4.9100929e-01 ...  2.4522670e-01
#    1.4572504e+00  1.6487986e+00]
#  ...
#  [-2.8849177e+00 -5.5736381e-01  8.9024037e-01 ... -8.5158527e-01
#    7.2306848e-01  1.0584487e+00]
#  [ 5.5584067e-01  8.3075726e-01  1.0838825e+00 ... -5.8627181e-02
#    1.3584747e+00 -9.9738002e-01]
#  [ 5.4926640e-01 -1.8552508e-02 -1.2863708e+00 ...  1.5413624e+00
#    6.8309128e-02  1.5815517e-02]], shape=(10, 784), dtype=float32)
print(bb1)
#tf.Tensor([8 1 2 5 4 7 3 8 6 9], shape=(10,), dtype=int32)