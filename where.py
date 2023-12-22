import tensorflow as tf

#找到对应张量
a = tf.random.normal([3,3])
print(a)
# tf.Tensor(
# [[-1.2754343  -0.43794826  0.45168066]
#  [ 0.23196214  0.83259153 -1.5422469 ]
#  [ 1.5909908  -0.98468405  0.1435056 ]], shape=(3, 3), dtype=float32)

#方法一
mask = a > 0
print(mask)
# tf.Tensor(
# [[False False  True]
#  [ True  True False]
#  [ True False  True]], shape=(3, 3), dtype=bool)

c = tf.boolean_mask(a,mask)
print(c)
# tf.Tensor([0.45168066 0.23196214 0.83259153 1.5909908  0.1435056 ], shape=(5,), dtype=float32)

#方法二
indics = tf.where(mask)
print(indics)
# tf.Tensor(
# [[0 2]
#  [1 0]
#  [1 1]
#  [2 0]
#  [2 2]], shape=(5, 2), dtype=int64)

d = tf.gather_nd(a,indics)
print(d)
# tf.Tensor([0.45168066 0.23196214 0.83259153 1.5909908  0.1435056 ], shape=(5,), dtype=float32)