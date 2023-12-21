import tensorflow as tf

# + - * / % //
b = tf.fill([2,2],2.)
a = tf.ones([2,2])

print(a+b)
# [[3. 3.]
#  [3. 3.]], shape=(2, 2), dtype=float32)
print(a-b)
# tf.Tensor(
# [[-1. -1.]
#  [-1. -1.]], shape=(2, 2), dtype=float32)
print(a*b)
# tf.Tensor(
# [[2. 2.]
#  [2. 2.]], shape=(2, 2), dtype=float32)
print(a/b)
# tf.Tensor(
# [[0.5 0.5]
#  [0.5 0.5]], shape=(2, 2), dtype=float32)
print(tf.math.log(a))
# tf.Tensor(
# [[0. 0.]
#  [0. 0.]], shape=(2, 2), dtype=float32)
print(tf.exp(a))
# tf.Tensor(
# [[2.7182817 2.7182817]
#  [2.7182817 2.7182817]], shape=(2, 2), dtype=float32)

print(tf.math.log(8.) / tf.math.log(2.))
# tf.Tensor(3.0, shape=(), dtype=float32)
print(tf.math.log(100.) / tf.math.log(10.))
# tf.Tensor(2.0, shape=(), dtype=float32)

#pow次方 sqrt开方
print(tf.pow(b,3))
# tf.Tensor(
# [[8. 8.]
#  [8. 8.]], shape=(2, 2), dtype=float32)
#
print(b ** 3)
# tf.Tensor(
# [[8. 8.]
#  [8. 8.]], shape=(2, 2), dtype=float32)

print(tf.sqrt(b))
# tf.Tensor(
# [[1.4142135 1.4142135]
#  [1.4142135 1.4142135]], shape=(2, 2), dtype=float32)

# @ 矩阵运算
print(a @ b)
# tf.Tensor(
# [[4. 4.]
#  [4. 4.]], shape=(2, 2), dtype=float32)
print(tf.matmul(a,b))
# tf.Tensor(
# [[4. 4.]
#  [4. 4.]], shape=(2, 2), dtype=float32)

# [4][2,3] @ [4][3,5] => [4][2,5]
a = tf.ones([4,2,3])
b = tf.fill([4,3,5],2.)
print(a @ b)
# tf.Tensor(
# [[[6. 6. 6. 6. 6.]
#   [6. 6. 6. 6. 6.]]

#  [[6. 6. 6. 6. 6.]
#   [6. 6. 6. 6. 6.]]

#  [[6. 6. 6. 6. 6.]
#   [6. 6. 6. 6. 6.]]

#  [[6. 6. 6. 6. 6.]
#   [6. 6. 6. 6. 6.]]], shape=(4, 2, 5), dtype=float32)