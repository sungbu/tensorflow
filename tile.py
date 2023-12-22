import tensorflow as tf

#数据复制
a = tf.reshape(tf.range(9),[3,3]);
print(a)
# tf.Tensor(
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]], shape=(3, 3), dtype=int32)

#复制 [1,2][第0个维度复制的次数,第1个维度复制的的次数]
b = tf.tile(a,[1,2])
print(b)
# tf.Tensor(
# [[0 1 2 0 1 2]
#  [3 4 5 3 4 5]
#  [6 7 8 6 7 8]], shape=(3, 6), dtype=int32)
c = tf.tile(a,[2,1])
print(c)
# tf.Tensor(
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]
#  [0 1 2]
#  [3 4 5]
#  [6 7 8]], shape=(6, 3), dtype=int32)
d = tf.tile(a,[2,2])
print(d)
# tf.Tensor(
# [[0 1 2 0 1 2]
#  [3 4 5 3 4 5]
#  [6 7 8 6 7 8]
#  [0 1 2 0 1 2]
#  [3 4 5 3 4 5]
#  [6 7 8 6 7 8]], shape=(6, 6), dtype=int32)