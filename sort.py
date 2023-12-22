import tensorflow as tf

#排序
a = tf.random.shuffle(tf.range(5))
print(a) # tf.Tensor([1 4 0 2 3], shape=(5,), dtype=int32)

a1 = tf.sort(a,direction='DESCENDING')
print(a1) # tf.Tensor([4 3 2 1 0], shape=(5,), dtype=int32)

idx = tf.argsort(a,direction='DESCENDING')
print(idx) # tf.Tensor([1 4 3 0 2], shape=(5,), dtype=int32)

a2 = tf.gather(a,idx)
print(a2) # tf.Tensor([4 3 2 1 0], shape=(5,), dtype=int32)


##tf.math.top_k 限制个数
b1 = tf.random.normal([3,3],mean=5,stddev=1)
b = tf.cast(b1,dtype=tf.int32)
print(b)
# [[3 5 5]
#  [4 6 6]
#  [3 4 5]], shape=(3, 3), dtype=int32)

res = tf.math.top_k(b,2)
print(res.indices)
# tf.Tensor(
# [[1 2]
#  [1 2]
#  [2 1]], shape=(3, 2), dtype=int32)
print(res.values)
# [[5 5]
#  [6 6]
#  [5 4]], shape=(3, 2), dtype=int32)