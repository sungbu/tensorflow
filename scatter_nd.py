import tensorflow as tf

# scatter_nd(
#     indies,
#     updates,
#     shape
# )

#按一定规则填充对应值

#一维案例
indices = tf.constant([[4],[3],[1],[7]])
updates = tf.constant([9,10,11,12])
shape = tf.constant([8])

out = tf.scatter_nd(indices, updates, shape)
print(out) # tf.Tensor([ 0 11  0 10  9  0  0 12], shape=(8,), dtype=int32)
