import tensorflow as tf

#增加一个维度
a = tf.random.normal((94,35,8))
print(a.shape) # (94, 35, 8)

a1 = tf.expand_dims(a,axis=0)
print(a1.shape) # (1, 94, 35, 8)

a2 = tf.expand_dims(a,axis=3)
print(a2.shape) # (94, 35, 8, 1)

#减少维度 专门减少shape为1的维度
b1 = tf.zeros([1,2,1,3])
print(b1.shape) # (1, 2, 1, 3)

bb1 = tf.squeeze(b1,axis=0)
print(bb1.shape) # (2, 1, 3)

bb2 = tf.squeeze(b1,axis=-2)
print(bb2.shape) # (1, 2, 3)
