import tensorflow as tf

a = tf.random.normal([4,28,28,3])
print(a.shape) # (4, 28, 28, 3)
print(a.ndim) # 4

a1 = tf.reshape(a,[4,784,3]) 
print(a1.shape)# (4, 784, 3)
print(a1.ndim) # 3

a2 = tf.reshape(a,[4,-1,3])
print(a2.shape)# (4, 784, 3)
print(a2.ndim)# 3

a3 = tf.reshape(a,[4,784*3])
print(a3.shape)# (4, 2352)
print(a3.ndim)# 2

a4 = tf.reshape(a,[4,-1])
print(a4.shape)# (4, 2352)
print(a4.ndim)# 2


#
aa1 = tf.reshape(tf.reshape(a,[4,-1]),[4,28,28,3])
print(aa1.shape)# (4, 28, 28, 3)

aa2 = tf.reshape(tf.reshape(a,[4,-1]),[4,14,56,3])
print(aa2.shape)# (4, 14, 56, 3)