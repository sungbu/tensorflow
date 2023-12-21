import tensorflow as tf
#      [4,28,28,3] +  [3] 维度不一致
# 第一步[4,28,28,3] +  [1,1,1,3]
# 第二步[4,28,28,3] +  [4,28,28,3]

#自动调整张量的形状
x = tf.random.normal([4,32,32,3])
print(x.shape) # (4, 32, 32, 3)

x1 = x + tf.random.normal([3])
print(x1.shape) # (4, 32, 32, 3)

x2 = x + tf.random.normal([32,32,1])
print(x2.shape) # (4, 32, 32, 3)


x3 = x + tf.random.normal([4,1,1,1])
print(x3.shape) # (4, 32, 32, 3)

x4 = x + tf.random.normal([1,4,1,1])
print(x4.shape) # error

b = tf.broadcast_to(tf.random.normal([4,1,1,1]),[4,32,32,3])
print(b.shape) # (4, 32, 32, 3)