import tensorflow as tf

#根据值范围替换
a = tf.range(10)
print(a)

b = tf.maximum(a,2) # if(x < 2) return 2 else return 原来的值
print(b) # tf.Tensor([2 2 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32)

c = tf.minimum(a,8)# if(x > 8) return 8 else return 原来的值
print(c) # tf.Tensor([0 1 2 3 4 5 6 7 8 8], shape=(10,), dtype=int32)

d = tf.clip_by_value(a,2,8) # if(x < 2) return 2 else if(x > 8) return 8 else return 原来的值
print(d) # tf.Tensor([2 2 2 3 4 5 6 7 8 8], shape=(10,), dtype=int32)


#clip_by_norm 保证向量的方向不变  值改变向量的模
a = tf.random.normal([2,2],mean=10)

a1 = tf.norm(a)
print(a1) # tf.Tensor(20.856255, shape=(), dtype=float32)

a2 = tf.clip_by_norm(a,15)
print(a2)
# tf.Tensor(
# [[7.4172664 7.3357487]
#  [7.687729  7.5544524]], shape=(2, 2), dtype=float32)

a3 = tf.norm(a2)
print(a3)
# tf.Tensor(15.0, shape=(), dtype=float32)