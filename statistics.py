import tensorflow as tf

# 张量的范数
# 二范数(平方求和开根号)
a = tf.ones([2,2])
a1 = tf.norm(a)
print(a1) # tf.Tensor(2.0, shape=(), dtype=float32)

a2 = tf.sqrt(tf.reduce_sum(tf.square(a)))
print(a2) # tf.Tensor(2.0, shape=(), dtype=float32)


b = tf.ones([4,28,28,3])
b1 = tf.norm(b)
print(b1) # tf.Tensor(96.99485, shape=(), dtype=float32)

b2 = tf.sqrt(tf.reduce_sum(tf.square(b)))
print(b2) # tf.Tensor(96.99485, shape=(), dtype=float32)


#一范数 ord=2(二范数)1(一范数)  axis将axis维度看成一个整体求值
c = tf.ones([2,2])

c2 = tf.norm(c,ord=2,axis=1)
print(c2) # tf.Tensor([1.4142135 1.4142135], shape=(2,), dtype=float32)

c3 = tf.norm(c,ord=1)
print(c3) # tf.Tensor(4.0, shape=(), dtype=float32)

c4 = tf.norm(c,ord=1,axis=0)
print(c4) # tf.Tensor([2. 2.], shape=(2,), dtype=float32)

c5 = tf.norm(c,ord=1,axis=1)
print(c5) # tf.Tensor([2. 2.], shape=(2,), dtype=float32)


#求最大最小值均值 axis表示对某个维度单独求最大最小和均值
d = tf.random.normal([4,10])

d1 = tf.reduce_min(d)
print(d1) # tf.Tensor(-1.7455113, shape=(), dtype=float32)

d2 = tf.reduce_max(d)
print(d2) # tf.Tensor(1.2073698, shape=(), dtype=float32)

d3 = tf.reduce_mean(d)
print(d3) # tf.Tensor(-0.047655616, shape=(), dtype=float32)

d4 = tf.reduce_min(d,axis=1)
print(d4) # tf.Tensor([-1.4557011  -0.74233264 -1.7455113  -1.2171638 ], shape=(4,), dtype=float32)

d5 = tf.reduce_max(d,axis=1)
print(d5) # tf.Tensor([1.1012672  1.2073698  0.79855865 0.69225967], shape=(4,), dtype=float32)

d6 = tf.reduce_mean(d,axis=1)
print(d6) # tf.Tensor([ 0.22874533  0.14407286 -0.26812136 -0.2953193 ], shape=(4,), dtype=float32)

##求最大最小值的位置 默认axis=0
e = tf.random.normal([4,10])

e1 = tf.argmax(e)
print(e1) # tf.Tensor([1 1 3 2 2 3 1 0 1 3], shape=(10,), dtype=int64)

e2 = tf.argmin(e)
print(e2) # tf.Tensor([0 3 1 1 1 1 2 2 2 1], shape=(10,), dtype=int64)

##比较大小
a = tf.constant([1,2,3,2,5])

b = tf.range(5)
print(b.numpy()) # [0 1 2 3 4]

c1 = tf.equal(a,b)
print(c1) # tf.Tensor([False False False False False], shape=(5,), dtype=bool)

d1 = tf.reduce_sum(tf.cast(c1,dtype=tf.int32))
print(d1) #tf.Tensor(0, shape=(), dtype=int32)



##去除重复元素
a = tf.constant([4,2,2,4,3])
a1 = tf.unique(a)
print(a1)
tf.Tensor(0, shape=(), dtype=int32)
# Unique(y=<tf.Tensor: shape=(3,), dtype=int32, numpy=array([4, 2, 3], dtype=int32)>, idx=<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 1, 0, 2], dtype=int32)>)




###实战 计算accurcy
a = tf.constant([[0.1,0.2,0.7],[0.9,0.05,0.05]])
y = tf.constant([2,1])
print(a)
# tf.Tensor(
# [[0.1  0.2  0.7 ]
#  [0.9  0.05 0.05]], shape=(2, 3), dtype=float32)
    
pred = tf.cast(tf.argmax(a,axis=1),dtype=tf.int32)
print(pred) # tf.Tensor([2 0], shape=(2,), dtype=int32)
print(y) # tf.Tensor([2 1], shape=(2,), dtype=int32)

equal = tf.equal(y,pred)
print(equal) # tf.Tensor([ True False], shape=(2,), dtype=bool)

correct = tf.reduce_sum(tf.cast(equal,dtype=tf.int32))
print(correct) # tf.Tensor(1, shape=(), dtype=int32)

acc = correct / 2

print(acc) # tf.Tensor(0.5, shape=(), dtype=float64)
