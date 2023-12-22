import tensorflow as tf

## 合并 concat 在现有维度上进行合并 (合并的维度可以不相等  其他维度必须相等)
a = tf.ones([4,35,8])
b = tf.ones([2,35,8])

c = tf.concat([a,b],axis=0)
print(c.shape) # (6, 35, 8)

d = tf.ones([4,3,8])
e = tf.concat([a,d],axis=1)
print(e.shape) # (4, 38, 8)

a = tf.ones([4,35,8])
b = tf.ones([4,35,8])

f = tf.concat([a,b],axis=-1)
print(f.shape) # (4, 35, 16)



## stack 合并 在新的维度上进行合并  (所有维度都必须相等)
g = tf.stack([a,b],axis=0)
print(g.shape) # (2, 4, 35, 8)

h = tf.stack([a,b],axis=3)
print(h.shape) # (4, 35, 8, 2)

## unstack 拆分 (拆分出axis维度个数个tensor)
aa,bb = tf.unstack(g,axis=0)
print(aa.shape) #(4, 35, 8)
print(bb.shape) #(4, 35, 8)

## split 和unstack相比 可以指定拆分的组合个数
res = tf.split(g,axis=3,num_or_size_splits=2)
print(len(res)) # 2
print(res[0].shape) # (2, 4, 35, 4)
print(res[1].shape) #(2, 4, 35, 4)