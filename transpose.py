import tensorflow as tf

#维度交换（转置） [h,w] -> [w,h]
a = tf.random.normal((4,3,2,1))
print(a.shape) # (4, 3, 2, 1)

a1 = tf.transpose(a) #默认全部转置
print(a1.shape) # (1, 2, 3, 4)

a2 = tf.transpose(a,perm=[0,1,3,2]) #[放以前的第0维,放以前的第1维,放以前的第3维,放以前的第2维]
print(a2.shape) # (4, 3, 2, 1) 
