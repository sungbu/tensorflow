import tensorflow as tf

a = tf.random.normal([4,28,28,3])
print(a.shape) # (4, 28, 28, 3)

a1 = tf.boolean_mask(a,mask = [True,True,False,False]) # 在班级维度只取0，1号班级
print(a1.shape) #(2, 28, 28, 3)


a2 = tf.boolean_mask(a,mask = [True,True,False],axis = 3) # 在科目维度只取0，1门成绩
print(a2.shape) #(4, 28, 28, 2)

a = tf.ones([2,3,4])

a3 = tf.boolean_mask(a,mask = [[True,False,False],[False,True,False]])
print(a3.shape) # (2, 4)