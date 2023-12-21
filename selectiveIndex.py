import tensorflow as tf

#自定义随机采样 data[classes,students,subjects]
##tf.gather 一个维度上取数据
a = tf.random.normal([4,35,8])
a1 = tf.cast(a,tf.int16)
#a1: 数据  0:在哪个维度（班级维度上）   indices:索引号
a2 = tf.gather(a1,axis = 0,indices = [2,3])

print(a2.shape) # (2, 35, 8)

a3 = tf.gather(a1,axis = 1,indices = [2,3,7,9,16])
print(a3.shape) # (4, 5, 8)

a4 = tf.gather(a1,axis = 2,indices = [2,3,7])
print(a4.shape) # (4, 35, 3)

##tf.gather_nd 多个维度上取数据
aa1 = tf.gather_nd(a,[0]) # 取0号班级所有的成绩
print(aa1.shape) # (35, 8)

aa2 = tf.gather_nd(a,[0,1]) # 取0号班级1号学生所有的成绩
print(aa2.shape) # (8,)

aa3 = tf.gather_nd(a,[0,1,2]) # 取0号班级1号学生2门课程的成绩 （标量）
print(aa3.shape) # ()
print(aa3.numpy()) # 0.77655345

aa4 = tf.gather_nd(a,[[0,1,2]])# 取0号班级1号学生2门课程的成绩 （张量tensor）
print(aa4.shape) # (1,)

#
aaa1 = tf.gather_nd(a,[[0,0],[1,1]]) # 取0号班级0号学生的所有成绩 和 1号班级1号学生的所有成绩
print(aaa1.shape) # (2, 8)

aaa2 = tf.gather_nd(a,[[0,0],[1,1],[2,2]]) # 取0号班级0号学生的所有成绩 和 1号班级1号学生的所有成绩 和 2号班级2号学生的所有成绩
print(aaa2.shape) # (3, 8)

aaa3 = tf.gather_nd(a,[[0,0,0],[1,1,1],[2,2,2]])# 取0号班级0号学生的第0门成绩 和 1号班级1号学生的第1门成绩 和 2号班级2号学生的第2门成绩
print(aaa3.shape) # (3,)