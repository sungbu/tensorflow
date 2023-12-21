import tensorflow as tf

# 创建一个整型的数据常量
num_int = tf.constant(1)
print(num_int)  # tf.Tensor(1, shape=(), dtype=int32)


# 创建一个浮点数的数据常量
num_float = tf.constant(1.0)
print(num_float)  # tf.Tensor(1.0, shape=(), dtype=float32)

# 创建一个指定类型的浮点数常量
float32_const = tf.constant(2.2, dtype=tf.float32)  # 创建一个 float32 类型的常量张量
print(float32_const) #tf.Tensor(2.2, shape=(), dtype=float32)

double_const = tf.constant(2., dtype=tf.double)  # 创建一个 double 类型的常量张量
print(double_const) #tf.Tensor(2.0, shape=(), dtype=float64)

# 创建一个布尔类型的常量张量
bool_const = tf.constant([True, False])  # 创建一个包含布尔值的常量张量
print(bool_const) # tf.Tensor([ True False], shape=(2,), dtype=bool)

# 创建一个字符串类型的常量张量
string_const = tf.constant('hello world.')  # 创建一个包含字符串的常量张量
print(string_const) # tf.Tensor(b'hello world.', shape=(), dtype=string)

# 在cpu中创建
with tf.device('cpu'):
    a = tf.constant([1])
# 在gpu中创建
with tf.device('gpu'):
    b = tf.range(4)

print(a.device)
print(b.device)

#形状
print(a.shape) # (1,)
#维度信息
print(a.ndim) # [1]

print(a.numpy()) # 1

#判断是不是tensor
print(tf.is_tensor(a)) # True

#查看数据类型
print(a.dtype) # <dtype: 'int32'>
print(a.dtype == tf.float32) # False

#转换成tensor
c = 2
aa = tf.convert_to_tensor(c)
# aa = tf.convert_to_tensor(a, dtype=tf.int32)
print(aa.numpy()) # 2
print(aa.dtype) # <dtype: 'int32'>

#转换数据类型
aaa = tf.cast(aa,dtype = tf.float32)
print(aaa.dtype) # <dtype: 'float32'>
print(aaa.numpy()) # 2.0

## 设置可求导参数
ccc = tf.Variable(aaa)
print(ccc) #<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>



# aa = a.gpu()
# print(aa.device)

# bb = b.cpu()
# print(bb.device)