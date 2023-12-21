import tensorflow as tf
import numpy as np

# 指定数据集文件路径
path_to_mnist = './mnist.npz'

# 使用TensorFlow加载手动下载的MNIST数据集
with np.load(path_to_mnist, allow_pickle=True) as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

# 将数据转换为TensorFlow张量
x = tf.convert_to_tensor(x_train, dtype=tf.float32)
y = tf.convert_to_tensor(y_train, dtype=tf.int32)
# x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
# y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
print(x.shape, y.shape, x.dtype, y.dtype) # (60000, 28, 28) (60000,) <dtype: 'float32'> <dtype: 'int32'>
print(tf.reduce_min(x),tf.reduce_max(x)) # 查看x的最小值和最大值
# tf.Tensor(0.0, shape=(), dtype=float32) tf.Tensor(255.0, shape=(), dtype=float32)
print(tf.reduce_min(y),tf.reduce_max(y)) # 查看y的最小值和最大值
# tf.Tensor(0, shape=(), dtype=int32) tf.Tensor(9, shape=(), dtype=int32)

x = x / 255.
print(tf.reduce_min(x),tf.reduce_max(x)) # 查看x的最小值和最大值
# tf.Tensor(0.0, shape=(), dtype=float32) tf.Tensor(1.0, shape=(), dtype=float32)

train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
print(len(train_db)) # 469 (60000 / 128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:',sample[0].shape,sample[1].shape)
# batch: (128, 28, 28) (128,)


#[b,784] => [b,256] = [b,128] => [b,10]
#[dim_in,dim_out],[dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 0.003
for epoch in range(10):
    for step,(x,y) in enumerate(train_db):
        # x:[128,28,28]
        #y:[128]

        #[b,28,28] => [b,28*28]
        x = tf.reshape(x,[-1,28*28])

        #tensorflow自动求导
        with tf.GradientTape() as tape:
            #x : [b,28*28]
            #h1 = x @ w1 + b1
            #[b,784] => [b,256]
            h1 = x @ w1 + tf.broadcast_to(b1,[x.shape[0],256])
            h1 = tf.nn.relu(h1)
            #[b,784] => [b,128]
            h2 = h1 @ w2 + b2
            h1 = tf.nn.relu(h2)
            #[b,784] => [b,10]
            out = h2 @ w3 + b3

            #计算误差
            #y:[b] => [b,10]
            y_onehot = tf.one_hot(y,depth=10)

            #mse = mean(sum(y-out)^2)
            # [b,10]
            loss = tf.square(y_onehot - out)
            #mean:scalar
            loss = tf.reduce_mean(loss)
        
        #梯度计算
        grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
        # wq = w1 -lr * w1_grad
        # w1 = w1 - lr * grads[0]
        w1.assign_sub(lr * grads[0])
        # b1 = b1 - lr * grads[1]
        b1.assign_sub(lr * grads[1])
        # w2 = w2 - lr * grads[2]
        w2.assign_sub(lr * grads[2])
        # b2 = b2 - lr * grads[3]
        b2.assign_sub(lr * grads[3])
        # w3 = w3 - lr * grads[4]
        w3.assign_sub(lr * grads[4])
        # b3 = b3 - lr * grads[5]
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch,step,'loss',float(loss))