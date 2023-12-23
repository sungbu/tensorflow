import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
from tensorflow import keras
import numpy as np

path_to_mnist = './mnist.npz'
batchsz = 128

#数据预处理
def preprocess(x,y):
    #将图片数据控制在0-1之间  [0-255] => [0-1]
    x = tf.cast(x,dtype=tf.float32) / 255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y

#加载数据集
with np.load(path_to_mnist,allow_pickle=True) as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

#将数据转换成张量
x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
x_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test)

#将label变成oneHot格式 [0] => [1,0,0,0,0,0,0,0,0,0]
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

#将数据转换成一个dataset 并且以128为一个匹次
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).shuffle(10000).batch(batchsz)

sample = next(iter(train_db))
print(sample[0].shape)
print(tf.reduce_min(sample[0]),tf.reduce_max(sample[0]))


#自定义layer
class MyDense(layers.Layer):
    def __init__(self,input_dim,outp_dim):
        super(MyDense,self).__init__()

        self.keras = self.add_variable('w',[input_dim,outp_dim])
        # self.bias = self.add_variable('h',[outp_dim])

    def call(self,inputs,training=None):

        x = inputs @ self.keras

        return x
    
#自定义网络
class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork,self).__init__()

        self.fc1 = MyDense(28*28,256)
        self.fc2 = MyDense(256,128)
        self.fc3 = MyDense(128,64)
        self.fc4 = MyDense(64,32)
        self.fc5 = MyDense(32,10)

    def call(self,inputs,training=None):
        x = tf.reshape(inputs,[-1,28*28])
        
        x = self.fc1(x)
        x = tf.nn.relu(x)

        x = self.fc2(x)
        x = tf.nn.relu(x)

        x = self.fc3(x)
        x = tf.nn.relu(x)

        x = self.fc4(x)
        x = tf.nn.relu(x)

        x = self.fc5(x)

        return x
    

#初始化网络
network = MyNetwork()

#网络参数配置
network.compile(optimizer = optimizers.Adam(lr=0.0001),
                loss = tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics = ['accuracy'])

#网络训练
network.fit(train_db,epochs=5,validation_data=test_db,validation_freq=1)

#测试
network.evaluate(test_db)

#模型权值保存
network.save_weights('ckp/weights.ckpt')
#删除模型
del network
print('save to ckpt/weights.ckpt')

#保存模型时没有保存模型所以要加载模型
network = MyNetwork()
network.compile(optimizer = optimizers.Adam(lr=0.0001),
                loss = tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics = ['accuracy'])

network.load_weights('ckp/weights.ckpt')
print('loaded weights from file.')
#测试
network.evaluate(test_db)