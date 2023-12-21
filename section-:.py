import tensorflow as tf

## 切片 A:B 从A开始到B结束的范围的数据 [A:B)
a = tf.range(10)
print(a) #tf.Tensor([0 1 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32)
print(a[-1:])#tf.Tensor([9], shape=(1,), dtype=int32)
print(a[-2:])#tf.Tensor([8 9], shape=(2,), dtype=int32)
print(a[:2])#tf.Tensor([0 1], shape=(2,), dtype=int32)
print(a[:-1])#tf.Tensor([0 1 2 3 4 5 6 7 8], shape=(9,), dtype=int32)

a1 = tf.random.normal([4,28,28,3])

print(a1[0].shape) # (28, 28, 3)
print(a1[0,:,:,:].shape) # (28, 28, 3)
print(a1[0,1,:,:].shape) # (28, 3)
print(a1[:,:,:,0].shape) # (4, 28, 28)
print(a1[:,:,:,2].shape) # (4, 28, 28)
print(a1[:,0,:,:].shape) # (4, 28, 3)
print(a1[:,0:1,:,:].shape) # (4, 1, 28, 3)

## 切片 A:B:C C为步长 A:B:1 没隔一个数据采样
print(a1[:,::2,::2,:].shape) #(4, 14, 14, 3) 隔行采样
print(a1[:,:14,:14,:].shape) #(4, 14, 14, 3) 裁切采样


## 逆序 ::-1
print(a.numpy())  # [0 1 2 3 4 5 6 7 8 9]
print(a[::-1].numpy()) # [9 8 7 6 5 4 3 2 1 0]
print(a[::-2].numpy()) # [9 7 5 3 1]
print(a[2::-2].numpy()) # [2 0]