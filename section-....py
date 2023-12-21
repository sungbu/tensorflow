import tensorflow as tf

a = tf.random.normal([4,28,28,3])

# ... 省略‘：’的作用
print(a[0].shape) #(28, 28, 3)
print(a[0,:,:,:].shape) #(28, 28, 3)
print(a[0,...].shape) # (28, 28, 3)


print(a[:,:,:,0].shape) # (4, 28, 28)
print(a[...,0].shape) # (4, 28, 28)


print(a[0,...,2].shape) # (28, 28)

print(a[1,0,...,0].shape) # (28,)