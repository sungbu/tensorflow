import tensorflow as tf

#数据的填充pad
a = tf.reshape(tf.range(9),[3,3])
print(a)

# [[1(上),2(下)],[3(左),4(右)]]
b = tf.pad(a,[[1,2],[3,4]])
print(b)
# tf.Tensor(
# [[0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 1 2 0 0 0 0]
#  [0 0 0 3 4 5 0 0 0 0]
#  [0 0 0 6 7 8 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]], shape=(6, 10), dtype=int32)


a = tf.random.normal([4,28,28,3])
print(a.shape)
# (4, 28, 28, 3)
b = tf.pad(a,[[0,0],[2,2],[2,2],[0,0]])
print(b.shape)
# (4, 32, 32, 3)
print(b[0,...,0])
# tf.Tensor(
# [[ 0.          0.          0.         ...  0.          0.
#    0.        ]
#  [ 0.          0.          0.         ...  0.          0.
#    0.        ]
#  [ 0.          0.         -0.49544364 ...  0.784862    0.
#    0.        ]
#  ...
#  [ 0.          0.          0.85463434 ... -1.1378715   0.
#    0.        ]
#  [ 0.          0.          0.         ...  0.          0.
#    0.        ]
#  [ 0.          0.          0.         ...  0.          0.
#    0.        ]], shape=(32, 32), dtype=float32)
