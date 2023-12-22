import tensorflow as tf
import os

def accuracy(output,target,topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output,maxk).indices
    pred = tf.transpose(pred,perm=[1,0])
    target_ = tf.broadcast_to(target,pred.shape)

    #[10,b]
    correct = tf.equal(pred,target_)

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k],[-1]),dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k * (100.0 / batch_size))
        res.append(acc)
    return res

output = tf.random.normal([10,6])
output = tf.math.softmax(output,axis=1)
target = tf.random.uniform([10],maxval=6,dtype=tf.int32)
print('prob:',output.numpy())
pred = tf.argmax(output,axis=1)
print('pred:',pred.numpy())
print('label:',target.numpy())

acc = accuracy(output,target,topk=(1,2,3,4,5,6))
print('top-1-6 acc',acc)

# prob: [[0.28637904 0.20587227 0.18056783 0.08008153 0.13776866 0.10933069]
#  [0.01082122 0.04127875 0.8393643  0.03012516 0.01395684 0.06445378]
#  [0.13295695 0.22558175 0.10422832 0.2584538  0.1907562  0.08802313]
#  [0.25487557 0.04269523 0.02212623 0.12857896 0.09117445 0.46054956]
#  [0.24628678 0.14641151 0.2731334  0.03323939 0.05849626 0.24243265]
#  [0.10737384 0.05534569 0.02467335 0.5765771  0.10618355 0.12984642]
#  [0.07471579 0.20492043 0.0268361  0.02339358 0.6375157  0.03261831]
#  [0.03507492 0.02529469 0.36519504 0.0288283  0.33027613 0.21533091]
#  [0.03851747 0.10186154 0.16496183 0.04658286 0.27506927 0.37300715]
#  [0.29671708 0.12218311 0.08439651 0.09917189 0.0376533  0.35987806]]
# pred: [0 2 3 5 2 3 4 2 5 5]
# label: [4 3 0 3 1 2 1 4 2 2]
# top-1-6 acc [0.0, 20.0, 40.0, 80.0, 90.0, 100.0]