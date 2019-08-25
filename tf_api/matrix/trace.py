# trace 练习测试 矩阵对角线只和
import tensorflow as tf

with tf.Session() as sess:
    a = tf.constant([[1, 5, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])
    z = tf.trace(a)
    print(sess.run(z))
