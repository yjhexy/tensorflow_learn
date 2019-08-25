# diag_part 练习测试
import tensorflow as tf

with tf.Session() as sess:
    a = tf.constant([[1, 5, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])
    z = tf.diag_part(a)
    print(sess.run(z))
