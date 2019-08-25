# tensor_diag 练习测试
import tensorflow as tf

with tf.Session() as sess:
    diagonal = tf.constant([1, 2, 3, 4], shape=[4]);
    result = tf.linalg.tensor_diag(diagonal);
    print(sess.run(result))
