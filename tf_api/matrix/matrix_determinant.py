# matrix_determinant 练习测试,作用，返回方阵的行列式 .行列式的概念可参考：https://www.jianshu.com/p/0fd8ac349b5e
import tensorflow as tf
import numpy as np

# 方阵
data = np.mat([[11.1,12.1],
               [21.1,22.1]]);

with tf.Session() as sess:
    z = tf.matrix_determinant(data);
    print(sess.run(z));
