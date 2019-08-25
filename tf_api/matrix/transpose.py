# transpose 练习测试,作用，矩阵行列互换
import tensorflow as tf
import numpy as np

data = np.mat([[11,12,13,14,15],
               [21,22,23,24,25]]);

print("转换前：");
print(data);
with tf.Session() as sess:
    z = tf.transpose(data);
    print("转换后:");
    print(sess.run(z));
