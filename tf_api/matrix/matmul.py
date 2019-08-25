# transpose 练习测试,矩阵相乘
import tensorflow as tf
import numpy as np

data_1 = np.mat([[1, 2, 3, 4],
                 [5, 6, 7, 8]]);

data_2 = np.mat([[1, 2],
                 [1, 2],
                 [1, 2],
                 [1, 2]]);

data_3 = np.mat([[1,1,1,1],
                [2,2,2,2]]);

with tf.Session() as sess:
    z = tf.matmul(data_1,data_2);
    print("转换后:");
    print(sess.run(z));

    # 矩阵3经过 transpose_b 后 实际上和 data_2 是一样的
    zz = tf.matmul(data_1,data_3,transpose_a=False,transpose_b=True);
    print(sess.run(zz));
