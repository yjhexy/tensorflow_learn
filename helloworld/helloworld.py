import tensorflow as tf
import numpy as np

# 3000个输入的数据集合
inputX = np.random.rand(3000, 1);

# 学习数据为 接近于 y=4x+1 的点阵，其中有一点点噪声
noise = np.random.normal(0, 0.05, inputX.shape);
outputY = inputX * 4 + 1 + noise;

# 第一层，4个神经单元
# weight1 是最终期望确定的值，所以是变量
weight1 = tf.Variable(np.random.rand(inputX.shape[1], 4))
# bias1 是最终期望确定的值，所以是变量
bias1 = tf.Variable(np.random.rand(inputX.shape[1], 4))
# 是运算过程中被代入的值，所以是占位符
x1 = tf.placeholder(tf.float64, [None, 1])
# 是一个一元线性回归模型
y1_ = tf.matmul(x1, weight1) + bias1

# 第二层，1个神经单元？？
weight2 = tf.Variable(np.random.rand(4,1))
bias2 = tf.Variable(np.random.rand(inputX.shape[1],1))
y2_= tf.matmul(y1_,weight2) + bias2


# 因为训练数据的结果是确定的，一会训练时候feed 进来就可以，所以这里是个占位符
y = tf.placeholder(tf.float64, [None, 1])
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y1_ - y), reduction_indices=[1]));
train = tf.train.GradientDescentOptimizer(0.25).minimize(loss);

init = tf.global_variables_initializer()
sess = tf.Session();
sess.run(init)

for i in range(1000):
    sess.run(train,feed_dict={x1:inputX,y:outputY})

print(weight1.eval(sess))
print("---------------")
print(bias1.eval(sess))
print("---------------")
print(weight2.eval(sess))
print("---------------")
print(bias2.eval(sess))
print("---------------")

x_data = np.matrix([[1.],
                    [2.],
                    [3.]])
print((sess.run(y2_,feed_dict={x1:x_data})))

