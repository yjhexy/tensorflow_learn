import numpy as np

# 3000行 ，1列的数据
inputX = np.random.rand(3000, 1);
print(inputX)
print(inputX.shape)

# shape[1] = 1 ,所以是1行4列的数据
weight1 = np.random.rand(inputX.shape[1], 4)

print(inputX)
print(weight1)
