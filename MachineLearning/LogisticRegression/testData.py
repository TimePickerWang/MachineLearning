import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MachineLearning.LogisticRegression.LogRegres as lr


#  获取测试数据
def load_data():
    data = pd.read_csv("./Data/testSet.txt", sep='\t', header=None)
    data.insert(0, 'Ones', 1)  # 第0行插入1
    data_mat = np.array(data.iloc[:, 0: -1])
    class_label = np.array(data.iloc[:, -1]).reshape(-1, 1)  # 转成列向量
    return data_mat, class_label


#  画决策边界
def plot_best_fit(data_mat, class_label, weights):
    data_zero = data_mat[class_label.ravel() == 0, :]  # 类标签为0的数据
    data_one = data_mat[class_label.ravel() == 1, :]  # 类标签为1的数据
    plt.scatter(data_zero[:, 1], data_zero[:, 2], c='R', alpha=0.5)
    plt.scatter(data_one[:, 1], data_one[:, 2], c='G', alpha=0.5)
    x = np.linspace(-3, 3, 2)  # -3与3之间通过2个点作出决策边界
    y = (-weights[0] - weights[1] * x)/weights[2]
    plt.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


data_mat, class_label = load_data()
# weights = lr.grad_ascent(data_mat, class_label)  # 梯度上升算法
weights = lr.stoc_grad_ascent(data_mat, class_label)  # 随机梯度上升算法
print(weights)
plot_best_fit(data_mat, class_label, weights)
