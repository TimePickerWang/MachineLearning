import matplotlib.pyplot as plt
import MachineLearning.LinearRegression.LinearRegres as lr

# 获取数据
x_mat, y_mat = lr.load_data("./Data/ex1.txt")
x_sort = x_mat.copy()
x_sort.sort(0)  # 排序是为了绘图


# w = lr.stan_regress(x_mat, y_mat)  # 利用最小二乘法
# y_hyp = x_sort * w
# print(lr.calc_coefficient((x_mat * w).T, y_mat.T))  # 打印相关系数

y_hyp = lr.lwlr(x_sort, x_mat, y_mat, k=0.01)  # 利用局部加权线性回归


# 绘制数据及拟合曲线
plt.scatter(x_mat[:, 1].flatten().A[0].reshape(1, -1), y_mat.flatten().A[0].reshape(1, -1), s=20, alpha=0.5, c="r")
plt.plot(x_sort[:, 1], y_hyp)
plt.show()
