import numpy as np
import matplotlib.pyplot as plt
import MachineLearning.LinearRegression.LinearRegres as lr

x_mat, y_mat = lr.load_data("./Data/abalone.txt")

# 岭回归
# w_mat = lr.ridge_test(x_mat, y_mat)
# plt.plot(w_mat)
# plt.show()


# 向前逐步回归
w0 = lr.stage_wise(x_mat, y_mat, eps=0.005, iter_num=1000)
print(w0)

# 利用最小二乘法
y_mean = np.mean(y_mat, 0)
y_mat = y_mat - y_mean
x_mat = lr.regularize(x_mat)
w1 = lr.stan_regress(x_mat, y_mat)
print(w1)


'''
# 用不同的数据和k值计算误差
y_hyp01_train = lr.lwlr(x_mat[0:99], x_mat[0:99], y_mat[0:99], 0.1)
y_hyp1_train = lr.lwlr(x_mat[0:99], x_mat[0:99], y_mat[0:99], 1)
y_hyp10_train = lr.lwlr(x_mat[0:99], x_mat[0:99], y_mat[0:99], 10)

y_hyp01_test = lr.lwlr(x_mat[100:199], x_mat[0:99], y_mat[0:99], 0.1)
y_hyp1_test = lr.lwlr(x_mat[100:199], x_mat[0:99], y_mat[0:99], 1)
y_hyp10_test = lr.lwlr(x_mat[100:199], x_mat[0:99], y_mat[0:99], 10)

cost_on_train_01 = lr.rss_error(y_hyp01_train, y_mat[0:99])
cost_on_train_1 = lr.rss_error(y_hyp1_train, y_mat[0:99])
cost_on_train_10 = lr.rss_error(y_hyp10_train, y_mat[0:99])

cost_on_test_01 = lr.rss_error(y_hyp01_test, y_mat[100:199])
cost_on_test_1 = lr.rss_error(y_hyp1_test, y_mat[100:199])
cost_on_test_10 = lr.rss_error(y_hyp10_test, y_mat[100:199])

print("k=0.1时,训练集代价:" + str(cost_on_train_01) + ",测试集代价:" + str(cost_on_test_01))
print("k=1时,训练集代价:" + str(cost_on_train_1) + ",测试集代价:" + str(cost_on_test_1))
print("k=10时,训练集代价:" + str(cost_on_train_10) + ",测试集代价:" + str(cost_on_test_10))
'''