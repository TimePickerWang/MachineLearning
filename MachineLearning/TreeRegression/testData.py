import numpy as np
import matplotlib.pyplot as plt
import MachineLearning.TreeRegression .regTree as rt


# 案例一(回归树)
# data_set = rt.load_data("./Data/ex00.txt")
# plt.scatter(data_set[:, 0], data_set[:, 1], c="r", alpha=0.5)
# tree = rt.creat_tree(data_set)
# print(tree)
# plt.show()


# 案例二(回归树)
# data_set = rt.load_data("./Data/ex0.txt")
# plt.scatter(data_set[:, 1], data_set[:, 2], c="r", alpha=0.5)
# tree = rt.creat_tree(data_set)
# print(tree)
# plt.show()


# 案例三(回归树后剪枝)
# data_set = rt.load_data("./Data/ex2.txt")  # 后剪枝训练数据
# plt.scatter(data_set[:, 0], data_set[:, 1], c="r", alpha=0.5)
# test_data = rt.load_data("./Data/ex2test.txt")  # 后剪枝测试数据
# plt.scatter(test_data[:, 0], test_data[:, 1], c="g", alpha=0.5)
# tree = rt.creat_tree(data_set, ops=(0, 1))
# print(tree)
# new_tree = rt.prune(tree, test_data)
# print(new_tree)
# plt.show()


# 案例四(模型树)
# data_set = rt.load_data("./Data/exp2.txt")
# plt.scatter(data_set[:, 0], data_set[:, 1], c="r", alpha=0.5)
# tree = rt.creat_tree(data_set, leaf_type=rt.model_leaf, err_type=rt.model_err, ops=(1, 10))
# print(tree)
# plt.show()


# 案例五(回归树、模型树、一般线性回归的比较,R2标准)
train_data = rt.load_data("./Data/bikeSpeedVsIq_train.txt")  # 训练集
test_data = rt.load_data("./Data/bikeSpeedVsIq_test.txt")  # 测试集

# 回归树
reg_tree = rt.creat_tree(train_data, leaf_type=rt.reg_leaf, err_type=rt.reg_err, ops=(1, 20))
y_hat_reg_tree = rt.creat_forecast(reg_tree, test_data[:, 0:-1], model_eval=rt.reg_tree_eval)
# 模型树
model_tree = rt.creat_tree(train_data, leaf_type=rt.model_leaf, err_type=rt.model_err, ops=(1, 20))
y_hat_model_tree = rt.creat_forecast(model_tree, test_data[:, 0:-1], model_eval=rt.model_tree_eval)
# 一般线性回归
w, _, _ = rt.linear_solve(train_data)
m, n = test_data.shape
test_set = np.ones((m, n))
test_set[:, 1:n] = test_data[:, 0:n-1]
y_hat_linear_reg = test_set * w

print("回归树相关系数:" + str(np.corrcoef(y_hat_reg_tree, test_data[:, 1], rowvar=0)[0, 1]))
print("模型树相关系数:" + str(np.corrcoef(y_hat_model_tree, test_data[:, 1], rowvar=0)[0, 1]))
print("线性回归相关系数:" + str(np.corrcoef(y_hat_linear_reg, test_data[:, 1], rowvar=0)[0, 1]))
# plt.scatter(train_data[:, 0], train_data[:, 1], c="r", alpha=0.5)
# plt.show()
