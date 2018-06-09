import numpy as np
import pandas as pd
import MachineLearning.LogisticRegression.LogRegres as lr


def colic_test():
    train_data = pd.read_csv("./Data/horseColicTraining.txt", sep='\t', header=None)
    test_data = pd.read_csv("./Data/horseColicTest.txt", sep='\t', header=None)
    train_mat = np.array(train_data.iloc[:, 0:-1])
    train_label = np.array(train_data.iloc[:, -1]).reshape(-1, 1)
    test_mat = np.array(test_data.iloc[:, 0:-1])
    test_label = np.array(test_data.iloc[:, -1]).reshape(-1, 1)
    test_num = len(test_label)
    weights = lr.stoc_grad_ascent(train_mat, train_label)  # 随机梯度下降
    # weights = lr.grad_ascent(train_mat, train_label)  # 梯度下降
    error_num = 0
    for i in range(test_num):
        classify_result = lr.classify(test_mat[i], weights)
        if classify_result != test_label[i]:
            error_num += 1
            # print("分类结果：%d,真实结果：%d" % (classify_result, test_label[i]))
    error_rate = float(error_num)/test_num
    print("错误率是:" + str(error_rate))
    return error_rate


# 进行多次试验，打印错误率均值
def multi_test():
    num_test = 5
    error_sum = 0.0
    for k in range(num_test):
        error_sum += colic_test()
    print("错误率平均是：" + str(error_sum/num_test))


# colic_test()
multi_test()
