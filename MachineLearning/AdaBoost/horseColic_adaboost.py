import pandas as pd
from MachineLearning.AdaBoost.boost import *


def colic_test():
    #  获取数据
    train_data = pd.read_csv("./Data/horseColicTraining2.txt", sep='\t', header=None)
    test_data = pd.read_csv("./Data/horseColicTest2.txt", sep='\t', header=None)
    train_mat = np.array(train_data.iloc[:, 0:-1])
    train_label = np.array(train_data.iloc[:, -1]).reshape(-1, 1)
    test_mat = np.array(test_data.iloc[:, 0:-1])
    test_label = np.array(test_data.iloc[:, -1]).reshape(-1, 1)

    # 使用adaboost算法获取分类器列表
    classifier_list = ada_boost_train_ds(train_mat, train_label, iter_num=60)
    result = ada_classify(test_mat, classifier_list)
    m = test_label.shape[0]
    error_rate = np.sum(result != test_label) / m
    print(error_rate)


colic_test()
