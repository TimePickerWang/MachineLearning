import random
import numpy as np


#  sigmoid函数
def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


#  梯度上升算法
def grad_ascent(data_mat, class_label, iter_num=500, alpha=0.001):
    weights = np.ones((data_mat.shape[1], 1))
    for i in range(iter_num):
        hyp = sigmoid(np.dot(data_mat, weights))
        error = class_label - hyp
        weights += alpha * np.dot(data_mat.transpose(), error)
    return weights


#  随机梯度上升算法
def stoc_grad_ascent(data_mat, class_label, iter_num=500):
    m, n = data_mat.shape
    weights = np.ones(n)
    for j in range(iter_num):
        data_index = [i for i in range(m)]  # 样本的索引值
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            rand_index = random.randint(0, len(data_index) - 1)  # 随机索引值
            h = sigmoid(np.dot(data_mat[rand_index], weights))
            error = class_label[rand_index] - h
            weights += alpha * error * data_mat[rand_index]
            del data_index[rand_index]
    return weights


# 分类函数
def classify(inx, weights):
    prob = sigmoid(np.sum(np.dot(inx, weights)))
    return 1 if prob > 0.5 else 0
