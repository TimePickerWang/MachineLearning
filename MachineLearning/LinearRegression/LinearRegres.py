import numpy as np
import pandas as pd


#  获取数据, 默认最后一列默认为标签
def load_data(filename):
    """
    param filename: 文件名

    return: x_mat:样本集
            y_mat:标签
    """
    data = pd.read_csv(filename, sep='\t', header=None)
    x_arr = np.array(data.iloc[:, 0:-1])
    y_arr = np.array(data.iloc[:, -1]).reshape(-1, 1)
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    return x_mat, y_mat


# 计算相关系数
def calc_coefficient(y_hyp, y_true):
    """
    param y_hyp: 预测标签
    param y_true: 真实标签

    return: 关系数相
    """
    r = np.corrcoef(y_hyp, y_true)[0, 1]
    return r


# 计算误差
def rss_error(y_hyp, y_true):
    """
    param y_hyp: 预测标签
    param y_true: 真实标签

    return: 误差值
    """
    inner = np.power((y_hyp - y_true), 2)
    return np.sum(inner)


# 标准化
def regularize(x_mat):
    x_mean = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_mean) / x_var
    return x_mat


# 利用最小二乘法
def stan_regress(x_mat, y_mat):
    """
    param x_mat: 训练样本集
    param y_mat: 训练样本标签

    return: w: 权重向量
    """
    xTx = x_mat.T * x_mat
    if np.linalg.det(xTx) == 0:  # 计算行列式，为零时不能求逆
        return
    w = xTx.I * (x_mat.T * y_mat)
    return w


#  局部加权线性回归
def lwlr(test_mat, x_mat, y_mat, k=0.01):
    """
    param test_mat: 测试样本集
    param x_mat: 训练样本集
    param y_mat: 训练样本标签
    param k: 核函数的k取值

    return: y_hyp:预测结果
    """
    test_num = test_mat.shape[0]
    y_hyp = np.zeros((test_num, 1))
    for i in range(test_num):
        test_vect = test_mat[i]
        m = x_mat.shape[0]
        weights = np.mat(np.eye(m))
        for j in range(m):
            diff = test_vect - x_mat[j, :]
            weights[j, j] = np.exp(diff * diff.T / (-2 * k ** 2))
        xTx = (x_mat.T * weights) * x_mat
        if np.linalg.det(xTx) == 0:
            continue
        w = xTx.I * x_mat.T * weights * y_mat
        y_hyp[i] = test_vect * w
    return y_hyp


# 岭回归
def ridge_regress(x_mat, y_mat, lam=0.2):
    """
    param x_mat: 训练样本集
    param y_mat: 训练样本标签
    param lam: lambda

    return: 权重向量
    """
    xTx = x_mat.T * x_mat
    denom = xTx + lam * np.eye(x_mat.shape[1])
    if np.linalg.det(denom) == 0:
        return
    w = denom.I * x_mat.T * y_mat
    return w


#  用30个不同的lambda进行训练，返回由30个权重系数组成的矩阵
def ridge_test(x_mat, y_mat):
    x_mat = regularize(x_mat)
    y_mat = y_mat - np.mean(y_mat, 0)
    test_num = 30
    w_mat = np.zeros((test_num, x_mat.shape[1]))
    for i in range(test_num):
        w = ridge_regress(x_mat, y_mat, np.exp(i-10))
        w_mat[i, :] = w.T
    return w_mat


# 向前逐步回归
def stage_wise(x_mat, y_mat, eps=0.01, iter_num=1000):
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mat = regularize(x_mat)
    m, n = x_mat.shape
    best_w = np.zeros((n, 1))
    lowest_error = float('Inf')
    for i in range(iter_num):
        for j in range(n):
            for sign in [-1, 1]:
                temp_w = best_w.copy()
                temp_w[j] += sign * eps
                y_hyp = x_mat * temp_w
                err = rss_error(y_hyp, y_mat)
                if err < lowest_error:
                    lowest_error = err
                    best_w = temp_w
    return best_w
