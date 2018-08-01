import random
import numpy as np
import pandas as pd


#  获取数据, 默认最后一列默认为标签
def load_data(filename):
    """
    param filename: 文件名

    return: data:样本集
            label:标签
    """
    data = pd.read_csv(filename, sep='\t', header=None)
    data_set = np.array(data.iloc[:, 0:-1])
    data_label = np.array(data.iloc[:, -1]).reshape(-1, 1)
    return data_set, data_label


# 从0~m中随机选择不等于i的值
def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


# 剪辑alpha
def clip_alphe(alpha, H, L):
    """
    param alpha: 剪辑前的alpha
    param H: alpha的上限
    param L: alpha的下限

    return: 剪辑后的alpha
    """
    if alpha > H:
        alpha = H
    if L > alpha:
        alpha = L
    return alpha


# 简化版SMO算法
def smo_simple(data_set, data_label, C, toler, max_iter):
    """
    param data_set: 样本集
    param data_label: 标签
    param C: 常数
    param toler: 容错率
    param max_iter: 最大迭代次数

    return: b: 偏置
            alphas: alpha向量
    """
    data_mat = np.mat(data_set)
    label_mat = np.mat(data_label)
    m, n = data_mat.shape
    b = 0
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alphas_changed = 0
        for i in range(m):
            f_xi = float(np.multiply(alphas, label_mat).T * (data_mat * data_mat[i, :].T) + b)
            E_i = f_xi - float(label_mat[i])
            if (label_mat[i] * E_i < -toler and alphas[i] < C) or \
                    (label_mat[i] * E_i > toler and alphas[i] > 0):
                j = select_j_rand(i, m)
                f_xj = float(np.multiply(alphas, label_mat).T * (data_mat * data_mat[j, :].T) + b)
                E_j = f_xj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:  # 求alpha_j的上界和下界
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    continue
                eta = data_mat[i, :] * data_mat[i, :].T + data_mat[j, :] * data_mat[j, :].T\
                      - 2 * data_mat[i, :] * data_mat[j, :].T
                if eta <= 0:
                    continue
                alphas[j] += label_mat[j] * (E_i - E_j) / eta  # 求alpha_j
                alphas[j] = clip_alphe(alphas[j], H, L)  # 剪辑alpha_j
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    continue
                alphas[i] += label_mat[i] * label_mat[j] * (alpha_j_old - alphas[j])  # 求alpha_i
                b_i = b - E_i - \
                      label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[i, :].T - \
                      label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[i, :] * data_mat[j, :].T
                b_j = b - E_j - \
                      label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[j, :].T - \
                      label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[j, :] * data_mat[j, :].T
                if 0 < alphas[i] < C:
                    b = b_i
                elif 0 < alphas[j] < C:
                    b = b_j
                else:
                    b = (b_i + b_j) / 2
                alphas_changed += 1
        if alphas_changed == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas


# 计算某个样本和所有样本的核
def kernel_trans(X, A, k_tup):
    m = X.shape[0]
    K = np.mat(np.zeros((m, 1)))
    if k_tup[0] == 'lin':
        K = X * A.T
    elif k_tup[0] == 'rbf':
        for j in range(m):
            delta = X[j, :] - A
            K[j] = delta * delta.T
            K = np.exp(K / (-1 * k_tup[1] ** 2))
    return K


class opt_struct:
    def __init__(self, data_set, class_labels, C, toler, k_tup):
        self.X = data_set
        self.label_mat = class_labels
        self.C = C
        self.tol = toler
        self.m = data_set.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.E_cache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], k_tup)


def calc_Ek(os, k):
    f_xk = float(np.multiply(os.alphas, os.label_mat).T * os.K[:, k] + os.b)
    E_k = f_xk - float(os.label_mat[k])
    return E_k


def updata_Ek(os, k):
    E_k = calc_Ek(os, k)
    os.E_cache[k] = [1, E_k]


def select_j(i, os, E_i):
    max_k = -1
    max_dalta_E = 0
    E_j = 0
    os.E_cache[i] = [1, E_i]
    valid_Ecache_list = np.nonzero(os.E_cache[:, 0])[0]
    if (len(valid_Ecache_list)) > 1:
        for k in valid_Ecache_list:
            if k == i:
                continue
            E_k = calc_Ek(os, k)
            delta_E = abs(E_i - E_k)
            if delta_E > max_dalta_E:
                max_k = k
                max_dalta_E = delta_E
                E_j = E_k
        return max_k, E_j
    else:
        j = select_j_rand(i, os.m)
        E_j = calc_Ek(os, j)
    return j, E_j


# 完整版SMO算法——内循环
def inner_loop(i, os):
    E_i = calc_Ek(os, i)
    if (os.label_mat[i] * E_i < -os.tol and os.alphas[i] < os.C) or \
        (os.label_mat[i] * E_i > os.tol and os.alphas[i] > 0):
        j, E_j = select_j(i, os, E_i)
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()
        if os.label_mat[i] != os.label_mat[j]:  # 求alpha_j的上界和下界
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H:
            return 0
        eta = os.K[i, i] + os.K[j, j] - 2 * os.K[i, j]
        if eta <= 0:
            return 0
        os.alphas[j] += os.label_mat[j] * (E_i - E_j) / eta  # 求alpha_j
        os.alphas[j] = clip_alphe(os.alphas[j], H, L)  # 剪辑alpha
        updata_Ek(os, j)  # 更新E_j

        if abs(os.alphas[j] - alpha_j_old) < 0.00001:
            return 0
        os.alphas[i] += os.label_mat[i] * os.label_mat[j] * (alpha_j_old - os.alphas[j])  # 求alpha_i
        updata_Ek(os, i)  # 更新E_i

        b_i = os.b - E_i - \
              os.label_mat[i] * (os.alphas[i] - alpha_i_old) * os.K[i, i] - \
              os.label_mat[j] * (os.alphas[j] - alpha_j_old) * os.K[i, j]
        b_j = os.b - E_j - \
              os.label_mat[i] * (os.alphas[i] - alpha_i_old) * os.K[i, j] - \
              os.label_mat[j] * (os.alphas[j] - alpha_j_old) * os.K[j, j]
        if 0 < os.alphas[i] < os.C:
            os.b = b_i
        elif 0 < os.alphas[j] < os.C:
            os.b = b_j
        else:
            os.b = (b_i + b_j) / 2
        return 1
    else:
        return 0


# 完整版SMO算法——外循环
def smo(data_set, data_label, C, toler, max_iter, k_tup=('lin', 0)):
    os = opt_struct(np.mat(data_set), data_label, C, toler, k_tup)
    iter = 0
    entire_set = True
    alpha_changed = 0
    while iter < max_iter and (alpha_changed > 0 or entire_set):
        alpha_changed = 0
        if entire_set:
            for i in range(os.m):
                alpha_changed += inner_loop(i, os)
        else:
            non_bound = np.nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in non_bound:
                alpha_changed += inner_loop(i, os)
        iter += 1
        if entire_set:
            entire_set = False
        elif alpha_changed == 0:
            entire_set = True
    return os.b, os.alphas


# 计算权重
def calc_w(alphas, data_set, data_label):
    X = np.mat(data_set)
    lable_mat = np.mat(data_label)
    m, n = data_set.shape
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * lable_mat[i], X[i, :].T)
    return w











