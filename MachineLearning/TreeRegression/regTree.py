import numpy as np
import pandas as pd


#  获取数据
def load_data(filename):
    data = pd.read_csv(filename, sep='\t', header=None)
    data_set = np.array(data)
    return data_set


# 根据某一特征的特征值划分数据
def bin_split_dataset(data_set, feature, value):
    mat0 = data_set[np.nonzero(data_set[:, feature] > value)[0], :]
    mat1 = data_set[np.nonzero(data_set[:, feature] <= value)[0], :]
    return mat0, mat1


# 判断输入是否是一棵树
def is_tree(obj):
    return type(obj).__name__ == 'dict'


# 回归树叶节点
def reg_leaf(sub_data):
    return np.mean(sub_data[:, -1])


# 回归树总方差
def reg_err(sub_data):
    return np.var(sub_data[:, -1]) * sub_data.shape[0]


# 模型树叶节点
def model_leaf(sub_data):
    w, X, y = linear_solve(sub_data)
    return w


# 模型树总方差
def model_err(sub_data):
    w, X, y = linear_solve(sub_data)
    y_hat = X * w
    return np.sum(np.power(y_hat - y, 2))


# 回归树合并节点
def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['right'] + tree['left']) / 2


# 模型树的线性回归
def linear_solve(data_set):
    m, n = data_set.shape
    X = np.mat(np.ones((m, n)))
    X[:, 1:n] = data_set[:, 0:n-1]  # X的第0列为常数1
    y = np.mat(data_set[:, -1].reshape((-1, 1)))
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        return
    w = xTx.I * X.T * y
    return w, X, y


# 选择划分树最合适的方式
def choose_best_feature_to_split(data_set, leaf_type, err_type, ops=(1, 4)):
    tol_S = ops[0]
    tol_N = ops[1]
    if len(set(data_set[:, -1].tolist())) == 1:
        return None, leaf_type(data_set)
    m, n = data_set.shape
    S = err_type(data_set)
    best_S = float('Inf')
    best_index = 0
    best_value = 0
    for feat_index in range(n - 1):
        for feat_val in set(data_set[:, feat_index]):
            mat0, mat1 = bin_split_dataset(data_set, feat_index, feat_val)
            if mat0.shape[0] < tol_N or mat1.shape[0] < tol_N:
                continue
            new_S = err_type(mat0) + err_type(mat1)
            if new_S < best_S:
                best_index = feat_index
                best_value = feat_val
                best_S = new_S
    if S - best_S < tol_S:
        return None, leaf_type(data_set)
    mat0, mat1 = bin_split_dataset(data_set, best_index, best_value)
    if mat0.shape[0] < tol_N or mat1.shape[0] < tol_N:
        return None, leaf_type(data_set)
    return best_index, best_value


def creat_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    feat, val = choose_best_feature_to_split(data_set, leaf_type, err_type, ops)
    if feat is None:
        return val
    ret_tree = {}
    ret_tree['spInd'] = feat
    ret_tree['spVal'] = val
    l_set, r_set = bin_split_dataset(data_set, feat, val)
    ret_tree['left'] = creat_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = creat_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


# 回归树后剪枝
def prune(tree, test_data):
    if test_data.shape[0] == 0:
        return get_mean(tree)
    l_set, r_set = bin_split_dataset(test_data, tree['spInd'], tree['spVal'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], l_set)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], r_set)
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        error_no_merge = np.sum(np.power(l_set[:, -1] - tree['left'], 2)) \
                         + np.sum(np.power(r_set[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2
        error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            return tree_mean
        else:
            return tree
    return tree


# 回归树预测
def reg_tree_eval(leaf, one_data):
    return float(leaf)


# 模型树预测
def model_tree_eval(leaf, one_data):
    one_data = one_data.reshape((1, -1))
    n = one_data.shape[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1: n+1] = one_data
    return np.dot(X, leaf)


# 一个样本数据的预测
def tree_forecast(tree, one_data, model_eval):
    if not is_tree(tree):
        return model_eval(tree, one_data)
    if one_data[tree['spInd']] > tree['spVal']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'], one_data, model_eval)
        else:
            return model_eval(tree['left'], one_data)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], one_data, model_eval)
        else:
            return model_eval(tree['right'], one_data)


# 所有样本的预测结果
def creat_forecast(tree, test_data, model_eval=reg_tree_eval):
    m = test_data.shape[0]
    y_hat = np.ones((m, 1))
    for i in range(m):
        y_hat[i, 0] = tree_forecast(tree, test_data[i], model_eval)
    return y_hat
