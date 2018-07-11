import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename, sep='\t'):
    data = pd.read_csv(filename, sep=sep, header=None)
    return np.asarray(data)


def pca(data_set, topn_fect):
    mean_vals = np.mean(data_set, axis=0)
    data_set = data_set - mean_vals  # 去均值
    cov_mat = np.cov(data_set, rowvar=False)  # 协方差矩阵
    eig_vals, eig_vects = np.linalg.eig(cov_mat)  # 协方差矩阵的特征值和特征向量
    eig_valind = np.argsort(eig_vals)
    eig_valind = eig_valind[: -(topn_fect + 1): -1]
    red_eig_vects = eig_vects[:, eig_valind]
    lowdim_data = np.dot(data_set, red_eig_vects)
    recon_set = np.dot(lowdim_data, red_eig_vects.T) + mean_vals
    return lowdim_data, recon_set


# 测试数据
# data = load_data("./Data/testSet.txt")
# low_data, recom_set = pca(data, 1)
# plt.scatter(data[:, 0], data[:, 1], c="r", alpha=0.5)
# plt.scatter(recom_set[:, 0], recom_set[:, 1], c="g", alpha=0.5, marker="^")
# plt.show()
