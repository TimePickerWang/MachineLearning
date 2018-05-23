#encoding=utf-8
import numpy as np


'''    knn算法
Input: inX: 待分类的向量 (1xN)
       dataSet: m个已完成分类的向量 (MxN)
       labels: m个已完成分类向量的类别标签 (Mx1)
       k: 需要比较的最近邻数目

Output:待分类向量最合适的类别
'''
def classify(inX, data_set, labels, k):
    data_size = data_set.shape[0]
    diff_mat = np.tile(inX, (data_size, 1)) - data_set
    sq_distances = np.power(diff_mat, 2).sum(axis=1)
    distances = np.sqrt(sq_distances)
    sorted_distances = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distances[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]


'''    归一化
Input: dataSet: 需要进行归一化的数据

Output:归一化后的结果
'''
def normalization(data_set):
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    ranges = max_value - min_value
    m = data_set.shape[0]
    norm_mat = data_set - np.tile(min_value, (m, 1))
    norm_mat = norm_mat/np.tile(ranges, (m, 1))
    return norm_mat
