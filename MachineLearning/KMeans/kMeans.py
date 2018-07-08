import numpy as np
import pandas as pd


#  获取数据
def load_data(filename):
    data = pd.read_csv(filename, sep='\t', header=None)
    data_set = np.array(data)
    return data_set


# 计算两个向量之间的距离
def dist_calc(vec_a, vec_b):
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


# 随机生成k个簇中心
def rand_center(data_set, k):
    n = data_set.shape[1]
    fect_min = np.min(data_set, axis=0)
    fect_max = np.max(data_set, axis=0)
    dect_range = fect_max - fect_min
    k_centers = fect_min + np.multiply(dect_range, np.random.rand(k, n))
    return k_centers


def k_means(data_set, k, dist_meas=dist_calc, creat_cent=rand_center):
    m = data_set.shape[0]
    cluster_assment = np.zeros((m, 2))
    centers = creat_cent(data_set, k)  # 随机生成的k个簇中心
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = float('Inf')
            min_index = -1
            for j in range(k):
                dist_to_j_cent = dist_meas(data_set[i, :], centers[j, :])  # 样本到每个簇中心的距离
                if dist_to_j_cent < min_dist:
                    min_dist = dist_to_j_cent
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i] = min_index, min_dist**2
        for cent in range(k):
            # 更新簇中心
            cluster_members = data_set[cluster_assment[:, 0] == cent]
            centers[cent, :] = np.mean(cluster_members, axis=0)
    return centers, cluster_assment


# 二分K-Means
def bi_k_means(data_set, k, dist_meas=dist_calc):
    m = data_set.shape[0]
    cluster_assment = np.zeros((m, 2))
    center = np.mean(data_set, axis=0)
    center_list = [center]
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(center, data_set[j, :]) ** 2
    while len(center_list) < k:
        lowest_sse = float('Inf')
        best_cent_to_split = -1
        for i in range(len(center_list)):
            data_to_split = data_set[cluster_assment[:, 0] == i]
            data_centers, data_assment = k_means(data_to_split, 2)
            sse_split = np.sum(data_assment[:, 1])
            sse_no_split = np.sum(cluster_assment[cluster_assment[:, 0] != i, 1])
            if sse_split + sse_no_split < lowest_sse:
                best_cent_to_split = i
                best_new_centers = data_centers
                best_cluster_ass = data_assment
                lowest_sse = sse_split + sse_no_split
        best_cluster_ass[best_cluster_ass[:, 0] == 0, 0] = best_cent_to_split
        best_cluster_ass[best_cluster_ass[:, 0] == 1, 0] = len(center_list)
        # 更新簇中心
        center_list[best_cent_to_split] = best_new_centers[0, :]
        center_list.append(best_new_centers[1, :])
        cluster_assment[cluster_assment[:, 0] == best_cent_to_split, :] = best_cluster_ass
    return np.asarray(center_list), cluster_assment
