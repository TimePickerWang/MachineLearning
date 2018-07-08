import matplotlib.pyplot as plt
from MachineLearning.KMeans.kMeans import *


# 案例一
# data_set = load_data("./Data/testSet.txt")
# centers, cluster_assment = k_means(data_set, 4)
# print(centers)
# plt.scatter(data_set[:, 0], data_set[:, 1], c="r", alpha=0.5)  # 显示数据
# plt.scatter(centers[:, 0], centers[:, 1], c="g", marker='x', alpha=0.5)  # 显示聚类中心
# plt.show()


# 案例二(二分K-Means)
data_set = load_data("./Data/testSet2.txt")
centers, cluster_assment = bi_k_means(data_set, 3)
print(centers)
plt.scatter(data_set[:, 0], data_set[:, 1], c="r", alpha=0.5)  # 显示数据
plt.scatter(centers[:, 0], centers[:, 1], c="g", marker='x', alpha=0.5)  # 显示聚类中心
plt.show()
