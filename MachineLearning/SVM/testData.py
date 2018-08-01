from sklearn import svm
import matplotlib.pyplot as plt
from MachineLearning.SVM.svm import *


# 案例一:线性可分数据
def test_set_one():
    data, label = load_data("./Data/testSet.txt")

    # b, alphas = smo_simple(data, label, 0.6, 0.001, 40)  # 使用简化版SOM算法
    b, alphas = smo(data, label, 0.4, 0.001, 40)  # 使用完整版SOM算法
    w = calc_w(alphas, data, label)  # 计算权重

    # 绘制数据
    p_data = data[np.nonzero(label == 1)[0], :]  # 正例
    n_data = data[np.nonzero(label != 1)[0], :]  # 反例
    support_veat = data[np.nonzero(alphas > 0)[0], :]  # 支持向量
    plt.scatter(p_data[:, 0], p_data[:, 1], c="g", alpha=.5)
    plt.scatter(n_data[:, 0], n_data[:, 1], c="r", marker='x', alpha=.5)
    plt.scatter(support_veat[:, 0], support_veat[:, 1], c="b", s=100, alpha=.3)
    plt.show()

    print(b)
    print(w)


# 案例二:线性不可分数据
def test_set_two(k=1.3):
    data, label = load_data("./Data/testSetRBF.txt")
    b, alphas = smo(data, label, 200, 0.0001, 10000, ('rbf', k))
    data_mat = np.mat(data)
    label_mat = np.mat(label)
    sv_index = np.nonzero(alphas > 0)[0]  # alpha > 0的索引值
    support_veat = data_mat[sv_index]  # 支持向量
    label_sv = label_mat[sv_index]  # 支持向量对应的标签
    m = data_mat.shape[0]
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(support_veat, data_mat[i, :], ('rbf', k))
        predict = kernel_eval.T * np.multiply(label_sv, alphas[sv_index]) + b
        if np.sign(predict) != np.sign(label_mat[i]):
            error_count += 1
    print("training error rate is: %f" % (float(error_count / m)))
    test_data, test_label = load_data("./Data/testSetRBF2.txt")
    test_mat = np.mat(test_data)
    test_label_mat = np.mat(test_label)
    test_m = test_data.shape[0]
    error_count = 0
    for i in range(test_m):
        kernel_eval = kernel_trans(support_veat, test_mat[i, :], ('rbf', k))
        predict = kernel_eval.T * np.multiply(label_sv, alphas[sv_index]) + b
        if np.sign(predict) != np.sign(test_label_mat[i]):
            error_count += 1
    print("test error rate is: %f" % (float(error_count / m)))

    # 绘制数据
    p_data = data[np.nonzero(label == 1)[0], :]  # 正例
    n_data = data[np.nonzero(label != 1)[0], :]  # 反例
    plt.scatter(p_data[:, 0], p_data[:, 1], c="g", alpha=.5)
    plt.scatter(n_data[:, 0], n_data[:, 1], c="r", marker='x', alpha=.5)
    plt.scatter(support_veat[:, 0].A, support_veat[:, 1].A, c="b", s=100, alpha=.3)
    plt.show()


def scikit_learn_SVM():
    data, label = load_data("./Data/testSetRBF.txt")
    clf = svm.SVC()
    clf.fit(data, label)
    test_data, test_label = load_data("./Data/testSetRBF2.txt")
    score = clf.score(test_data, test_label)  # 平均准确率
    sv = clf.support_vectors_  # 支持向量

    p_data = data[np.nonzero(label == 1)[0], :]  # 正例
    n_data = data[np.nonzero(label != 1)[0], :]  # 反例
    plt.scatter(p_data[:, 0], p_data[:, 1], c="g", alpha=.5)
    plt.scatter(n_data[:, 0], n_data[:, 1], c="r", marker='x', alpha=.5)
    plt.scatter(sv[:, 0], sv[:, 1], c="b", s=100, alpha=.3)
    plt.show()

    print(score)  # 平均准确率


# test_set_one()
# test_set_two()
scikit_learn_SVM()
