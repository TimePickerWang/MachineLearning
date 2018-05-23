import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MachineLearning.KNN.KNN as knn

# 2.2使用k-近邻算法改进约会网站的配对效果


# 读数据
def file_to_matrix(filename):
    #  这里跟书上不同，使用pandas自带的方法用读文件，以sep的值为分隔符，默认为逗号
    data = pd.read_csv(filename, sep='\t', header=None, names=['fly_miles', 'games', 'ice_cream', 'feeling'])
    data_mat = data.iloc[:, 0:3]
    class_label_veaotr = data.iloc[:, 3]
    # DataFrame.as_matrix(columns=None)将DataFrame转为numpy的ndarray
    data_mat = data_mat.as_matrix()
    class_label_veaotr = class_label_veaotr.as_matrix()
    return data_mat, class_label_veaotr


# 显示数据
def show_pic(data_mat, date_label):
    fig = plt.figure(figsize=(12, 10))

    # gemes and ice_cream
    ax = fig.add_subplot(221)
    ax.scatter(data_mat[:, 1].getA1(), data_mat[:, 2].getA1(), 10*np.array(date_label),
                10*np.array(date_label), alpha=0.5)
    ax.set_title('gemes and ice cream')
    ax.legend()

    # fly_miles and games
    ax = fig.add_subplot(222)
    ax.scatter(data_mat[:, 0].getA1(), data_mat[:, 1].getA1(), 10 * np.array(date_label),
                10 * np.array(date_label), alpha=0.5)
    ax.set_title('fly miles and games')
    ax.legend()

    # fly_miles and ice_cream
    ax = fig.add_subplot(223)
    ax.scatter(data_mat[:, 0].getA1(), data_mat[:, 2].getA1(), 10 * np.array(date_label),
                10 * np.array(date_label), alpha=0.5)
    ax.set_title('fly miles and ice cream')
    ax.legend()
    # plt.savefig("../pic/knn_appointment.jpg")
    plt.show()


# 测试算法
def dating_class_test(data, label, k):
    ho_ratio = 0.15  # 取数据集的15%为测试集
    wrong_num = 0
    m = data.shape[0]
    test_data_num = int(m * ho_ratio)
    data_mat = data[test_data_num:m, :]  # 训练集
    label_mat = label[test_data_num:m]
    for i in range(test_data_num):
        test_result = knn.classify(data[i, :], data_mat, label_mat, k)
        real_result = label[i]
        if test_result != real_result:
            wrong_num += 1
            print("wrong result:" + str(test_result) + ",real_result: " + str(real_result))
    error_rate = float(wrong_num/test_data_num)
    return error_rate


# 绘制k的值从3到k的错误率图像
def error_rate_line(max_k):
    k_list = np.array(range(3, max_k + 1))
    error_rate_list = []
    for k in k_list:
        error_rate_list.append(dating_class_test(norm_mat, date_label, k))
    plt.plot(k_list, error_rate_list, alpha=0.5)
    plt.title("error rate with different k value")
    plt.show()


data_mat, date_label = file_to_matrix("./Data/datingTestSet.txt")  # 读数据
# # show_pic(data_mat, date_label)  # 显示数据
norm_mat = knn.normalization(data_mat)  # 归一化
print("error rate:" + str(dating_class_test(norm_mat, date_label, 5)))  # 测试算法4为k的值
# error_rate_line(15)  # 绘制k的值从3到15的错误率图像

