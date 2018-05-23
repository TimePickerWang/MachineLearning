import numpy as np
from os import listdir
import MachineLearning.KNN.KNN as knn
import matplotlib.pyplot as plt

# 2.3使用k-近邻算法识别手写数字


# 将每一个文件转为1*1024的矩阵
def img_to_vector(filename):
    vect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):  # 一个文件32行
        line = fr.readline()
        for j in range(32):  # 一行有32列
            vect[0, i * 32 + j] = int(line[j])
    return vect


# 测试算法
def handwriting_class_test(k=3):
    training_dir = "./Data/trainingDigits/"
    test_dir = "./Data/testDigits/"
    trainingfile_list = listdir(training_dir)
    testfile_list = listdir(test_dir)
    training_num = len(trainingfile_list)
    test_num = len(testfile_list)
    labels = []
    training_mat = np.zeros((training_num, 1024))
    wrong_num = 0
    for i in range(training_num):
        file_name = trainingfile_list[i]
        label = int(file_name.split("_")[0])
        labels.append(label)
        training_mat[i, :] = img_to_vector(training_dir + file_name)
    for i in range(test_num):
        file_name = testfile_list[i]
        test_vector = img_to_vector(test_dir + file_name)
        real_result = int(file_name.split("_")[0])
        classify_result = knn.classify(test_vector, training_mat, labels, k)
        if real_result != classify_result:
            wrong_num += 1
            print("filename: " + file_name + ",wrong result:" +
                  str(classify_result) + ",real_result: " + str(real_result))
    print("error rate:" + str(float(wrong_num/test_num)))
    return float(wrong_num/test_num)


# 绘制k的值从3到k的错误率图像
def error_rate_line(max_k):
    k_list = np.array(range(3, max_k + 1))
    error_rate_list = []
    for k in k_list:
        error_rate_list.append(handwriting_class_test(k))
    plt.plot(k_list, error_rate_list, alpha=0.5)
    plt.title("error rate with different k value")
    plt.show()


handwriting_class_test()
# error_rate_line(15)
