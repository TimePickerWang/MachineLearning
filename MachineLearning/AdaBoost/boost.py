import numpy as np


# 根据某一特征进行分类
def stump_classify(data_set, dim, threshold, thrash_ineq):
    """
    param data_set: 样本集
    param dim: 特征的某维
    param threshold: 用于比较的阈值
    param thrash_ineq: 不等号，"lt"对应<=,"gt"对应>

    return: ret_array:根据比较特征某一维的大小产生的预测结果
    """
    ret_array = np.ones((data_set.shape[0], 1))
    if thrash_ineq == 'lt':
        ret_array[data_set[:, dim] <= threshold] = -1.0
    else:
        ret_array[data_set[:, dim] > threshold] = -1.0
    return ret_array


# 找到具有最低错误率的单层决策树
def build_stump(data_set, class_label, D):
    """
    param data_set: 样本集
    param class_label: 样本标签向量
    param D: 权重向量

    return: best_stump:一个字典，包含取得最小错误率时单层决策树的信息
            min_error:最小的错误率
            best_class_est:取得最小错误率时的预测结果
    """
    m, n = data_set.shape
    num_steps = 10.0
    best_stump = {}
    best_class_est = np.zeros((m, 1))
    min_error = float('Inf')
    for i in range(n):
        range_min = data_set[:, i].min()
        range_max = data_set[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            threshold = range_min + j * step_size
            for thrash_ineq in ['lt', 'gt']:
                predicted_vals = stump_classify(data_set, i, threshold, thrash_ineq)
                err_arr = np.ones((m, 1))
                err_arr[predicted_vals == class_label] = 0  # 预测对的都置为0
                weighted_err = np.dot(D.T, err_arr)
                if weighted_err < min_error:
                    min_error = weighted_err
                    best_class_est = predicted_vals[:]
                    best_stump["dim"] = i
                    best_stump["threshold"] = threshold
                    best_stump["ineq"] = thrash_ineq
    return best_stump, min_error, best_class_est


# 用多个弱分类器组成一个强分类器
def ada_boost_train_ds(data_set, class_label, iter_num=40):
    """
    param data_set: 样本集
    param class_label: 样本标签向量
    param iter_num: 循环次数(也是弱分类器的个数)

    return: 弱分类器列表
    """
    week_classifier_list = []  # 弱分类器列表
    m = data_set.shape[0]
    D = np.ones((m, 1)) / m
    agg_class_est = np.zeros((m, 1))
    class_label = class_label.reshape(-1, 1)  # 转为列向量
    for i in range(iter_num):
        best_stump, error, class_est = build_stump(data_set, class_label, D)
        # print("D:" + str(D.T))
        alpha = 0.5 * np.log((1-error) / max(error, 1e-16))  # 计算alpha
        best_stump["alpha"] = alpha
        week_classifier_list.append(best_stump)  # 将最佳单层决策树加到单层决策数组
        # print("class_est:" + str(class_est.T))
        expon = alpha * np.multiply(-class_label, class_est)  # 分类跟真实类别相同会是负的
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # 更新权重向量D
        agg_class_est += alpha * class_est  # 更新类别估计值
        # print("agg_class_est:" + str(agg_class_est))
        agg_error = np.multiply(np.sign(agg_class_est) != class_label, np.ones((m, 1)))
        error_rate = np.sum(agg_error) / m
        if error_rate == 0.0:
            break
    return week_classifier_list


# 分类函数
def ada_classify(data_to_class, classifier_list):
    """
    param data_to_class: 待分类的数据
    param classifier_list: 弱分类器列表

    return: 分类结果
    """
    data_to_class = np.array(data_to_class)
    m = data_to_class.shape[0]
    arr_class_est = np.zeros((m, 1))
    for i in range(len(classifier_list)):
        class_est = stump_classify(data_to_class, classifier_list[i]["dim"],
                                   classifier_list[i]["threshold"], classifier_list[i]["ineq"])
        arr_class_est += classifier_list[i]["alpha"] * class_est
        # print(arr_class_est)
    return np.sign(arr_class_est)



'''------------------------------------------------------------------------'''
#  测试数据
def loadSimpData():
    datMat = np.array([[1.,  2.1],
        [2.,  1.1],
        [1.3,  1.],
        [1.,  1.],
        [2.,  1.]])
    classLabels = np.array([[1.0, 1.0, -1.0, -1.0, 1.0]])
    return datMat, classLabels

# data_set, label = loadSimpData()
# week_classifier_list = ada_boost_train_ds(data_set, label, 10)
# result = ada_classify([[0, 0], [3, 3], [-1, 5]], week_classifier_list)
# print(result)
