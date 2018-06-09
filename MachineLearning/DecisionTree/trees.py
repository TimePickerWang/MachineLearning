from math import log
import pickle


'''    计算香农熵
Input: data_set: 需要计算熵的m个向量，每个向量的最后一位是类别

Output:香农熵的值
'''
def calc_shannonEnt(data_set):
    m = len(data_set)
    lable_counts = {}
    for featVec in data_set:
        current_lable = featVec[-1]
        lable_counts[current_lable] = lable_counts.get(current_lable, 0) + 1
    shannon_ent = 0
    for key in lable_counts:
        prob = float(lable_counts[key])/m
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


'''    划分数据集
Input: data_set: 待划分的数据集
       axis：划分数据集的特征
       value：需要返回的特征的值
Output:划分后的数据集
'''
def split_dataSet(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec += feat_vec[axis+1:]
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


'''    
Input: data_set: 待划分的数据集
       
Output:划分结果最好（划分后信息增益最高）的特征索引
'''
def choose_best_feature_to_split(data_set):
    feature_num = len(data_set[0]) - 1  # 特征的个数
    m = len(data_set)  # 向量个数
    base_entropy = calc_shannonEnt(data_set)  # 经验熵
    best_info_gain = 0.0  # 最好的信息增益值
    best_feature = -1  # 划分后信息增益最高的特征索引值
    for i in range(feature_num):
        # 数据集中所有第i个特征的值存到feat_list中
        feat_list = [example[i] for example in data_set]
        unique_feat = set(feat_list)
        new_entropy = 0.0  # 条件熵
        for feature in unique_feat:
            #  根据第i个特征划分数据
            sub_data_set = split_dataSet(data_set, i, feature)
            prob = len(sub_data_set)/m
            new_entropy += prob * calc_shannonEnt(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


'''
Input: class_list: 分类名称的列表

Output:通过投票机制，返回出现次数最多的分类名称
'''
def majority_label(class_list):
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=lambda item: item[1], reverse=True)
    return sorted_class_count[0][0]


'''
Input: data_set: 待划分的数据集
       labels:标签列表
Output:表示决策树结构的字典
'''
def creat_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    # 若所有的类标签完全相同，直接返回该类标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 若使用完所有特征仍不能将数据划分成包含唯一类别的分组，则通过投票机制选出现次数最多的类别作为返回
    if len(data_set[0]) == 1:
        return majority_label(class_list)
    best_feature = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label: {}}
    del labels[best_feature]
    feature_values = [example[best_feature] for example in data_set]
    unique_feature = set(feature_values)
    for value in unique_feature:
        sub_lables = labels[:]
        my_tree[best_feature_label][value] = creat_tree(
            split_dataSet(data_set, best_feature, value), sub_lables)
    return my_tree


'''
Input: input_tree: 表示决策树结构的字典
       feat_labels: 标签列表
       test_vec：待测试的向量
Output:分类结果
'''
def classify(input_tree, feat_labels, test_vec):
    first_fect = list(input_tree.keys())[0]
    second_dict = input_tree[first_fect]
    feat_index = feat_labels.index(first_fect)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_lable = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_lable = second_dict[key]
    return class_lable


'''    
Input: input_tree: 表示决策树结构的字典
       filename: 需要存储的文件名
'''
#  将字典类型的树结构序列化后存储在文件中
def store_tree(input_tree, filename = "./testTree.txt"):
    fw = open(filename, 'wb')  # 以二进制格式打开一个文件只用于写入。
    pickle.dump(input_tree, fw)
    fw.close()


'''    
Input: filename: 需要读取的文件名

Output: 决策树字典
'''
def grab_tree(filename = "./testTree.txt"):
    fr = open(filename, "rb")
    return pickle.load(fr)



'''------------------------------------------------------------------------'''
#  测试数据
def test_fun():
    data_set = [[0, 1, 0, 0, 'yes'],
                [1, 1, 0, 0, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 1, 0, 'yes'],
                [1, 0, 0, 0, 'yes'],
                [0, 0, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 1, 1, 1, 'no'],
                [0, 0, 0, 1, 'no'],
                [0, 1, 1, 1, 'no'],
                [0, 1, 0, 1, 'no']]
    labels = ["has care", "has house", "has money", "has wife"]
    tree = creat_tree(data_set, labels[:])
    store_tree(tree)


# 分类测试数据
def classify_test_data():
    tree = grab_tree()
    test_vec1 = [1, 0, 1, 0]
    test_vec2 = [0, 1, 0, 1]
    labels = ["has care", "has house", "has money", "has wife"]
    print("分类结果：" + classify(tree, labels, test_vec1))
    print("分类结果：" + classify(tree, labels, test_vec2))


# test_fun()
# classify_test_data()
