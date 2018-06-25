import numpy as np


'''
Input: data_set: 构建词集的数据

Output:所有文档中出现的不重复词的列表
'''
def creat_vocablist(data_set):
    vocab_set = set()
    for vect in data_set:
        vocab_set = vocab_set | set(vect)
    return list(vocab_set)


'''  词集模型：词集中每个词只能出现一次
Input: vocablist: 词集列表
       input_set: 待转化为向量的文档
Output: 文档转化后的向量，值为1表示出现，0表示没有出现
'''
def setOfWords2vect(vocablist, input_set):
    return_vect = [0] * len(vocablist)
    for word in input_set:
        if word in vocablist:
            return_vect[vocablist.index(word)] = 1
        else:
            print("thr world %s isn't in my vocabulary" % word)
    return return_vect


'''  词袋模型：词袋中每个词可以出现多次
Input: vocablist: 词集列表
       input_set: 待转化为向量的文档
Output: 文档转化后的向量，值大于等于0
'''
def bagOfWords2vect(vocablist, input_set):
    return_vect = [0] * len(vocablist)
    for word in input_set:
        if word in vocablist:
            return_vect[vocablist.index(word)] += 1
        else:
            print("thr world %s isn't in my vocabulary" % word)
    return return_vect


'''
Input: train_data_set:测试集
       train_class_vec:测试集类别列表

Output: 两个类别的概率向量和侮辱性文档概率
'''
def trainNB0(train_data_set, train_class_vec):
    m = len(train_data_set)
    trait_num = len(train_data_set[0])
    p_abusive = sum(train_class_vec) / float(m)  # 侮辱性文档概率
    p0Num = np.ones(trait_num)
    p1Num = np.ones(trait_num)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(m):
        if train_class_vec[i] == 1:
            p1Num += train_data_set[i]
            p1Denom += sum(train_data_set[i])
        else:
            p0Num += train_data_set[i]
            p0Denom += sum(train_data_set[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, p_abusive


'''
Input: vec2classify:待分类的向量
       p0Vect、p1Vect、p_abusive：两个类别的概率向量和侮辱性文档概率
Output: 分类结果
'''
def classifyNB(vec2classify, p0Vect, p1Vect, p_abusive):
    p1 = np.sum(vec2classify * p1Vect) + np.log(p_abusive)
    p0 = np.sum(vec2classify * p0Vect) + np.log(1.0 - p_abusive)
    return 1 if p1 > p0 else 0



'''------------------------------------------------------------------------'''
#  测试数据
def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文档,0代表非侮辱性文档
    return posting_list, class_vec


#  利用测试数据进行分类
def testingNB():
    data_set, class_vec = load_data_set()
    my_vocab_list = creat_vocablist(data_set)  # 词集
    # 将测试数据全部转为只含0,1的数值型矩阵
    train_data_mat = [setOfWords2vect(my_vocab_list, data_set[i]) for i in range(len(data_set))]
    p0_vect, p1_vect, p_Ab = trainNB0(train_data_mat, class_vec)
    test_doc = ['love', 'my', 'dalmation']
    doc_vect = setOfWords2vect(my_vocab_list, test_doc)
    print(str(test_doc) + " classifiy resule:" + str(classifyNB(doc_vect, p0_vect, p1_vect, p_Ab)))
    test_doc = ['stupid', 'garbage']
    doc_vect = setOfWords2vect(my_vocab_list, test_doc)
    print(str(test_doc) + " classifiy resule:" + str(classifyNB(doc_vect, p0_vect, p1_vect, p_Ab)))


# testingNB()
