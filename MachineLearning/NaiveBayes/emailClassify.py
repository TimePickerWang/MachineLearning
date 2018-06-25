import re
import random
import numpy as np
import MachineLearning.NaiveBayes.bayes as bayes


# 将一篇邮件解析为词条列表
def text_parse(text):
    list_of_tokens = re.split(r"\W*", text)
    return [token.lower() for token in list_of_tokens if len(token) > 2]


# 转换所有邮件为词条列表，并生成词集
def email_info_transfer():
    doc_list = []  # 转换后的词条集
    class_list = []  # 类标签列表
    for i in range(1, 26):
        token_list = text_parse(open("./Data/spam/%d.txt" % i).read())
        doc_list.append(token_list)
        class_list.append(1)
        token_list = text_parse(open("./Data/ham/%d.txt" % i).read())
        doc_list.append(token_list)
        class_list.append(0)
    my_vocab_list = bayes.creat_vocablist(doc_list)
    return doc_list, class_list, my_vocab_list


#  取不同的训练集和测试集，对邮件进行指定次数的分类，返回错误率的平均值
def email_classify(iter_num):
    doc_list, class_list, my_vocab_list = email_info_transfer()
    error_rate_list = []
    for i in range(iter_num):
        index_of_training_set = [i for i in range(50)]  # 训练集的索引
        index_of_test_set = []  # 测试集的索引
        for k in range(10):  # 取10个样本为测试集
            rand_index = random.randint(0, len(index_of_training_set) - 1)
            index_of_test_set.append(index_of_training_set[rand_index])
            del index_of_training_set[rand_index]
        train_mat = []
        train_class = []
        for index in index_of_training_set:
            train_mat.append(bayes.setOfWords2vect(my_vocab_list, doc_list[index]))
            train_class.append(class_list[index])
        p0_vect, p1_vect, p_Ab = bayes.trainNB0(train_mat, train_class)
        error_count = 0  # 分类错误个数
        for index in index_of_test_set:
            word_vect = bayes.setOfWords2vect(my_vocab_list, doc_list[index])
            if bayes.classifyNB(word_vect, p0_vect, p1_vect, p_Ab) != class_list[index]:
                error_count += 1
        error_rate = float(error_count)/len(index_of_test_set)
        print("第%d次分类的错误个数：%d,错误率:%f" % (i + 1, error_count, error_rate))
        error_rate_list.append(error_rate)
    return np.mean(error_rate_list)


print("平均错误率:" + str(email_classify(10)))
