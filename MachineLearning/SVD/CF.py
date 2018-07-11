import numpy as np


# 欧氏距离
def euclid_sim(vect_a, vect_b):
    return 1.0 / (1.0 + np.linalg.norm(vect_a - vect_b))


# 皮尔逊相似度
def pears_sim(vect_a, vect_b):
    if len(vect_a) < 3:
        return 1
    return 0.5 + 0.5 * np.corrcoef(vect_a, vect_b, rowvar=0)[0, 1]


# 余弦相似度
def cos_sim(vect_a, vect_b):
    vect_a = vect_a.reshape((-1, 1))
    vect_b = vect_b.reshape((-1, 1))
    num = np.dot(vect_a.T, vect_b)
    denom = np.linalg.norm(vect_a) * np.linalg.norm(vect_b)
    return float(0.5 + 0.5 * (num / denom))


# 对用户某一未评分的项目进行预测
def stand_est(data_set, user, sim_meas, item):
    n = data_set.shape[1]
    sim_total = 0.0
    rat_sim_total = 0.0
    for j in range(n):
        user_rating = data_set[user, j]
        if user_rating == 0:
            continue
        # 两列数据对应位置同时不为0的行索引
        no_zero_index = np.nonzero(np.logical_and(data_set[:, item] > 0, data_set[:, j] > 0))[0]
        if len(no_zero_index) == 0:
            continue
        else:
            similarity = sim_meas(data_set[no_zero_index, item], data_set[no_zero_index, j])
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


# 利用svd分解矩阵后进行推荐
def svd_est(data_set, user, sim_meas, item):
    n = data_set.shape[1]
    sim_total = 0.0
    rat_sim_total = 0.0
    u, sigma, vt = np.linalg.svd(data_set)
    sig4 = np.eye(4) * sigma[:4]
    xformed_items = np.dot(np.dot(data_set.T, u[:, :4]), np.linalg.inv(sig4))
    for j in range(n):
        user_rating = data_set[user, j]
        if user_rating == 0:
            continue
        else:
            similarity = sim_meas(xformed_items[item, :], xformed_items[j, :])
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


def recommend(data_set, user, n=3, sim_meas=cos_sim, es_method=stand_est):
    unrated_items = np.nonzero(data_set[user, :] == 0)[0]
    if len(unrated_items) == 0:
        print("you rated everything")
    item_scores = []
    for item in unrated_items:
        estimated_score = es_method(data_set, user, sim_meas, item)
        item_scores.append((item, estimated_score))
    # 返回预测分数从高到低的前3项
    return sorted(item_scores, key=lambda item_score: item_score[1], reverse=True)[:n]


def loadExData():
    data = [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]
    return np.asarray(data)


data = loadExData()
u, sigma, vt = np.linalg.svd(data)
print(recommend(data, 2, sim_meas=cos_sim, es_method=svd_est))
print(recommend(data, 2, sim_meas=cos_sim, es_method=stand_est))
