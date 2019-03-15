import numpy as np
import random

'''
def fit_fun(X):
    A = 1
    return 2 * A + X[0] ** 2 - A * np.cos(2 * np.pi * X[0]) + X[1] ** 2 - A * np.cos(2 * np.pi * X[1])
'''
def fit_fun(X):  # 适应函数
    return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))


class Unit:
    # 初始化
    def __init__(self, x_min, x_max, dim):
        self.__pos = np.array([x_min + random.random()*(x_max - x_min) for i in range(dim)])
        self.__mutation = np.array([0.0 for i in range(dim)])  # 个体突变后的向量
        self.__crossover = np.array([0.0 for i in range(dim)])  # 个体交叉后的向量
        self.__fitnessValue = fit_fun(self.__pos)  # 个体适应度

    def set_pos(self, i, value):
        self.__pos[i] = value

    def get_pos(self):
        return self.__pos

    def set_mutation(self, i, value):
        self.__mutation[i] = value

    def get_mutation(self):
        return self.__mutation

    def set_crossover(self, i, value):
        self.__crossover[i] = value

    def get_crossover(self):
        return self.__crossover

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class DE:
    def __init__(self, dim, size, iter_num, x_min, x_max, best_fitness_value=float('Inf'), F=0.5, CR=0.8):
        self.F = F
        self.CR = CR
        self.dim = dim  # 维度
        self.size = size  # 总群个数
        self.iter_num = iter_num  # 迭代次数
        self.x_min = x_min
        self.x_max = x_max
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  # 全局最优解
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.unit_list = [Unit(self.x_min, self.x_max, self.dim) for i in range(self.size)]

    def get_kth_unit(self, k):
        return self.unit_list[k]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position

    # 变异
    def mutation_fun(self):
        for i in range(self.size):
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(0, self.size - 1)  # 随机数范围为[0,size-1]的整数
                r2 = random.randint(0, self.size - 1)
                r3 = random.randint(0, self.size - 1)
            mutation = self.get_kth_unit(r1).get_pos() + \
                       self.F * (self.get_kth_unit(r2).get_pos() - self.get_kth_unit(r3).get_pos())
            for j in range(self.dim):
                #  判断变异后的值是否满足边界条件，不满足需重新生成
                if self.x_min <= mutation[j] <= self.x_max:
                    self.get_kth_unit(i).set_mutation(j, mutation[j])
                else:
                    rand_value = self.x_min + random.random()*(self.x_max - self.x_min)
                    self.get_kth_unit(i).set_mutation(j, rand_value)

    # 交叉
    def crossover(self):
        for unit in self.unit_list:
            for j in range(self.dim):
                rand_j = random.randint(0, self.dim - 1)
                rand_float = random.random()
                if rand_float <= self.CR or rand_j == j:
                    unit.set_crossover(j, unit.get_mutation()[j])
                else:
                    unit.set_crossover(j, unit.get_pos()[j])

    # 选择
    def selection(self):
        for unit in self.unit_list:
            new_fitness_value = fit_fun(unit.get_crossover())
            if new_fitness_value < unit.get_fitness_value():
                unit.set_fitness_value(new_fitness_value)
                for i in range(self.dim):
                    unit.set_pos(i, unit.get_crossover()[i])
            if new_fitness_value < self.get_bestFitnessValue():
                self.set_bestFitnessValue(new_fitness_value)
                for j in range(self.dim):
                    self.set_bestPosition(j, unit.get_crossover()[j])

    def update(self):
        for i in range(self.iter_num):
            self.mutation_fun()
            self.crossover()
            self.selection()
            self.fitness_val_list.append(self.get_bestFitnessValue())
        return self.fitness_val_list, self.get_bestPosition()
