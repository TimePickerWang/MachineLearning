from OptAlgorithm.PSO import PSO
from OptAlgorithm.DE import DE
import matplotlib.pyplot as plt
import numpy as np


dim = 2
size = 10
iter_num = 500
x_max = 10
max_vel = 0.05

pso = PSO(dim, size, iter_num, x_max, max_vel)
fit_var_list1, best_pos1 = pso.update()
print("PSO最优位置:" + str(best_pos1))
print("PSO最优解:" + str(fit_var_list1[-1]))
plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list1, c="R", alpha=0.5, label="PSO")

de = DE(dim, size, iter_num, -x_max, x_max)
fit_var_list2, best_pos2 = de.update()
print("DE最优位置:" + str(best_pos2))
print("DE最优解:" + str(fit_var_list2[-1]))
plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list2, c="G", alpha=0.5, label="DE")

plt.legend()  # 显示lebel
plt.show()
