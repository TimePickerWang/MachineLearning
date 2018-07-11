from MachineLearning.PCA.pca import *


def replace_nan_with_mean(data_set):
    n = data_set.shape[1]
    for i in range(n):
        mean_val = np.mean(data_set[np.nonzero(np.logical_not(np.isnan(data_set[:, i])))[0], i])
        data_set[np.nonzero(np.isnan(data_set[:, i]))[0], i] = mean_val
    return data_set


data = load_data("./Data/secom.data", ' ')
data_set = replace_nan_with_mean(data)
mean_val = np.mean(data_set, axis=0)
mean_removed = data_set - mean_val
cov_set = np.cov(mean_removed, rowvar=False)
eig_vals, eig_vects = np.linalg.eig(cov_set)
# print(eig_vals)

eig_vals_ind = np.argsort(eig_vals)
eig_vals_ind = eig_vals_ind[::-1]
sorted_eig_vals = eig_vals[eig_vals_ind]
total = np.sum(sorted_eig_vals)
var_percentage = sorted_eig_vals/total * 100
plt.plot(var_percentage[:20], marker='o', alpha=.5, c="r")
plt.xlabel("Principal Component Number")
plt.ylabel("Percentage of Variance")
plt.show()
