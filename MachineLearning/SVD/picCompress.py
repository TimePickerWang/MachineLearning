import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


def pic_compress(k, pic_array):
    u, sigma, vt = np.linalg.svd(pic_array)
    sig = np.eye(k) * sigma[: k]
    new_pic = np.dot(np.dot(u[:, :k], sig), vt[:k, :])  # 还原图像
    size = u.shape[0] * k + k + k * vt.shape[1]  # 压缩后大小
    return new_pic, size


filename = "./Data/pic1.jpg"
ori_img = np.array(ndimage.imread(filename, flatten=True))
new_img, size = pic_compress(30, ori_img)
print("original size:" + str(ori_img.shape[0] * ori_img.shape[1]))
print("compress size:" + str(size))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(ori_img)
ax[0].set_title("before compress")
ax[1].imshow(new_img)
ax[1].set_title("after compress")
plt.show()
