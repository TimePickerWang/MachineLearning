Machine Learning in Action学习笔记，一个文件夹代表一个算法，每个文件夹包含算法所需的数据集、源码和图片，图片放在pic文件夹中，数据集放在在Data文件夹内。书中的代码是python2的，有不少错误，这里代码是我用python3写的，且都能直接运行


以下是部分实验结果图片

## 1. K-means

![image](https://github.com/TimePickerWang/MachineLearningInAction/blob/master/MachineLearning/KMeans/pic/kMeans_testSet_clustered.png?raw=true)

#### 二分K-means

![image](https://github.com/TimePickerWang/MachineLearningInAction/blob/master/MachineLearning/KMeans/pic/kMeans_testSet2_biKMeans.png?raw=true)

## 2. LinearRegression

#### 线性回归（欠拟合）

![image](https://github.com/TimePickerWang/MachineLearningInAction/blob/master/MachineLearning/LinearRegression/pic/linerRegress_testData.png?raw=true)

#### 局部加权线性回归（k=0.01）

![image](https://github.com/TimePickerWang/MachineLearningInAction/blob/master/MachineLearning/LinearRegression/pic/linerReress_lwlr_k0.01.png?raw=true)

#### 局部加权线性回归(过拟合，k=0.003)

![image](https://github.com/TimePickerWang/MachineLearningInAction/blob/master/MachineLearning/LinearRegression/pic/linerReress_lwlr_k0.003.png?raw=true)

## 3. PCA

#### 红色为原始数据，绿色维降维结果

![image](https://github.com/TimePickerWang/MachineLearningInAction/blob/master/MachineLearning/PCA/pic/pca_testSet.png?raw=true)

## 4.SVM(蓝色标记为支持向量)

#### 线性可分

![image](https://github.com/TimePickerWang/MachineLearningInAction/blob/master/MachineLearning/SVM/pic/svm_textSet_result.png?raw=true)

#### 线性不可分

![image](https://github.com/TimePickerWang/MachineLearningInAction/blob/master/MachineLearning/SVM/pic/svm_testSet2_result.png?raw=true)

#### 利用scikit-learn，准确率可达0.9

![image](https://github.com/TimePickerWang/MachineLearningInAction/blob/master/MachineLearning/SVM/pic/svm_scikitlen_result_.png?raw=true)




