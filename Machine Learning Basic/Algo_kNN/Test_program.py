import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


datingDataMat, datingLabels = kNN.file2matrix('Database\datingTestSet.txt')  #读取数据集
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)  #归一化特征值

print(normMat)
# 绘制散点图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(normMat[:,1], normMat[:,2],
           s = 15.0*array(datingLabels), c = 15.0*array(datingLabels), cmap='viridis')
plt.show()