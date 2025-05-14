import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


datingDataMat, datingLabels = kNN.file2matrix('Database\datingTestSet.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1],
           s = 15.0*array(datingLabels), c = 15.0*array(datingLabels), cmap='viridis')
plt.show()