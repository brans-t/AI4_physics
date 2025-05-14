from numpy import *
import operator

def createDataSet():
    """样本数据"""
    group = array([[1.0,1.1],[0,0],[1.0,1.0],[0,0.1]])
    labels = ['A','B','A','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """kNN algorithm:
    基于训练数据集dataSet和对应的标签labels，对输入的数据集inX进行分类，k为考虑的最近邻
    的个数。
    """
    dataSetSize = dataSet.shape[0]                                #确认训练数据集中的样本数量；shape：这是 NumPy 数组的一个属性，返回一个元组，表示数组的维度
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet          #将输入向量inX复制成与训练数据集相同大小的矩阵，然后再与样本数据相减
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                           #计算欧氏距离；axis=1：沿着行的方向进行操作，即对每一行进行操作
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()                      #对距离进行排序以，并返回数组元素的排序索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]                   #收集前k个最近邻样本标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1   #统计前k个最近邻样本，每个标签出现的次数
    sortedClassCount = sorted(classCount.items(),                    #对标签次数进行降序排列
                            key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    """将文本记录转换为NumPy的解析程序"""
    fr = open(filename)                                         #打开文件并赋值给变量fr
    arrayOLines = fr.readlines()                                #从文件中读取所有行并存储到一个列表中
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))                       #创建一个形状为 numberOfLines x 3 的矩阵，其中所有元素都是 0
    classLabelVector = []                                       #标签列表
    index = 0
    for line in arrayOLines:
        line = line.strip()                                     #去除字符串两端空白字符（包括空格、制表符、换行符等）
        listFromLine = line.split('\t')                         #将 line 按照制表符（\t）分割成一个列表
        returnMat[index, :] = listFromLine[0:3]                 #将 listFromLine 的前三个元素赋值给二维数组（矩阵）returnMat 的第 index 行
        classLabelVector.append(int(listFromLine[-1]))          #将列表最后一个元素转化为整数并存储到 classLabelVector 中
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """归一化特征值"""
    minVals = dataSet.min(0)                                #计算 NumPy 数组 dataSet 的每一列的最小值和最大值;参数 0 表示沿着数组的第 0 轴（即列方向）计算
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))                    #使用 NumPy 创建了一个与 dataSet 形状相同的全零矩阵
    m = dataSet.shape[0]                                   #shape 是 NumPy 数组的一个属性，返回一个元组，包含数组的维度信息;dataSet.shape[0] 表示 dataSet 的行数，即数据点的数量
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals