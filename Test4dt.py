# -*- coding: utf-8 -*-
from DataUtil import *
from DecisionTree import *
from matplotlib import pyplot


def decisionTree():
    # variables definition
    MAX_FEATURE_VALUE = 2
    ROW = 5
    COLUMN = 3
    CLASS_COUNT = 3

    # random data
    dt = DecisionTree()
    dataUtil = DataUtil()
    dataSet, dataLabel = dataUtil.randomDataSet4Int(MAX_FEATURE_VALUE, ROW, COLUMN, CLASS_COUNT)
    for i in range(ROW):
        dataSet[i][-1] = dataLabel[i]
    featureNames = ['feature%d' % i for i in range(COLUMN-1)]
    print 'dataSet:'
    print dataSet
    print 'dataLabel:'
    print dataLabel
    
    # plot the data
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:,0], dataSet[:,1], 15*array(dataLabel), 15*array(dataLabel))
    # pyplot.show()
    
    # build decision tree
    print dt.buildTree(dataSet, featureNames)


if __name__ == '__main__':
    decisionTree()
