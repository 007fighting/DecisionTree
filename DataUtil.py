# -*- coding: utf-8 -*-

from numpy import *


class DataUtil:
    def __init__(self):
        pass

    def randomDataSet(self, row, column, classes):
        '''rand data set'''
        if row <= 0 or column <= 0 or classes <= 0:
            return None, None
        dataSet = random.rand(row, column)
        dataLabel = [random.randint(classes)+1 for i in range(row)]
        return dataSet, dataLabel

    def randomDataSet4Int(self, maxinum, row, column, classes):
        '''rand int data set'''
        if row <= 0 or column <= 0 or classes <= 0:
            return None, None
        dataSet = random.randint(maxinum, size=(row, column))
        dataLabel = [random.randint(classes)+1 for i in range(row)]
        return dataSet, dataLabel

    def file2DataSet(self, filePath):
        '''read data set from file'''
        f = open(filePath)
        lines = f.readlines()
        dataSet = None
        dataLabel = []
        i = 0
        for line in lines:
            items = line.strip().split('\t')
            if dataSet is None:
                dataSet = zeros((len(lines), len(items)-1))
            dataSet[i,:] = items[0:-1]
            dataLabel.append(items[-1])
            i += 1
        return dataSet, dataLabel

    def randomX(self, column):
        '''rand a vector'''
        return random.rand(1, column)[0]

    def norm(self, dataSet):
        '''normalize'''
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        m = dataSet.shape[0]
        return (dataSet - tile(minVals, (m, 1)))/tile(ranges, (m, 1))

    def spitData(self, dataSet, dataLabel, ratio):
        '''split data with ratio'''
        totalSize = dataSet.shape[0]
        trainingSize = int(ratio*totalSize)
        testingSize = totalSize - trainingSize

        # random data
        trainingSet = zeros((trainingSize, dataSet.shape[1]))
        trainingLabel = []
        testingSet = zeros((testingSize, dataSet.shape[1]))
        testingLabel = []
        trainingIndex = 0
        testingIndex = 0
        for i in range(totalSize):
            r = random.randint(1, totalSize)
            if (r <= trainingSize and trainingIndex < trainingSize) or testingIndex >= testingSize:
                trainingSet[trainingIndex,:] = dataSet[i,:]
                trainingLabel.append(dataLabel[i])
                trainingIndex += 1
            else:
                testingSet[testingIndex,:] = dataSet[i,:]
                testingLabel.append(dataLabel[i])
                testingIndex += 1
        return trainingSet, trainingLabel, testingSet, testingLabel
