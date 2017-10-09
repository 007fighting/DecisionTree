# -*- coding: utf-8 -*-
from numpy import *
from operator import *


class DecisionTree:
    def __init__(self):
        pass
    
    def calcShannonEnt(self, dataSet):
        '''calculate shannon ent'''
        n = len(dataSet)
        
        # calculate label counts
        labelCounts = {}
        for vec in dataSet:
            if vec[-1] not in labelCounts.keys():
                labelCounts[vec[-1]] = 0
            labelCounts[vec[-1]] += 1
        
        # calculate shannonEnt
        shannonEnt = 0.0
        for label in labelCounts.keys():
            p = float(labelCounts[label]) / n
            shannonEnt -= p * math.log(p, 2)
        return shannonEnt
    
    def splitDataSet(self, dataSet, axis, value):
        '''split data set'''
        subDataSet = []
        if axis >= dataSet.shape[1] - 1:
            return dataSet
        for vec in dataSet:
            if vec[axis] == value:
                tmp = concatenate((vec[:axis], vec[axis+1:]))
                subDataSet.append(tmp)
        return array(subDataSet)
    
    def chooseBestFeatureToSplit(self, dataSet):
        '''choose the best feature to split data set'''
        m = len(dataSet[0]) - 1
        shannonEnt = self.calcShannonEnt(dataSet)
        bestIndex = -1
        bestInfoGain = 0
        for i in range(m):
            values = [vec[i] for vec in dataSet]
            uniqValues = set(values)
            newShannonEnt = 0
            for value in uniqValues:
                subDataSet = self.splitDataSet(dataSet, i, value)
                p = len(subDataSet)*1.0/len(dataSet)
                newShannonEnt += p * self.calcShannonEnt(subDataSet)
            infoGain = shannonEnt - newShannonEnt
            # print 'shannonEnt=%d, newShannonEnt=%d' % (shannonEnt, newShannonEnt)
            # print '%d, infoGain=%d' % (i, infoGain)
            if infoGain > bestInfoGain:
                bestIndex = i
                bestInfoGain = infoGain
        return bestIndex
    
    def majorCnt(self, dataSet):
        if dataSet.shape[0] == 0 or dataSet.shape[1] == 0:
            return -1
        labelCounts = {}
        for vec in dataSet:
            if vec[-1] not in labelCounts.keys():
                labelCounts[vec[-1]] = 0
            labelCounts[vec[-1]] += 1
        sortedCounts = sorted(labelCounts.iteritems(), key=itemgetter(1), reverse=True)
        return sortedCounts[0][0]
    
    def buildTree(self, dataSet, featureNames):
        labels = [vec[-1] for vec in dataSet]
        if labels.count(labels[0]) == len(dataSet):
            return labels[0]
        if dataSet.shape[1] == 1:
            return self.majorCnt(dataSet)
        bestFeature = self.chooseBestFeatureToSplit(dataSet)
        # print 'ddd:'
        # print dataSet
        if bestFeature == -1:
            return self.majorCnt(dataSet)
        bestFeatureName = featureNames[bestFeature]
        # print 'bestFeature=%s' % bestFeatureName
        tree = {bestFeatureName: {}}
        values = [vec[bestFeature] for vec in dataSet]
        uniqValues = set(values)
        del(featureNames[bestFeature])
        for value in uniqValues:
            subDataSet = self.splitDataSet(dataSet, bestFeature, value)
            subFeatureNames = featureNames[:]
            tree[bestFeatureName][value] = self.buildTree(subDataSet, subFeatureNames)
        return tree

    def classify(self, tree, featureNames, x):
        firstStr = tree.keys()[0]
        secondDict = tree[firstStr]
        index = featureNames.index(firstStr)
        for key in secondDict.keys():
            if x[index] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    label = self.classify(secondDict[key], featureNames, x)
                else:
                    label = secondDict[key]
        return label

    def storeTree(self, tree, filePath):
        import pickle
        fw = open(filePath, 'w')
        pickle.dump(tree, fw)
        fw.close()

    def loadTree(self, filePath):
        import pickle
        fr = open(filePath)
        return pickle.load(fr)
