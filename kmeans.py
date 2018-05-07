#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import random
import re
import matplotlib.pyplot as plt
import matplotlib
import time

def paint(dataArr , centArr , fileName):
    plt.clf()
    if dataArr is not None:
        for item in dataArr:
            plt.plot(item[0] , item[1] , 'or')
    if centArr is not None : 
        for item in centArr:
            plt.plot(item[0] , item[1] , 'dg')
    plt.savefig(fileName)

def calcuDistance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))  #注意这里的减号

def loadDataSet():
    dataSet = np.loadtxt("kMeansTest.csv")
    return dataSet

def initCentroids(dataSet, k): #dk尝试获取四个端点
    m =[ float("inf") , float("inf"), 0 - float("inf") , 0 - float("inf") ] 
    for i in range(2):
        for k in range(len(dataSet)):
            if(m[i] >  dataSet[k][i]): m[i] = dataSet[k][i] 
            if(m[i+2] < dataSet[k][i]) : m[i+2] = dataSet[k][i]
    return np.array([ [ m[0] , m[1] ] ,          [ m[0] , m[3] ] ,    [ m[2] , m[1]  ] ,            [ m[2] , m[3]  ] ])  


def initCentByRand(dataSet, k):
    dataSet = list(dataSet)
    return random.sample(dataSet, k)

def minDistance(dataSet, centroidList):
    # 对每个属于dataSet的item， 计算item与centroidList中k个质心的距离，找出距离最小的，并将item加入相应的簇类中
    rangeCent = range(len(centroidList))
    #clusterDict = dict().fromkeys(rangeCent, []) #dict保存簇类结果
    clusterDict = dict() 
    for i in rangeCent : clusterDict[i] =  []
    sum = 0.0
    for item in dataSet:
        minIdx = -1
        minDis = float("inf") # 初始化为最大值
        for idx in rangeCent:
            distance = calcuDistance(item, centroidList[idx])  
            if distance < minDis:
                minDis = distance
                minIdx = idx
        sum += minDis
        clusterDict[minIdx].append(item)  #加入相应的类别中
    return clusterDict , sum    #不同的类别

def getVar(centroidList, clusterDict): #通过欧氏距离衡量当前中心点最佳, 可以尝试使用多种质心, 对比其中的总欧氏距离
    # 将蔟类中各个向量与质心的距离累加求和
    sum = 0.0
    for key in clusterDict.keys():
        for item in clusterDict[key]:
            sum += calcuDistance(centroidList[key], item)
    return sum

def getCentroids(clusterDict):#重新计算k个质心
    centroidList = []
    for key in clusterDict.keys():
        centroid = np.mean(clusterDict[key], axis=0)
        centroidList.append(centroid)
    return centroidList  

def test(initCent, dataSet, paint = False):
    start = time.time()
    centroidList = initCent(dataSet, 4)
    initCentList = centroidList 
    lastVar = float("inf")
    times= 0
    while True:
        if(paint): paint(dataSet , centroidList , str(times)+ ".png" )
        clusterDict  , s = minDistance(dataSet, centroidList)
        newVar = s # getVar(centroidList, clusterDict)
        if abs(newVar - lastVar) < 0.00001: 
            print "init:" , list(initCentList) , "times:", times ,   "resVar:" , newVar , "use:" , (time.time() - start)
            return newVar
        
        lastVar = newVar
        times += 1
        centroidList = getCentroids(clusterDict)

dataSet = loadDataSet()
test(initCentroids ,dataSet)
random.seed(time.time())
for i in range(1, 100):
    test(initCentByRand , dataSet)

