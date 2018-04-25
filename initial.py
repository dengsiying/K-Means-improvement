#!/usr/bin/python
# -*-coding:utf-8-*-

import random
import math
import numpy as np

random.seed(1)
np.random.seed(1)


"""  
    函数功能：选择初始中心点  
    points: 数据集 
    pNum: 数据集样本个数 
    cNum: 选取聚类中心个数
"""

def initCenters(points, pNum, cNum):
    #初始中心点列表
    centers = []
    #在样本中随机选取一个点作为第一个中心点
    firstCenterIndex = random.randint(0, pNum - 1)
    centers.append(points[firstCenterIndex])
    #初始距离列表
    distance = []
    #对于每个中心点类别
    for cIndex in range(1, cNum):
        #sum为数据集中每个样本点和其最近的中心点的距离和
        sum = 0.0
        #遍历整个数据集
        for pIndex in range(0, pNum):
            #计算每个样本和最近的中心点的距离
            dist = nearest(points[pIndex], centers, cIndex)
            #将距离存到距离列表中
            distance.append(dist)
            #距离相加
            sum += dist
        #随机在（0，sum）中间取一个数值
        ran = random.uniform(0, sum)
        #遍历数据集
        for pIndex in range(0, pNum):
            #ran-=D(x)
            ran -= distance[pIndex]
            if ran > 0: continue
            centers.append(points[pIndex])
            break
    return centers


""" 
    函数功能：计算点和中心之间的最小距离 
    point: 数据点 
    centers: 已经选择的中心
    cIndex: 已经选择的中心个数 
"""


def nearest(point, centers, cIndex):
    #初始一个足够大的最小距离
    minDist = 65536.0
    dist = 0.0
    for index in range(0, cIndex):
        dist = distance(point, centers[index])
        if minDist > dist:
            minDist = dist
    return minDist



""" 
    函数功能：计算点和中心之间的距离 
    point: 数据点 
    center:中心 
     
"""
def distance(point, center):
    dim = len(point)
    if dim != len(center):
        return 0.0
    a = 0.0
    b = 0.0
    c = 0.0
    for index in range(0, dim):
        a += point[index] * center[index]
        b += math.pow(point[index], 2)
        c += math.pow(center[index], 2)
    b = math.sqrt(b)
    c = math.sqrt(c)
    try:
        return a / (b * c)
    except Exception as e:
        print(e)
    return 0.0



