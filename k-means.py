#!/usr/bin/python
# -*-coding:utf-8-*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

np.random.seed(1)

"""  
    函数功能：随机初始化中心点  
    X: 数据集 
    k: 初始聚类中心个数
"""
def init_centroids(X, k):
    np.random.seed(1)
    #m为数据集样本个数 n为属性个数
    m, n = X.shape
    #创建（k，n）的零数组
    centroids = np.zeros((k, n))
    #从0到m之间随机选取k个整数值，作为索引
    idx = np.random.randint(0, m, k)
    #将k个索引对应的数据点作为k个初始聚类中心
    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids


"""  
    函数功能：寻找每个样本距离最近的中心点  
    X: 数据集 
    centroids: 初始聚类中心个数
"""
def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    #初始每个样本的对应类别的索引值
    idx = np.zeros(m)
    #误差平方和SSE
    sse = 0
    #遍历整个数据集
    for i in range(m):
        #初始最小距离 设定一个很大的值
        min_dist = 1000000
        #对于每个初始中心点
        for j in range(k):
            #计算样本与中心点的距离
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            #如果距离小于当前最小距离
            if dist < min_dist:
                #最小距离更新为该距离
                min_dist = dist
                #更新该样本的类别索引值为该中心点
                idx[i] = j
        #计算SSE值
        sse += min_dist
    return idx,sse

"""  
    函数功能：更新中心点  
    X: 数据集 
    idx：样本对应类别的索引值
    k：中心点个数
"""
def compute_centroids(X, idx, k):
    m, n = X.shape
    #初始聚类中心（k，n）的零数组
    centroids = np.zeros((k, n))
    #对于每个中心点
    for i in range(k):
        #对于当前中心点类别
        indices = np.where(idx == i)
        #更新其中心点为所有属于该类别的样本点的质心
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    #返回更新后的中心点
    return centroids
"""  
    函数功能：运行k-means聚类算法  
    X: 数据集 
    initial_centroids：初始聚类中心
    max_iters：最大迭代次数
"""
def run_k_means(X, initial_centroids, max_iters):
    global sse
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    #随机的初始聚类中心
    centroids = initial_centroids

    for i in range(max_iters):
        #为每个样本寻找距离最近的中心点
        idx,sse = find_closest_centroids(X, centroids)
        #更新中心点
        centroids = compute_centroids(X, idx, k)

    return idx, centroids,sse



#加载数据集
data = loadmat('data/data.mat')
X = data['X']
data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])

sns.lmplot('X1', 'X2', data=data2, fit_reg=False)
#plt.show()


initial_centroids = init_centroids(X,3)
# print('incenter:',initial_centroids)
idx, centroids,sse = run_k_means(X, initial_centroids, 4)
# print(centroids)
print('误差平方和SSE=',sse)
data2['C'] = idx
#print(data2)
sns.lmplot('X1', 'X2', hue='C', data=data2, fit_reg=False,legend=False)
plt.title('K-Means')
plt.scatter(x=centroids[:,0],y=centroids[:,1],c='r',marker='x')
plt.legend(loc=1)
plt.show()