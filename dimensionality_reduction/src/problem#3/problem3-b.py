#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:21:04 2019

@author: tong
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

def isomap1(data,n,k):
    X=data.T
    isomap=manifold.Isomap(n_neighbors=k,n_components=n,eigen_solver='auto',path_method='FW',neighbors_algorithm='auto')
    isomap.fit(X)
    X_r=isomap.fit_transform(X)
    return X_r
m = loadmat("isomap.mat")
data=m["images"]
coordinate=isomap1(data,2, 15)
plt.scatter(coordinate[:,0], coordinate[:,1], color='lightblue', marker='+')

def distancematrix(X):
    m,n = X.shape
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n,1))
    return H + H.T - 2*G

"Find three points that are close to each other and show what they look like. "
distance=distancematrix(coordinate.T)

def indmin_matrix(M):
    row,col = divmod(np.argmin(M), np.shape(M)[1])
    return row,col


distance=np.where(distance,distance,100000)

index_nearest=indmin_matrix(distance)
p_1=index_nearest[0]
p_2=index_nearest[1]
list_1=list(distance[p_1,:])
list_2=list(distance[p_2,:])
list_3 =list(np.array(list_1) + np.array(list_2))
p_3=list_3.index(min(list_3))

print("The three points are:")
print(p_1,p_2,p_3)

print("There orinigal data are;")
print(data[:,p_1])
print(data[:,p_2])
print(data[:,p_3])

