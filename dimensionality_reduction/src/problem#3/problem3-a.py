# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from scipy.io import loadmat
import numpy as np


m = loadmat("isomap.mat")
data=m["images"]

def distancematrix(X):
    m,n = X.shape
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n,1))
    return H + H.T - 2*G

D=distancematrix(data)

def floyd(D,n_neighbors):
    Max=np.max(D)*1000
    n1,n2=D.shape
    k=n_neighbors
    D1=np.ones((n1,n1))*Max
    D_arg=np.argsort(D,axis=1)
    for i in range(n1):
        D1[i,D_arg[i,0:k+1]]=D[i,D_arg[i,0:k+1]]
    for k in range(n1):
        for i in range(n1):
            for j in range(n1):
                if D1[i,k]+D1[k,j]<D1[i,j]:
                    D1[i,j]=D1[i,k]+D1[k,j]
                    
    return D1

D_f=floyd(D,15)
print("The graph distance matrix is")
print(D)


