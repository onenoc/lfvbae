import lfvbaeAlphaStable as lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import math
import sys

def S(alpha,beta):
    inner = 1+(beta**2)*(np.tan(np.pi*alpha/2)**2)
    S = inner**(-1/(2*alpha))
    return S

def B(alpha, beta):
    return np.arctan(beta*np.tan(np.pi*alpha/2))

def alphaStable(alpha,beta,w,u):
    S_var = S(alpha,beta)
    B_var = B(alpha,beta)
    first = np.sin(alpha*(u+B_var)/(np.cos(u)**(alpha/2)))
    second = np.cos(u-alpha*(u+B_var))/w
    return S_var*first*(second**((1-alpha)/alpha))

if __name__=='__main__':
    alpha = 0.5
    beta = 0

    X = []
    for i in range(100):
        w = np.random.exponential()
        u = np.random.uniform(-np.pi/2,np.pi/2)
        X.append(alphaStable(alpha,beta,w,u))
    X = np.asarray(X)
    X = np.reshape(X,(100,1))

    m = 100
    n = 1
    learning_rate = 0.0000001
    dimTheta = 2
    encoder = lfvbae.VA(dimTheta, m, n, learning_rate)
    encoder.initParams()
    encoder.createObjectiveFunction()

    W = np.random.exponential(size=m)
    U = np.random.uniform(-np.pi/2,np.pi/2,m)
    

    V = np.random.normal(0,1,2)
    print encoder.lowerboundfunction(X,W,U,V)

    
