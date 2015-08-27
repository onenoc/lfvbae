import lfvbaeFisherWright as lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import math

def fisher_wright_np(x0, x1, x2, k=1.0):
    N = 2000.0
    p1 = 0.1
    p0 = 1/(1+k*x2/N)
    q = x0*p0/(x0+x1)
    qhat = (x0*(1-p0)+x1*p1)/((x0+x1)*(1-q))
    x0n = np.random.binomial(N,q)
    x1n = np.random.binomial(N-x0n,qhat)
    x2n = N-x0n-x1n
    return x0n, x1n, x2n

if __name__=='__main__':
    n = 1
    bias = 0
    m = 20
    learning_rate = 0.01
    N_fw = 2000.0
    encoder = lfvbae.VA(n, m, learning_rate=learning_rate, i=100)
    encoder.initParams()
    encoder.createObjectiveFunction()
    #print encoder.f2(np.asarray([20.0, 380.0, 1600.0]), 100, 0.5)
    #print encoder.test_f(np.asarray([20.0, 380.0, 1600.0]), 0.1)
    
    i = 100

    x0, x1, x2 = 20.0, 380.0, 1600.0
    k=5.0
    trajectory = []
    for j in range(i):
        x0, x1, x2 = fisher_wright_np(x0, x1, x2, k)
        trajectory.append((x0, x1, x2))
    
    trajectory = np.asarray(trajectory)
    xStart = np.asarray([20.0, 380.0, 1600.0])
    y = trajectory
    v = np.random.normal(0,1)

    U1 = np.random.uniform(0, 1, (i, N_fw))
    U2 = np.random.uniform(0, 1, (i, N_fw))

    #trajectory2 = encoder.create_trajectory(xStart,k)
    for j in range(100):
        encoder.iterate(xStart,y)
    '''
    encoder.lowerboundfunction(xStart, i, y, v, U1, U2)
    encoder.gradientfunction(xStart, i, y, v, U1, U2)
    '''
    #what's wrong is that some values of k don't give valid outputs

