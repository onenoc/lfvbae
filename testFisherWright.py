import lfvbaeFisherWright as lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import math
import sys

def fisher_wright_np(x0, x1, x2, N,k=1.0):
    p1 = 0.1
    p0 = 1/(1+k*x2/N)
    q = x0*p0/(x0+x1)
    qhat = (x0*(1-p0)+x1*p1)/((x0+x1)*(1-q))
    x0n = np.random.binomial(N,q)
    x1n = np.random.binomial(N-x0n,qhat)
    x2n = N-x0n-x1n
    #print "q is"
    #print q
    #print N*q,N*(1-q)
    #print (N-x0n)*qhat,(N-x0n)*(1-qhat)
    if N*q<5:
        print "Nq too small"
    if N*(1-q)<5:
        print "N*(1-q) too small"
    if (N-x0n)*qhat<5:
        print "(N-x0n)*qhat too small"
    if (N-x0n)*qhat<5:
        print "(N-x0n)*qhat too small"
    return x0n, x1n, x2n

if __name__=='__main__':
    n = 1
    bias = 0
    m = 20
    learning_rate = 0.003
    N_fw = 20000.0
    encoder = lfvbae.VA(n, m, learning_rate=learning_rate, i=100,N_fw=N_fw)
    encoder.initParams()
    encoder.createObjectiveFunction()
    
    i = 100

    x0, x1, x2 = 1*N_fw/5, 1*N_fw/5, 2*N_fw/5
    k=2.0
    trajectory = []
    for j in range(i):
        x0, x1, x2 = fisher_wright_np(x0, x1, x2, N_fw,k)
        trajectory.append((x0, x1, x2))
    
    trajectory = np.asarray(trajectory)
    xStart = np.asarray([x0,x1,x2])
    y = trajectory
    v = np.random.normal(0,1)

    V1 = np.random.uniform(0, 1, i)
    V2 = np.random.uniform(0, 1, i)

    #trajectory2 = encoder.create_trajectory(xStart,k)
    while encoder.converge==0:
        encoder.iterate(xStart,y)

    mu = encoder.params[0].get_value()
    sigma = np.exp(encoder.params[1].get_value())
    lambd = np.exp(encoder.params[2].get_value())

    print "estimate"
    print (6*np.exp(mu)+1)/(np.exp(mu)+1)
    print "sigma"
    print sigma
    print "lambda"
    print "lambda"
    print "95% interval"
    print (6*np.exp(mu+2*sigma)+1)/(np.exp(mu+2*sigma)+1), (6*np.exp(mu-2*sigma)+1)/(np.exp(mu-2*sigma)+1)
    #encoder.lowerboundfunction(xStart, i, y, v, V1, V2)
    #encoder.gradientfunction(xStart, i, y, v, V1, V2)
    #what's wrong is that some values of k don't give valid outputs

