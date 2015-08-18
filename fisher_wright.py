import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX
import matplotlib.pyplot as plt

def fisher_wright(x):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    k = x[3]
    N = sharedX(2000.0,name='N')
    p1 = sharedX(0.1,name='N')
    p0 = 1/(1+k*x2/N)
    q = x0*p0/(x0+x1)
    qhat = (x0*(1-p0)+x1*p1)/((x0+x1)*(1-q))
    srng = RandomStreams()
    x0n = srng.binomial(n=N,p=q)
    x1n = srng.binomial(n=N-x0n,p=qhat)
    x2n = N-x0n-x1n
    xOut = T.stack(x0n,x1n,x2n,k)
    return xOut

def fisher_wright_2(x0, x1, x2, k=1.0):
    N = 2000.0
    p1 = 0.1
    p0 = 1/(1+k*x2/N)
    q = x0*p0/(x0+x1)
    qhat = (x0*(1-p0)+x1*p1)/((x0+x1)*(1-q))
    x0n = np.random.binomial(N,q)
    x1n = np.random.binomial(N-x0n,qhat)
    x2n = N-x0n-x1n
    return x0n, x1n, x2n

xStart = T.dvector('xStart')
i = T.iscalar('i')
results, updates = th.scan(fn=fisher_wright, outputs_info=[{'initial':xStart,'taps':[-1]}],n_steps=i)

f=th.function(inputs=[i, xStart],outputs=results,updates=updates)
trajectory = f(100,np.asarray([20.0, 380.0, 1600.0, 1.0], dtype=np.float64))
plt.plot(trajectory[:,0])
plt.plot(trajectory[:,1])
plt.plot(trajectory[:,2])
plt.show()

x0, x1, x2 = 20.0, 380.0, 1600.0
k=1.0
trajectory = []
for i in range(100):
    x0, x1, x2 = fisher_wright_2(x0, x1, x2, k)
    trajectory.append((x0, x1, x2))

trajectory = np.asarray(trajectory)
plt.plot(trajectory[:,0])
plt.plot(trajectory[:,1])
plt.plot(trajectory[:,2])
plt.show()
