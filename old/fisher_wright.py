import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX
import matplotlib.pyplot as plt

def fisher_wright(un1, un2, x, k):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    N = sharedX(2000000.0,name='N')
    p1 = sharedX(0.1,name='N')
    p0 = 1/(1+k*x2/N)
    q = x0*p0/(x0+x1)
    qhat = (x0*(1-p0)+x1*p1)/((x0+x1)*(1-q))
    x0n = binomial(q, un1, T.cast(N, 'int32'))
    x1n = binomial(qhat, un2, T.cast(N-x0n,'int32'))
    x2n = N-x0n-x1n
    xOut = T.stack(x0n,x1n,x2n)
    return xOut

def fisher_wright_normal_approx(v1, v2, x,k):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    N = sharedX(2000.0,name='N')
    p1 = sharedX(0.1,name='N')
    p0 = 1/(1+k*x2/N)
    q = x0*p0/(x0+x1)
    qhat = (x0*(1-p0)+x1*p1)/((x0+x1)*(1-q))
    x0n = N*q+v1*T.sqrt((N*q*(1-q)))
    x1n = (N-x0n)*qhat+v2*T.sqrt(((N-x0n)*qhat*(1-qhat)))
    x2n = N-x0n-x1n
    xOut = T.stack(x0n,x1n,x2n)
    return xOut

def bernoulli(u, p):
    return T.le(u,p)

def binomial(p, un, n):
    results, updates = th.scan(fn=bernoulli, outputs_info=None, sequences=un[0:n], non_sequences=p)
    binomial = results.sum()
    return binomial

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
U1=T.dmatrix('U1')
U2=T.dmatrix('U2')
k = T.dscalar('k')
results, updates = th.scan(fn=fisher_wright, outputs_info=[{'initial':xStart,'taps':[-1]}],sequences=[U1,U2],non_sequences=k, n_steps=i)

v1 = T.dvector('v1')
v2 = T.dvector('v2')
resultsNA, updatesNA = th.scan(fn=fisher_wright_normal_approx, outputs_info=[{'initial':xStart,'taps':[-1]}],sequences=[v1,v2],non_sequences=k, n_steps=i)

f=th.function(inputs=[i, xStart, U1, U2, k],outputs=results,updates=updates)
fNA=th.function(inputs=[i, xStart, v1, v2, k],outputs=resultsNA,updates=updatesNA)
N_fw = 2000000
ir=100.0
U1r = np.random.uniform(0, 1, (ir, N_fw))
U2r = np.random.uniform(0, 1, (ir, N_fw))

v1r = np.random.normal(0,1,ir)
v2r = np.random.normal(0,1,ir)

SS = T.sum(resultsNA**2)

dSS_dk = T.grad(SS,k)
calc_SS = th.function([i, xStart, v1,v2, k],outputs=SS,updates=updates)
gradient_SS = th.function([i, xStart, v1, v2, k],outputs=dSS_dk,updates=updates)

trajectory = f(ir,np.asarray([20000.0, 380000.0, 1600000.0], dtype=np.float64), U1r, U2r, 5.0)
trajectoryNA = fNA(ir,np.asarray([20.0, 380.0, 1600.0], dtype=np.float64),v1r,v2r,5.0)

print (calc_SS(ir,np.asarray([20.0, 380.0, 1600.0], dtype=np.float64), U1r, U2r, 4.50001)-calc_SS(ir,np.asarray([20.0, 380.0, 1600.0], dtype=np.float64), U1r, U2r, 4.5))/0.00001
print gradient_SS(ir,np.asarray([20.0, 380.0, 1600.0], dtype=np.float64), v1r, v2r, 5.0)

sums_SS = []
x_values=[]
for j in range(1000):
    sums_SS.append(calc_SS(ir,np.asarray([20.0, 380.0, 1600.0], dtype=np.float64), v1r, v2r, 4+0.00001*j))
    x_values.append(4+0.00001*j)
    


print trajectory
plt.plot(trajectory[:,0])
plt.plot(trajectory[:,1])
plt.plot(trajectory[:,2])
plt.show()
  
'''
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
'''
