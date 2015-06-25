import lfvbae2 as lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

def generate_data(m,n,weight_vector,bias,sigma_e):
    X = np.random.uniform(0, 100,(m, n))
    e = np.random.normal(0, sigma_e,(m,1))
    if bias:
        X = np.column_stack((X,np.ones(m)))
    dot = np.reshape(np.dot(X,weight_vector), (m,1))
    Y = dot+e
    return Y,X

def get_true_posterior(muPrior, sigmaPrior,n,bias, sigma_e):
    alpha = 1./sigmaPrior
    Sinv = np.dot(alpha, sigma_e*np.identity(n+bias))+np.dot(X.T,X)
    S = np.linalg.inv(Sinv)
    muTrue = np.dot(S,np.dot(X.T,Y))
    return muTrue,np.sqrt(S)

m = 20000
n=1
bias=0
sigma_e=0.1

y,X = generate_data(m,n,np.array([-3]),0, sigma_e)


batch = np.column_stack((y,X))

#dimX, dimTheta, m, n
encoder = lfvbae.VA(n+bias, n+bias, m, 1)

encoder.initParams()
encoder.createObjectiveFunction()
for i in range(5000):
    encoder.iterate(batch)
