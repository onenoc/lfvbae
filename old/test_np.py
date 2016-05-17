import lfvbae_np as lfvbae
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import math

def generate_data(m,n,weight_vector,bias,sigma_e):
    np.random.seed(50)
    X = np.random.uniform(0, 1,(m, n))
    np.random.seed(50)
    e = np.random.normal(0, sigma_e,(m,1))
    if bias:
        X = np.column_stack((X,np.ones(m)))
    dot = np.reshape(np.dot(X,weight_vector), (m,1))
    Y = dot+e
    return Y,X

def true_posterior_standard_normal(n, bias, sigma_e,X,y):
    beta = 1/(sigma_e**2)
    Sinv = np.identity(n+bias)+beta*np.dot(X.T,X)
    S = np.linalg.inv(Sinv)
    muTrue = beta*np.dot(S,np.dot(X.T,y))
    return muTrue,S

if __name__=='__main__':
    m = 20
    n=1
    bias=0
    sigma_e=0.1
    Lu=1
    learning_rate = 0.0001

    iterations = 20000
    y,X = generate_data(m,n,np.array([2]),bias, sigma_e)

    encoder = lfvbae.VA(n+bias, n+bias, m, 1, sigma_e, Lu,learning_rate=learning_rate)

    encoder.initParams()
    print encoder.lowerBoundFunction(X,y,0.1,0.1)
