import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy.stats import beta, gamma
from scipy import misc
import math
from matplotlib import pyplot as plt
import pdb
#from vbil import BBVI, sample_theta
#from vbil import lower_bound as lower_boundBBVI

all_gradients = []
M=1.
Sy=1.
sigmaData = 1.
e_method = 0.
e_val = 0.5
muPrior = 0.
sigmaPrior = 10.

def iterate(params,i,m,v):
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    u = np.random.normal(0,1,M)
    grad_lower_bound = grad(lower_bound)
    g = grad_lower_bound(params,u,i)

    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    #a = 5*(num_samples**(1./2))*1e-2
    a = 0.05
    params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v

def lower_bound(params,u,i):
    mu = params[0]
    sigma = np.exp(params[1])
    theta = generate_normal(mu,sigma,u)
    E = expectation(params,theta)
    KL = KL_via_sampling(params,theta)
    return E-KL

def expectation(params,theta):
    E = log_likelihood(theta)
    return  np.mean(E)

def log_likelihood(theta):
    return -M*np.log(sigmaData)-M*np.log(np.sqrt(2*np.pi))-np.sum((trueData()-theta)**2)/(2*(sigmaData**2))

def trueData():
    return np.ones(M)

#np.random.seed(5)
#return Sy+sigmaData*np.random.randn(M)

#Correct
def generate_normal(mu,sigma,u):
    X = mu+sigma*u
    return X

#Correct
def normal_pdf(x,mu,sigma):
    return np.exp(-(x-mu)**2/(2*(sigma**2)))/(sigma*np.sqrt(2*np.pi))

#Correct
def KL_via_sampling(params,theta):
    muVB = params[0]
    sigmaVB = np.exp(params[1])
    prior = normal_pdf(theta,muPrior,sigmaPrior)
    vbPosterior = normal_pdf(theta,muVB, sigmaVB)
    return np.mean(np.log(vbPosterior/prior))

if __name__=='__main__':
    #true posterior
    muTrue = (sigmaPrior**2)*Sy/(sigmaData**2+sigmaPrior**2)+(sigmaData**2)*muPrior/(sigmaData**2+sigmaPrior**2)
    sigmaTrue = 1./(1/(sigmaPrior**2)+1/(sigmaData**2))
    params = np.random.uniform(2,4,2)
    
    i=0
    m = np.array([0.,0.])
    v = np.array([0.,0.])
    print params
    for i in range(5000):
        params,m,v = iterate(params,i,m,v)
        print params
    #plot true posterior
    x = np.linspace(muTrue-2*sigmaTrue,muTrue+2*sigmaTrue,100)
    plt.plot(x,normal_pdf(x,muTrue,sigmaTrue))
    plt.plot(x,normal_pdf(x,params[0],np.exp(params[1])))
    plt.show()

