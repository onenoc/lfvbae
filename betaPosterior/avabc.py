import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy.stats import beta
from scipy import misc
import math
from matplotlib import pyplot as plt
#import seaborn as sns

'''
@changing: to change this for a new experiment, modify the following:
    simulator()
    generate_variational()
    data()
    inputs to iterate()
'''

def iterate(params, prior_params, sim_variables, u1, u2, u3, m, v):
    '''
    @param params: variational distribution parameters
    @param prior_params: prior parameters
    @sim_variables: simulator variables variables
    @param u1: for reparametrizing variational distribution
    @param u2: for reparametrizing simulator
    @param u3: for reparametrizing KL divergence
    @param m, v: for Adam
    '''
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    grad_lower_bound = grad(lower_bound_binomial)
    g = grad_lower_bound(params,prior_params, sim_variables,u1,u2,u3)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    a = 0.25
    #params = params+a*g
    params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v

def lower_bound(params,sim_variables,u1,u2,u3):
    '''
    @summary: calculate lower bound
    @param params: variational distribution parameters
    @param prior_params: prior parameters
    @param sim_variables: simulator variables
    @param u1: for reparametrizing variational distribution
    @param u2: for reparametrizing simulator
    @param u3: for reparametrizing KL divergence
    '''
    E = expectation(params,sim_variables,u1,u2)
    KL = KL_via_sampling(params,prior_params,1,1,u3)
    return E-KL

def expectation(params,sim_variables, u1, u2):
    '''
    @summary: expectation (estimator) of log-likelihood
    @param params: variational distribution parameters
    @param sim_variables: simulator variables
    @param u1: for reparametrizing variational distribution
    @param u2: for reparametrizing simulator
    '''
    theta = generate_variational(params,u1)
    E=0
    for i in range(len(theta)):
        E+=abc_log_likelihood(sim_variables,theta[i],i,u2)
    return E/len(theta)

def abc_log_likelihood(sim_variables,theta,u2):
    '''
    @summary: calculate ABC log-likelihood
    @param sim_variables: simulator variables
    @param theta: draw from variational distribution
    '''
    N = len(v)
    x = simulator(sim_variables,theta,u2)
    e = sim_variables[0]
    log_kernels = log_abc_kernel(x,n,k,e)
    ll = log_kernels
    return ll

def simulator(sim_variables,theta,u2):
    '''
    @summary: simulator
    @sim_variables: simulator variables/settings that are fixed
    @param theta: global parameter, drawn from variational distribution
    @param u2: controls simulator randomness
    '''
    n = sim_variables[1]
    p = theta
    mu = n*p
    sig2 = np.sqrt(n*p*(1-p))
    gaussian = mu+sig2*u2
    gaussian = np.clip(gaussian,0,n)
    return gaussian

def log_abc_kernel(x,e):
    Sx = x
    Sy = data()
    return -np.log(e)-np.log(2*np.pi)/2-(Sy-Sx)**2/(2*(e**2))

def generate_variational(params,u1):
    '''
    @summary: generate samples from variational distribution
    @param params: variational parameters
    @param u1: samples from another distribution to use for reparametrization
    '''
    return generate_kumaraswamy(params,u1)

def variational_pdf(theta,params):
    return kumaraswamy_pdf(theta,params)

def kumaraswamy_pdf(theta, params):
    a = params[0]
    b = params[1]
    return a*b*(theta**(a-1))*((1-theta**a)**(b-1))

def KL_via_sampling(params, prior_params,u):
    a1 = params[0]
    a2 = params[1]
    theta = generate_variational(params,u)
    E = np.log(

def generate_kumaraswamy(params,u1)
    a=params[0]
    b=params[1]
    return a*b*(theta**(a-1))*((1-theta**a)**(b-1))