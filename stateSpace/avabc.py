import autograd.numpy as np
from autograd import grad
import math
from matplotlib import pyplot as plt
from joblib import Memory
from generate_trajectory import generate_trajectory, generate_observations
memory = Memory(cachedir='joblib_cache',verbose=0)
#import seaborn as sns

'''
@changing: to change this for a new experiment, modify the following:
    simulator()
    generate_variational()
    data()
    inputs to iterate()
'''

def iterate(params, sim_variables, u1, u2, u3, m, v):
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
    grad_lower_bound = grad(lower_bound)
    g = grad_lower_bound(params, sim_variables,u1,u2,u3)
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
    mu = params[0]
    sigma = params[1]
    KL = -1/2*(1+np.log(sigma**2)-mu**2-sigma**2)
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
    E=abc_log_likelihood(theta,sim_variables,u2)
    return E

def abc_log_likelihood(theta,sim_variables,u2):
    '''
    @summary: calculate ABC log-likelihood
    @param sim_variables: simulator variables
    @param theta: draw from variational distribution
    '''
    N = len(v)
    x = simulator(theta,sim_variables,u2)
    e = 0.5
    log_kernels = log_abc_kernel(x,e)
    ll = log_kernels
    return ll

def simulator(theta, sim_variables,u2):
    Gamma = sim_variables[0]
    C = sim_variables[1]
    Sigma = sim_variables[2]
    T = len(u2)/2
    trajectory = generate_trajectory(theta,Gamma, startState,T,u2[:len(u2)/2])
    obs = generate_observations(trajectory,C,Sigma,startState,T,u2[len(u2)/2:])
    return summary_statistics(obs)

def log_abc_kernel(x,e):
    Sx = x
    Sy = data()
    return -np.log(e)-np.log(2*np.pi)/2-np.sum((Sy-Sx)**2)/(2*(e**2))

def summary_statistics(observations):
    print observations[1:-1]
    s1 = np.sum(observations[1:-1])
    s2 = np.sum(observations[1:-2])**2
    s3 = np.sum(observations[1:]*observations[0:-1])
    s4 = observations[0]+observations[-1]
    s5 = observations[0]**2+observations[-1]
    return np.array([s1, s2, s3, s4, s5])

def generate_variational(params,u1):
    '''
    @summary: generate samples from variational distribution
    @param params: variational parameters
    @param u1: samples from another distribution to use for reparametrization
    '''
    return generate_gaussian(params,u1)

def variational_pdf(theta,params):
    return gaussian_pdf(theta,params)

def gaussian_pdf(theta, params):
    mu = params[0]
    sigma = params[1]
    return np.exp(-(theta-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def generate_gaussian(params,u):
    mu=params[0]
    sigma=params[1]
    return mu+sigma*u

@memory.cache
def data():
    A = 1.2
    Gamma = 0.5
    C = 1
    Sigma = 0.1
    T = 100
    startState = 1
    u1 = np.random.normal(0,1,T)
    u2 = np.random.normal(0,1,T)
    trajectory = generate_trajectory(A,Gamma, startState,T,u1)
    observations = generate_observations(trajectory,C,Sigma,startState,T,u2)
    return summary_statistics(observations)

if __name__=='__main__':
    A = 1.2
    Gamma = 0.5
    C = 1
    Sigma = 0.1
    T = 100
    startState = 1
    u1 = np.random.normal(0,1,T)
    u2 = np.random.normal(0,1,T)
    prior_params = [0, 1]
    trajectory = generate_trajectory(A,Gamma, startState,T,u1)
    observations = generate_observations(trajectory,C,Sigma,startState,T,u2)
    params = np.array([1., 1.])
    sim_variables = [Gamma, C, Sigma]
    m = np.array([0.,0.])
    v = np.array([0.,0.])
    i=0
    for i in range(5):
        u1 = np.random.normal(0,1)
        u2 = np.random.uniform(0,1,2*T)
        u3 = np.random.uniform(0,1)
        params = iterate(params, sim_variables, u1, u2, u3, m, v)
    print params
