import autograd.numpy as np
from autograd import grad

def likelihood_i(theta,y,u):
    tau = theta[-1]
    beta = theta[:-1]
    alpha = get_alpha(tau,u)
    w = get_w(beta,alpha)

def get_w(beta,alpha):
    

def get_alpha(tau, u):
    return u*tau

def bernoulli(p,y):
    return (p**y)*((1-p)**(1-y))

def get_p_ij(beta,alpha,x_ij):
    return np.dot(beta,x_ij)+alpha_i
