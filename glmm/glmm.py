import autograd.numpy as np
from autograd import grad

def likelihood_i(theta,y,u):
    tau = theta[-1]
    beta = theta[:-1]
    alpha = get_alpha(tau,u)

def get_alpha(tau, u):
    alpha = 
    
