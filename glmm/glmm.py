import autograd.numpy as np
from scipy import stats
from autograd import grad

#right now we have only one time step, also doesn't handle one time step properly, needs to multiply once not 10 times

def likelihood_i(theta,y,u,X_i):
    tau = theta[-1]
    beta = theta[:-1]
    alpha_i = get_alpha(tau,u)
    py_i_alpha_theta = get_py_i_alpha_theta(beta,alpha_i,X_i)
    return np.mean(py_i_alpha_theta)

def get_alpha(tau, u):
    return u*tau
    
def get_py_i_alpha_theta(beta,alpha,X_i):
    p_ij = get_p_ij(beta,alpha,X_i)
    return np.prod(p_ij)

def bernoulli(p_ij,y):
    return (p_ij**y)*((1-p_ij)**(1-y))

def get_p_ij(beta,alpha_i,X_i):
    return sigmoid(np.dot(beta,X_i)+alpha_i)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__=='__main__':
    X = np.append(np.ones((1,4)),np.random.rand(2,4),axis=0)
    X_i = X[:,0]
    y_i = 1
    tau = 0.00001
    beta= np.array([0.5,0.2,0.8])
    theta = np.append(beta,tau)
    u = np.random.rand(10)
    print likelihood_i(theta,y_i,u,X_i)