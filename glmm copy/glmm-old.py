import autograd.numpy as np
from scipy import stats
from autograd import grad

def likelihood_i(theta,y,u,X_i):
    tau = theta[-1]
    beta = theta[:-1]
    alpha = get_alpha(tau,u)
    w = get_w(beta,tau,alpha,X_i)

def get_w(beta,tau,alpha,X_i):
    py_i_alpha_theta = get_py_i_alpha_theta(beta,alpha,X_i)
    palpha_i_theta = get_palpha_i_theta(tau,alpha)

def get_alpha(tau, u):
    return u*tau

def get_palpha_i_theta(tau,alpha):
    return np.exp(-(alpha)**2/(2*(tau**2)))/(tau*np.sqrt(2*np.pi)) 

def get_py_i_alpha_theta(beta,alpha,X_i):
    p_ij = get_p_ij(beta,alpha,X_i)
    print p_ij
    return np.prod(p_ij)

def bernoulli(p_ij,y):
    return (p_ij**y)*((1-p_ij)**(1-y))

def get_p_ij(beta,alpha,X_i):
    return sigmoid(np.dot(beta,X_i)+alpha_i)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

if __name__=='__main__':
    X_i = np.append(np.ones((1,4)),np.random.rand(2,4),axis=0)
    alpha_i = 2
    beta= np.array([0.5,0.2,0.8])
    py_i_alpha_theta = get_py_i_alpha_theta(beta,alpha_i,X_i)
