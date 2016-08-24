import autograd.numpy as np
from scipy import stats
from autograd import grad

def generate_data(beta,num_samples):
    X = np.random.uniform(-2,2,num_samples)
    P = logistic(beta*X)
    y = np.random.binomial(1,P)
    return X,y

def likelihood(beta, y,X):
    likelihood = 1
    for i in range(len(y)):
        likelihood *= likelihood_i(beta,y[i],X[i])
    return likelihood

def likelihood_i(beta,yi,xi):
    pi = get_pi(beta,xi)
    return bernoulli(pi,yi)

def get_pi(beta,xi):
    return logistic(beta*xi)

def bernoulli(pi, yi):
    return (pi**yi)*((1-pi)**(1-yi))

def logistic(x):
    return 1 / (1 + np.exp(-x))

if __name__=='__main__':
    #create some data with beta = 2
    X,y = generate_data(2,100)
    #test likelihood for several beta values, beta = 2 should give high likelihood
    print likelihood(2,y,X)