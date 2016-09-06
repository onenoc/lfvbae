import autograd.numpy as np
from scipy import stats
from autograd import grad

def iterate(params,y,X,i,m,v,num_samples):
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    g = gradient_lower_bound(params,y,X,num_samples)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    a = (num_samples**(1./2))*5e-1
    params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v

def generate_data(beta,tau,num_samples,num_times):
    X = np.random.uniform(-2,2,(num_samples,num_times,num_features))
    alpha = np.random.normal(0,tau,num_samples)
    alpha = np.reshape(np.tile(alpha,num_times),(num_times,num_samples))
    alpha = np.transpose(alpha)
    P = logistic(np.dot(X,beta)+alpha)
    y = np.random.binomial(1,P)
    return X,y

def gradient_lower_bound(params,y,X,num_samples):
    eps = np.random.rand(num_samples)
    gradient_lower_bound = grad(lower_bound)
    g = gradient_lower_bound(params,y,X,eps)
    return g

def lower_bound(params,y,X,eps):
    E = expectation(params,y,X,eps)
    KL = KL_via_sampling(params,eps)
    return E-KL

def expectation(params,y,X,eps):
    beta = params[0]+np.exp(params[1])*eps
    E = 0
    for j in range(len(beta)):
        E+= log_likelihood(beta[j],y,X)
    return E/len(beta)

def KL_via_sampling(params,eps):
    theta = params[0]+np.exp(params[1])*eps
    muPrior = 0
    sigmaPrior = 1
    paramsPrior = np.array([muPrior,sigmaPrior])
    E = np.log(normal_pdf(theta,params)/normal_pdf(theta,paramsPrior))
    E = np.mean(E)
    return E

def normal_pdf(theta,params):
    mu = params[0]
    sigma = np.exp(params[1])
    return np.exp(-(theta-mu)**2/(2*sigma**2))/np.sqrt(2*sigma**2*np.pi)

def log_likelihood(beta, y,X):
    log_likelihood = 0
    for i in range(len(y)):
        log_likelihood += np.log(likelihood_i(beta,y[i],X[i]))
    return log_likelihood

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
    params = np.array([0.,0.])
    X,y = generate_data(1,5000)
    #test likelihood for several beta values, beta = 2 should give high likelihood
    m = np.array([0.,0.])
    v = np.array([0.,0.])
    for i in range(100):
        params,m,v =iterate(params,y,X,i,m,v,1)
        print params[0], np.exp(params[1])
#    eps = np.random.rand(50)
#    print lower_bound(params,y,X,eps)
