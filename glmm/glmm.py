import autograd.numpy as np
from scipy import stats
from autograd import grad
#pass in one uniform AND one Gaussian for each alpha needed
#each participant gets a single alpha term per particle (constant across time periods)
#first, don't use alpha but have a placeholder for it
def iterate(params,y,X,i,m,v,num_samples):
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    N = 1
    g = gradient_lower_bound(params,y,X,num_samples,N)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    a = (num_samples**(1./2))*10e-1
    params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v

def generate_data(beta,tau,n,num_times):
    num_features = len(beta)-1
    X = np.random.uniform(-2,2,(n,num_times,num_features))
    alpha = np.random.normal(0,tau,n)
    alpha = np.reshape(np.tile(alpha,num_times),(num_times,n))
    alpha = np.transpose(alpha)
    P = logistic(beta[0]+np.dot(X,beta[1:])+alpha)
    y = np.random.binomial(1,P)
    return X,y

def gradient_lower_bound(params,y,X,num_samples,N):
    eps = np.random.normal(0,1,(num_samples,np.shape(X)[-1]+1))
    n = np.shape(y)[0]
    u = np.random.uniform(0,1,num_samples*N*n)
    gradient_lower_bound = grad(lower_bound)
    g = gradient_lower_bound(params,y,X,eps,N,u)
    return g

def lower_bound(params,y,X,eps,N,u):
    E = expectation(params,y,X,eps,N,u)
    KL = KL_two_gaussians(params)
    return E-KL

def expectation(params,y,X,eps,N,u):
    mu = params[0:(len(params)-1)/2]
    Sigma = np.exp(params[(len(params)-1)/2:-1])
    tau = params[-1]
    E = 0
    n = X.shape[0]
    for j in range(np.shape(eps)[0]):
        beta = mu+Sigma*eps[j,:]
        E+=log_likelihood(beta,y,X,u[j*(n*N):(j+1)*(n*N)],tau,N)
    return E/len(beta)

def KL_two_gaussians(params):
    mu = params[0:(len(params)-1)/2]
    Sigma = np.diag(np.exp(params[(len(params)-1)/2:-1]))
    muPrior = np.zeros((len(params)-1)/2)
    sigmaPrior = np.identity((len(params)-1)/2)
    return 1/2*(np.log(np.linalg.det(Sigma)/np.linalg.det(sigmaPrior))-d+np.trace(np.dot(np.linalg.inv(Sigma),sigmaPrior))+np.dot(np.transpose(mu-muPrior),np.dot(np.linalg.inv(Sigma),mu-muPrior)))

def log_likelihood(beta, y,X,u,tau,N):
    ll = 0
    #generate N*n particles
    alpha = np.zeros(len(u))
    #iterate over participants
    for i in range(y.shape[0]):
        #iterate over particles
        #ll+=log_likelihood_particle(beta,y[i,:],X[i,:,:])
        for j in range(N):
            ll+=log_likelihood_particle(beta,y[i,:],X[i,:,:],alpha[(i+1)*j])
    return ll

def log_likelihood_particle(beta, y,X,alpha_i):
    log_likelihood = 0
    for i in range(len(y)):
        #iterate over timesteps
        log_likelihood += np.log(likelihood_i(beta,y[i],X[i],alpha_i))
    return log_likelihood

def likelihood_i(beta,yi,xi,alpha_i):
    pi = get_pi(beta,xi,alpha_i)
    return bernoulli(pi,yi)

def get_pi(beta,xi,alpha_i):
    xi = np.insert(xi, 0, 1)
    return logistic(np.dot(beta,xi)+alpha_i)

def bernoulli(pi, yi):
    return (pi**yi)*((1-pi)**(1-yi))

def logistic(x):
    return 1 / (1 + np.exp(-x))

if __name__=='__main__':
    #create some data with beta = 2
    d = 3
    params = np.random.normal(0,1,2*d+1)
    #generate_data(beta,tau,n,num_times)
    X,y = generate_data(np.array([0.5,10]),1,500,4)#537
    #test likelihood for several beta values, beta = 2 should give high likelihood
    m = np.zeros(2*d+1)
    v = np.zeros(2*d+1)
    for i in range(150):
        params,m,v =iterate(params,y,X,i,m,v,1)
        mu = params[0:len(params)/2]
        print mu
#    eps = np.random.rand(50)
#    print lower_bound(params,y,X,eps)

