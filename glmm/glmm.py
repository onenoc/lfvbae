import autograd.numpy as np
from scipy import stats
from autograd import grad
#NEED KL FOR LOGNORMAL
#each participant gets a single alpha term per particle (constant across time periods)
#first, don't use alpha but have a placeholder for it
def iterate(params,y,X,i,m,v,num_samples):
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    N = 10
    g = gradient_lower_bound(params,y,X,num_samples,N)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    a = (num_samples**(1./2))
    params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v

def generate_data(beta,tau,n,num_times):
    num_features = len(beta)-1
    X = np.random.uniform(-2,2,(n,num_times,num_features))
    alpha = np.random.normal(0,tau,n)
    alpha = np.reshape(np.tile(alpha,num_times),(num_times,n))
    alpha = np.transpose(alpha)
    P = logistic(beta[0]+np.dot(X,beta[1:])+0*alpha)
    y = np.random.binomial(1,P)
    return X,y

def gradient_lower_bound(params,y,X,num_samples,N):
    eps = np.random.normal(0,1,(num_samples,np.shape(X)[-1]+1))
    n = np.shape(y)[0]
    u = np.random.normal(0,1,num_samples*N*n)
    z = np.random.normal(0,1,num_samples*N*n)
    gradient_lower_bound = grad(lower_bound)
    g = gradient_lower_bound(params,y,X,eps,N,z,u)
    return g

def lower_bound(params,y,X,eps,N,z,u):
    E = expectation(params,y,X,eps,N,z,u)
    tauParams = params[-2:]
    KL = KL_two_gaussians(params)#+KL_two_inv_lognormal(tauParams,u[:N])
    return E#-KL

def expectation(params,y,X,eps,N,z,u):
    mu = params[0:(len(params)-2)/2]
    Sigma = np.exp(params[(len(params)-2)/2:-2])
    tauParams = params[-2:]
    E = 0
    n = X.shape[0]
    for j in range(np.shape(eps)[0]):
        beta = mu+Sigma*eps[j,:]
        E+=log_likelihood(beta,y,X,z[j*(n*N):(j+1)*(n*N)],u[j*(n*N):(j+1)*(n*N)],tauParams,N)
    return E/np.shape(eps)[0]

def KL_two_gaussians(params):
    mu = params[0:(len(params)-2)/2]
    Sigma = np.exp(params[(len(params)-2)/2:-2])
    d = len(mu)
    muPrior = np.zeros(d)
    sigmaPrior = np.ones(d)*50
    return np.sum(np.log(sigmaPrior/Sigma)+(Sigma**2+(mu-muPrior)**2)/(2*(sigmaPrior**2))-1/2)
#mu1 is variational
#mu2 is prior
    #-1/2*np.sum(1+np.log(Sigma**2)-mu**2-Sigma**2)

def KL_two_inv_lognormal(params,u):
    muPrior = np.log(1)
    sigmaPrior = 0.1
    q_samples = 1/generate_lognormal(params,u)
#    print params
#    print muPrior, sigmaPrior
#    print 1/lognormal_pdf(q_samples,np.array([muPrior,sigmaPrior]))
    return np.sum(np.log(lognormal_pdf(q_samples,np.array([muPrior,sigmaPrior]))/lognormal_pdf(q_samples,params)))/len(u)


def log_likelihood(beta, y,X,z,u,tauParams,N):
    ll = 0
    #generate N*n particles
#    inv_lognormal = 1./generate_lognormal(tauParams,u)
#    alpha = np.sqrt(inv_lognormal)*z
#    print np.mean(np.sqrt(inv_lognormal))
    alpha = np.zeros(len(z))
    #iterate over participants
    count = 0
    for i in range(y.shape[0]):
        for j in range(N):
            ll+=log_likelihood_particle(beta,y[i,:],X[i,:,:],alpha[count])
            count += 1
    return ll

#Correct
def generate_lognormal(params,u):
    mu = params[0]
    sigma = np.exp(params[1])
    Y = mu+sigma*u
    X = np.exp(Y)
    return X

#Correct
def lognormal_pdf(theta,params):
    mu=params[0]
    sigma=np.exp(params[1])
    x=theta
    return np.exp(-(np.log(x)-mu)**2/(2*(sigma**2)))/(x*sigma*np.sqrt(2*np.pi))

def log_likelihood_particle(beta, y,X,alpha_i):
    log_likelihood = 0
    pi = get_pi(beta,X,alpha_i)
    likelihood = bernoulli(pi,y)
    log_likelihood = np.sum(np.log(likelihood))
    return log_likelihood

def likelihood_i(beta,yi,xi,alpha_i):
    pi = get_pi(beta,xi,alpha_i)
    return bernoulli(pi,yi)

def get_pi(beta,X,alpha_i):
    return logistic(beta[0]+np.dot(X,beta[1:])+alpha_i)

def bernoulli(pi, yi):
    return (pi**yi)*((1-pi)**(1-yi))

def logistic(x):
    return 1 / (1 + np.exp(-x))

if __name__=='__main__':
    #create some data with beta = 2
    beta = np.array([-2,10])
    d = len(beta)
    params = np.random.normal(0,1,2*d+2)
#    params[-2] = 0
#    params[-1] = 0.1
    #generate_data(beta,tau,n,num_times)
    X,y = generate_data(beta,1.5,50,4)#537
    #test likelihood for several beta values, beta = 2 should give high likelihood
    m = np.zeros(2*d+2)
    v = np.zeros(2*d+2)
    for i in range(150):
        params,m,v =iterate(params,y,X,i,m,v,1)
        mu = params[0:(len(params)-2)/2]
        print mu
        Sigma = params[(len(params)-2)/2:-2]
        #print np.exp(Sigma)
        if np.isnan(params).any():
            params = np.random.normal(0,1,2*d+2)
            m = np.zeros(2*d+2)
            v = np.zeros(2*d+2)
#print 1/np.exp(params[-2])
#    eps = np.random.rand(50)
#    print lower_bound(params,y,X,eps)

