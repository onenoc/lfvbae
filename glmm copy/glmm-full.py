import autograd.numpy as np
from scipy import stats
from autograd import grad
#need to sample N alphas per participant
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
    a = (num_samples**(1./2))*5e-1
    params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v

def generate_data(beta,tau,n,num_times):
    num_features = len(beta)-1
    X = np.random.uniform(-2,2,(n,num_times,num_features))
    alpha = np.random.normal(0,tau,n)
    alpha = np.reshape(np.tile(alpha,num_times),(num_times,n))
    alpha = np.transpose(alpha)
    P = logistic(beta[0]+np.dot(X,beta[1:]))#+alpha)
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
    #for each sample of theta, calculate likelihood
    #likelihood has participants
    #for each participant, we have N particles
    #with L samples, n participants, N particles per participant and sample, we have
    #L*n*N particles
    #get the first column to be mu
    d = np.shape(X)[-1]+1
    mu = params[0:d,0]
    toSigma = params[0:d,1:d+1]
    intSigma = toSigma-np.diag(np.diag(toSigma))+np.diag(np.exp(np.diag(toSigma)))
    Sigma = intSigma-np.tril(intSigma)+np.transpose(np.triu(intSigma))
    print mu
    print Sigma
    n = X.shape[0]
    E = 0
    #iterate over number of samples of theta
    for j in range(np.shape(eps)[0]):
        beta = mu+np.dot(Sigma,eps[j,:])
        #this log likelihood will iterate over both the participants and the particles
        E+=log_likelihood(beta,y,X,u[j*(n*N):(j+1)*(n*N)])
    return E/len(beta)

def KL_two_gaussians(params):
    d = np.shape(params)[0]-1
    mu = params[0:d,0]
    toSigma = params[0:d,1:d+1]
    intSigma = toSigma-np.diag(np.diag(toSigma))+np.diag(np.exp(np.diag(toSigma)))
    Sigma = intSigma-np.tril(intSigma)+np.transpose(np.triu(intSigma))
    muPrior = np.zeros(d)
    sigmaPrior = np.identity(d)
    #print Sigma
    #print np.linalg.det(Sigma)
    return 1/2*(np.log(np.linalg.det(Sigma)/np.linalg.det(sigmaPrior))-d+np.trace(np.dot(np.linalg.inv(Sigma),sigmaPrior))+np.dot(np.transpose(mu-muPrior),np.dot(np.linalg.inv(Sigma),mu-muPrior)))

def KL_via_sampling(params,eps):
    #also need to include lognormal as a replacement for gamma distribution
    #this is giving log of negatives
    d = np.shape(params)[0]-1
    mu = params[0:d,0]
    Sigma = params[0:d,1:d+1]
    di = np.diag_indices(d)
    Sigma[di] = np.exp(Sigma[di])
    muPrior = np.zeros(d)
    sigmaPrior = np.identity(d)
    E = 0
    for j in range(np.shape(eps)[0]):
        beta = mu+np.dot(Sigma,eps[j,:])
        E+= np.log(normal_pdf(beta,mu,Sigma)/normal_pdf(beta,muPrior,sigmaPrior))
    E = np.mean(E)
    return E

def normal_pdf(theta,mu,Sigma):
    d = len(mu)
    #return np.exp(-(theta-mu)**2/(2*sigma**2))/np.sqrt(2*sigma**2*np.pi)
    return (2*np.pi)**(-d/2)*np.linalg.det(Sigma)**(-1/2)*np.exp(-np.dot(np.transpose(theta-mu),np.dot(np.linalg.inv(Sigma), theta-mu))/2)

def log_likelihood(beta, y,X,u):
    ll = 0
    #generate N*n particles
    alpha = np.zeros(len(u))
    #iterate over participants
    for i in range(y.shape[0]):
        #iterate over particles
        for j in range(len(alpha)):
            ll+=log_likelihood_particle(beta,y[i,:],X[i,:,:],alpha[j])
    return ll


def log_likelihood_particle(beta, y,X,alpha):
    #this is wrong.  Is it???
    log_likelihood = 0
    for i in range(len(y)):
        #iterate over timesteps
        log_likelihood += np.log(likelihood_i(beta,y[i],X[i],alpha))
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
    params = np.random.normal(0,1,(d+1,d+1))
    cov = np.identity(d)
    params[0:d,1:d+1] = cov
    print params
    #generate_data(beta,tau,n,num_times)
    X,y = generate_data(np.array([0.5,0.8,1]),1,20,4)#537
    #test likelihood for several beta values, beta = 2 should give high likelihood
    m = np.zeros((d+1,d+1))
    v = np.zeros((d+1,d+1))
    for i in range(10):
        params,m,v =iterate(params,y,X,i,m,v,30)
        mu = params[0:d,0]
        print mu
#    eps = np.random.rand(50)
#    print lower_bound(params,y,X,eps)
