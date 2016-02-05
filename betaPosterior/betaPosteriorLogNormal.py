import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy.stats import beta
from scipy import misc
import math
from matplotlib import pyplot as plt
import seaborn as sns

def iterate(params,n,k,i,m,v):
    N = 1
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    U1=np.random.normal(0,1,1)
    U2=np.random.normal(0,1,1)
    sn=np.random.normal(0,1,N)
    grad_lower_bound = grad(lower_bound)
    g = grad_lower_bound(params,n,k,U1,U2,sn)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    a = 0.001
    #params = params+a*g
    #params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v

def lower_bound(params,n,k,U1,U2,v):
    E = expectation(params,n,k,U1,v)
    KL = KL_via_sampling(params,0,1,U2)
    return E-KL

#Correct (probably)
def expectation(params,n,k,U1,v):
    theta = generate_lognormal(params,U1)
    e = np.exp(params[2])
    E=abc_log_likelihood(n,k,theta,i,v,e)
    return E

def likelihood(n,k,theta,i):
    return binomial_pmf(n,k,theta)

def abc_log_likelihood(n,k,theta,i,v,e):
    N = len(v)
    x = simulator(n,theta,v)
    log_kernels = log_abc_kernel(x,n,k,e)
    ll = log_kernels
    return ll

def simulator(n,theta,v):
    a = n
    c = 0
    b = 1
    d = 1
    p=theta
    mu = n*p
    sig2 = np.sqrt(n*p*(1-p))
    print n*p*(1-p)
    print "gaussian"
    print mu+sig2*v
    gaussian = mu+sig2*v
    #if gaussian<0:
    #    gaussian=0
    #    print "a 0"
    gaussian = np.clip(gaussian,0,n)
    return gaussian

def log_abc_kernel(x,n,k,e):
    '''
        @summary: kernel density, we use normal here
        @param y: observed data
        @param x: simulator output, often the mean of kernel density
        @param e: bandwith of density
        '''
    Sx = x
    Sy = k
    return -np.log(e)-np.log(2*np.pi)/2-(Sy-Sx)**2/(2*(e**2))

def binomial_pmf(n,k,theta):
    return nCr(n,k)*(theta**k)*((1-theta)**(n-k))

def nCr(n, k):
    """
        A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
        """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

#Correct
def generate_lognormal(params,u):
    mu=params[0]
    sigma=params[1]
    return np.exp(mu+sigma*u)

#Correct
def lognormal_pdf(theta,params):
    mu=params[0]
    sigma=params[1]
    return (1/(theta*sigma*np.sqrt(2*np.pi)))*np.exp(-((np.log(theta)-mu)**2)/(2*sigma**2))

#Correct
def KL_via_sampling(params,mu2,sigma2,U):
    mu = params[0]
    sigma = params[1]
    theta = generate_lognormal(params,U)
    E = np.log(lognormal_pdf(theta,params)/lognormal_pdf(theta,np.array([mu2,sigma2])))
    E = np.mean(E)
    return E

if __name__=='__main__':
    n = 100
    k = 70
    params = np.random.uniform(0,1,2)
    params = np.append(params,1.)
    m = np.array([0.,0.,0.])
    v = np.array([0.,0.,0.])
    for i in range(1):
        params,m,v = iterate(params,n,k,i,m,v)
        if i%1==0:
            print params
    print "true mean"
    print (k+1.)/(n+2.)
    U = np.random.normal(0,1,100000)
    samples = generate_lognormal(params,U)
    print "estimated mean"
    print np.mean(samples)
    a = k+1
    b = n-k+1
    x = np.linspace(0,1,100)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, beta.pdf(x, a,b),'r-', lw=5, alpha=0.6, label='beta pdf',color='blue')
    ax.plot(x,lognormal_pdf(x,params),'r-', lw=5, alpha=0.6, label='lognormal pdf',color='green')
    plt.show()
