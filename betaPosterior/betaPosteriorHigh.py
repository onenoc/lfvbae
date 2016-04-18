import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy.stats import beta
from scipy import misc
import math
from matplotlib import pyplot as plt
#import seaborn as sns

def iterate(params,n,k,i,m,v):
    S = 1
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    g = grad_lower_bound(params,n,k)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    a = 0.25
    #params = params+a*g
    params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v

def grad_lower_bound(params,n,k):
    grad_expectation=params(n,k,L)

def grad_expectation(params,n,k,v,L):
    '''
    @L: number of samples
    '''
    E = 0
    for i in range(L):
        #sample theta from kuma
        theta = generate_kumaraswamy(params)
        #take derivative of at that theta value
        grad_kuma_pdf = grad(kumaraswamy_pdf)
        #take log ABC-likelihood at that theta value
        f = abc_log_likelihood(n,k,theta,e)
        E+= f*grad_kuma_pdf(theta,params)
    
#Correct
def generate_kumaraswamy(params):
    u = np.random.normal(0,1)
    a=params[0]
    b=params[1]
    return (1-(1-u)**(1./b))**(1./a)

def kumaraswamy_pdf(theta,params):
    a=params[0]
    b=params[1]
    return a*b*(theta**(a-1))*((1-theta**a)**(b-1))

def abc_log_likelihood(n,k,theta,i,v,e):
    N = len(v)
    x = simulator(n,theta,v)
    log_kernels = log_abc_kernel(x,n,k,e)
    #print log_kernels
    #ll = misc.logsumexp(log_kernels)
    #ll = np.log(np.sum(np.exp(log_kernels)))
    #ll = np.log(1./N)+ll
    #print ll
    ll = log_kernels
    return ll

def simulator(n,theta):
    a = n
    c = 0
    b = 1
    d = 1
    p=theta
    mu = n*p
    sig2 = np.sqrt(n*p*(1-p))
    gaussian = np.random.normal(mu,sig2)
    #if gaussian<0:
    #    gaussian=0
    #    print "a 0"
    gaussian = np.clip(gaussian,0,n)
    return gaussian


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
def KL_via_sampling(params,a2,b2,U):
    a1 = params[0]
    b1 = params[1]
    theta = generate_kumaraswamy(params,U)
    E = np.log(kumaraswamy_pdf(theta,params)/kumaraswamy_pdf(theta,np.array([a2,b2])))
    E = np.mean(E)
    return E


if __name__=='__main__':
    n = 100
    k = 80
    params = np.random.uniform(10,100,2)
    params = np.append(params,1.)
    m = np.array([0.,0.,0.])
    v = np.array([0.,0.,0.])
    for i in range(5000):
        params,m,v = iterate(params,n,k,i,m,v)
        if i%100==0:
            print params
            #print m,v
            #U1=np.random.uniform(0,1,100)
            #U2=np.random.uniform(0,1,100)
            #U3=np.random.uniform(0,1,n)
            #print lower_bound(params,n,k,U1,U2,U3)
    print params
    plt.clf()
    print "true mean"
    print (k+1.)/(n+2.)
    U = np.random.uniform(0,1,100000)
    samples = generate_kumaraswamy(params,U)
    print "estimated mean"
    print np.mean(samples)
    a = k+1
    b = n-k+1
    x = np.linspace(0,1,100)
    #fig, ax = plt.subplots(1, 1)
    plt.plot(x,beta.pdf(x, a,b),'--',color='red',label='true')
    plt.plot(x,kumaraswamy_pdf(x,params),'-',color='blue',label='VI true likelihood')
    #plt.plot(x, beta.pdf(x, a,b),'r-', lw=5, label='beta pdf',color='blue')
    #plt.plot(x,kumaraswamy_pdf(x,params),'r-', lw=5, label='kuma pdf',color='green')
    plt.legend()
    plt.show()
