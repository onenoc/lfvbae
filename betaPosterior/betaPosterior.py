import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy.stats import beta
from scipy import misc
import math
from matplotlib import pyplot as plt
import seaborn as sns

def iterate(params,n,k,i,m,v):
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    U1=np.random.uniform(0,1,1)
    U2=np.random.uniform(0,1,1)
    sn=np.random.normal(0,1)
    grad_lower_bound = grad(lower_bound)
    g = grad_lower_bound(params,n,k,U1,U2,sn)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    #a=1./(10+i)
    a = 0.1
    #params = params+a*g
    params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v

def lower_bound(params,n,k,U1,U2,v):
    E = expectation(params,n,k,U1,v)
    KL = KL_via_sampling(params,1,1,U2)
    return E-KL

#Correct (probably)
def expectation(params,n,k,U1,v):
    theta = generate_kumaraswamy(params,U1)
    e = np.exp(params[2])
    #E = np.log(binomial_pmf(n,k,theta))
    #for i in range(len(theta)):
    #E = np.log(likelihood(n,k,theta,0))
    E=abc_log_likelihood(n,k,theta,i,v,e)
    #E = np.mean(E)
    return E

def likelihood(n,k,theta,i):
    return binomial_pmf(n,k,theta)

def abc_log_likelihood(n,k,theta,i,v,e):
    N = 1
    x = simulator(n,theta,v)
    log_kernels = log_abc_kernel(x,n,k,e)
    #ll = misc.logsumexp(log_kernels)
    #ll = np.log(1./N)+ll
    ll = log_kernels 
    return ll
    
def simulator(n,theta,v):
    p=theta
    mu = n*p
    sig2 = np.sqrt(n*p*(1-p))
    gaussian = mu+sig2*v
    if gaussian<0:
        gaussian=0
    return gaussian

def log_abc_kernel(x,n,k,e):
    '''
    @summary: kernel density, we use normal here
    @param y: observed data
    @param x: simulator output, often the mean of kernel density
    @param e: bandwith of density
    '''
    #e = 0.1
    #e=np.std(x)/np.sqrt(500)
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
def generate_kumaraswamy(params,u):
    a=params[0]
    b=params[1]
    return (1-(1-u)**(1./b))**(1./a)

#Correct
def kumaraswamy_pdf(theta,params):
    a=params[0]
    b=params[1]
    return a*b*(theta**(a-1))*((1-theta**a)**(b-1))

#Correct
def KL_via_sampling(params,a2,b2,U):
    a1 = params[0]
    b1 = params[1]
    theta = generate_kumaraswamy(params,U)
    E = np.log(kumaraswamy_pdf(theta,params)/kumaraswamy_pdf(theta,np.array([a2,b2])))
    E = np.mean(E)
    return E

if __name__=='__main__':
    n = 10
    k = 5
    params = np.random.uniform(10,100,2)
    params = np.append(params,0)
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
    plt.clf()
    #U = np.random.uniform(0,1,100000)
    #beta_samples = np.random.beta(k+1,n-k+1,100000)
    #kuma_samples = generate_kumaraswamy(params,U)
    #sns.distplot(beta_samples)
    #sns.distplot(kuma_samples)
    print "true mean"
    print (k+1.)/(n+2.)
    #print "estimated mean"
    #print np.mean(kuma_samples)
    a = k+1
    b = n-k+1
    x = np.linspace(0,1,100)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, beta.pdf(x, a,b),'r-', lw=5, alpha=0.6, label='beta pdf',color='blue')
    ax.plot(x,kumaraswamy_pdf(x,params),'r-', lw=5, alpha=0.6, label='kuma pdf',color='green')
    plt.show()
