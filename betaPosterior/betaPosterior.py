import autograd.numpy as np
from autograd import grad
from scipy import special
import math
from matplotlib import pyplot as plt
import seaborn as sns

def iterate(params,n,k,i):
    a=1./(5+i)
    U1=np.random.uniform(0,1,10)
    U2=np.random.uniform(0,1,10)
    grad_lower_bound = grad(lower_bound)
    grad_params = grad_lower_bound(params,n,k,U1,U2)
    params = params+a*grad_params
    return params

#true posterior is beta(21,11)

def test():
    U1=np.random.uniform(0,1,100)
    U2=np.random.uniform(0,1,100)
    n=30
    k=20
    params = np.array([1,1])
    grad_KL = grad(KL_via_sampling)
    grad_KL_params =0

def lower_bound(params,n,k,U1,U2):
    E = expectation(params,n,k,U1)
    KL = KL_via_sampling(params,1,1,U2)
    return E-KL

#Correct (probably)
def expectation(params,n,k,U):
    theta = generate_kumaraswamy(params,U)
    E = np.log(binomial_pmf(n,k,theta))
    for i in range(len(theta)):
        E = np.log(binomial_pmf(n,k,theta[i]))
    E = np.mean(E)
    return E

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
    n = 30
    k = 9
    params = np.random.uniform(1,50,2)
    for i in range(1000):
        params = iterate(params,n,k,i)
        if i%100==0:
            print params
            U1=np.random.uniform(0,1,1000)
            U2=np.random.uniform(0,1,1000)
            print lower_bound(params,n,k,U1,U2)
    plt.clf()
    U = np.random.uniform(0,1,100000)
    beta_samples = np.random.beta(k+1,n-k+1,100000)
    kuma_samples = generate_kumaraswamy(params,U)
    sns.distplot(beta_samples)
    sns.distplot(kuma_samples)
    plt.show()
