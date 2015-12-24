import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy import stats

#tested to make sure same posterior gives KL 0
def beta_KL(alpha1,beta1,alpha2,beta2):
    return np.log(special.beta(alpha2,beta2)/special.beta(alpha1,beta1)-(alpha2-alpha1)*special.polygamma(0,alpha1)-(beta2-beta1)*special.polygamma(0,beta1)+(alpha2-alpha1+beta2-beta1)*special.polygamma(0,alpha1+beta1))

def lower_bound(params,uniforms):
    alpha = params[0]
    beta = params[1]
    sorted_uniforms = np.sort(uniforms)
    theta = sorted_uniforms[alpha-1]
    KL = beta_KL(alpha,beta,1,1)

def expectation(a,b,n,Y,U):
    theta = generate_kumaraswamy(a,b,U)
    E = stats.binom.pmf(Y,n,theta)
    E = np.mean(E)
    return E

def iterate(params,i):
    a = 1/(500+i)
    alpha = params[0]
    beta = params[1]
    n = beta+1-alpha
    uniforms = np.random.uniform(0,1,n)
    grad_lower_bound = grad(lower_bound)
    params = params+grad_lower_bound(params,uniforms)*

def generate_random_beta(alpha,beta):
    n=beta+alpha-1
    uniforms = np.random.uniform(0,1,n)
    sorted_uniforms = np.sort(uniforms)
    return sorted_uniforms[alpha-1] 

def generate_kumaraswamy(a,b,u):
    return (1-(1-u)**(1./b))**(1./a)

def kumaraswamy_pdf(theta,a,b):
    return a*b*(theta**(a-1))*((1-theta**a)**(b-1))

def KL_via_sampling(a1,b1,a2,b2,U):
    theta = generate_kumaraswamy(a1,b1,U)
    E = np.log(kumaraswamy_pdf(theta,a1,b1)/kumaraswamy_pdf(theta,a2,b2))
    E = np.mean(E)
    return E

