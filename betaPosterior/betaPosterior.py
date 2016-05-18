import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy.stats import beta
from scipy import misc
import math
from matplotlib import pyplot as plt
from vbil import lower_bound as lower_boundVBIL

all_gradients = []
n=100
k=20
def iterate(params,num_samples,num_particles,i,m,v):
    S = 1
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    U1=np.random.uniform(0,1,num_samples)
    sn=np.random.normal(0,1,num_particles)
    grad_lower_bound = grad(lower_bound)
    g = grad_lower_bound(params,U1,sn)
    samples = generate_kumaraswamy(params,U1)
    LB = lower_boundVBIL(params,samples,num_particles)
    all_gradients.append(g)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    a = 0.25
    params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v

def lower_bound(params,U1,sn):
    E = expectation(params,U1,sn)
    KL = KL_via_sampling(params,1,1,U1)
    return E-KL

#Correct (probably)
def expectation(params,U1,sn):
    theta = generate_kumaraswamy(params,U1)
    E=0
    for i in range(len(theta)):
        E+=abc_log_likelihood(theta[i],i,sn)
    return E/len(theta)

def abc_log_likelihood(theta,i,sn):
    N = len(sn)
    x = simulator(theta,sn)
    log_kernels = log_abc_kernel(x)
    if len(log_kernels)>1:
        log_kernels_max = log_kernels.max()
        ll = np.log(np.sum(np.exp(log_kernels-log_kernels_max)))+log_kernels_max
        ll = np.log(1./N)+ll
    else:
        ll = log_kernels
    return ll
    
def simulator(theta,v):
    a = n
    c = 0
    b = 1
    d = 1
    p=theta
    mu = n*p
    sig2 = np.sqrt(n*p*(1-p))
    gaussian = mu+sig2*v
    gaussian = np.clip(gaussian,0,n)
    return gaussian

def log_abc_kernel(x):
    '''
    @summary: kernel density, we use normal here
    @param y: observed data
    @param x: simulator output, often the mean of kernel density
    @param e: bandwith of density
    '''
    e = 0.1
    Sx = x
    Sy = k
    return -np.log(e)-np.log(2*np.pi)/2-(Sy-Sx)**2/(2*(e**2)) 

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

def AVABC(params, num_samples,num_particles):
    m = np.array([0.,0])
    v = np.array([0.,0.])
    for i in range(1000):
        params,m,v = iterate(params,num_samples,num_particles,i,m,v)
        if i%100==0:
            print params
    return params

if __name__=='__main__':
    params = np.random.uniform(10,100,2)

    lower_bounds = []
    num_samples = 10
    num_particles = 10
    
    params = AVABC(params,num_samples,num_particles)
    print params
    print "true mean"
    print (k+1.)/(n+2.)
    U = np.random.uniform(0,1,100000)
    samples = generate_kumaraswamy(params,U)
    print "estimated mean"
    print np.mean(samples)
    a = k+1
    b = n-k+1
    x = np.linspace(0,1,100)
#    fig, ax = plt.subplots(1, 1)
#    plt.plot(x,beta.pdf(x, a,b),'--',color='red',label='true')
#    plt.plot(x,kumaraswamy_pdf(x,params),'-',color='blue',label='VI true likelihood')
    plt.plot(x, beta.pdf(x, a,b),label='true posterior',color='blue')
    plt.plot(x,kumaraswamy_pdf(x,params),label='VABC',color='green')
    plt.legend()
    plt.show()
