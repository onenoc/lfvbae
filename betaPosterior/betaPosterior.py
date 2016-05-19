import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy.stats import beta
from scipy import misc
import math
from matplotlib import pyplot as plt
from vbil import lower_bound as lower_boundVBIL
from vbil import BBVI

all_gradients = []
n=100
k=70
i_num = 1
def iterate(params,num_samples,num_particles,i,m,v):
    S = 1
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    i_num = i
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
    return params,m,v,LB

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
    e = 0.001
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

def AVABC(params, num_samples,num_particles,K,convergence):
    m = np.array([0.,0])
    v = np.array([0.,0.])
    iterating = 1
    lower_bounds = []
    i=0
    while iterating==1:
        params,m,v,LB = iterate(params,num_samples,num_particles,i,m,v)
        LB=LB/10
        if i%10==0:
            print params
        i+=1
        lower_bounds.append(-LB)
        if len(lower_bounds)>K+1:
            lb2 = np.mean(np.array(lower_bounds[-K:]))
            lb1 = np.mean(np.array(lower_bounds[-K-1:-1]))
            lower_bounds.append(-LB)
            if abs(lb2-lb1)<convergence:
                iterating = 0
            if i%10==0:
                print abs(lb2-lb1)
    return params, lower_bounds, i

if __name__=='__main__':
    params = np.random.uniform(0,100,2)
    lower_bounds = []
    num_samples = 10
    num_particles = 10
    K=50
    convergence=1e-3
    paramsAVABC,lower_boundsAVABC,i = AVABC(params,num_samples,num_particles,K,convergence)
    paramsBBVI,lower_boundsBBVI,iBBVI = BBVI(params,num_samples,num_particles,K,convergence)
    print params
    print "true mean"
    print (k+1.)/(n+2.)
    a = k+1
    b = n-k+1
    x = np.linspace(0,1,100)
    plt.plot(lower_boundsBBVI,label='BBVI S=%i, sim=%i' % (num_samples,num_particles),color='red')
    plt.plot(lower_boundsAVABC,label='AVABC  S=%i, sim=%i' % (num_samples,num_particles),color='blue')
    plt.title('Beta-Bernoulli Lower Bound')
    plt.legend(loc=4)
    plt.show()
#    fig, ax = plt.subplots(1, 1)
#    plt.plot(x,beta.pdf(x, a,b),'--',color='red',label='true')
#    plt.plot(x,kumaraswamy_pdf(x,params),'-',color='blue',label='VI true likelihood')
    plt.plot(x, beta.pdf(x, a,b),label='true posterior',color='green')
    plt.plot(x,kumaraswamy_pdf(x,paramsAVABC),label='AVABC',color='blue')
    plt.plot(x,kumaraswamy_pdf(x,paramsBBVI),label='BBVI',color='red')
    plt.title('AVABC vs BBVI vs true posterior, %i samples, %i particles' % (num_samples,num_particles))
    plt.legend(loc=2)
    plt.show()
    print 'AVABC params'
    print paramsAVABC
    params = [ 14.42637141  52.71715884]

