import autograd.numpy as np
from autograd import elementwise_grad, grad
from scipy import special
from scipy import stats
from scipy.stats import beta
from scipy import misc
from matplotlib import pyplot as plt
import math

all_gradients = []
lower_bounds = []
M=10
n=100
k=70
Sy=k
def iterate(params,num_samples,num_particles,i,m,v):
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    samples = sample_theta(params,num_samples)
    LB = lower_bound(params,samples,num_particles)
    g = -grad_KL(params, num_samples,num_particles)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    a = 0.25
    params = params+a*m_h/(np.sqrt(v_h)+e)
    #params = params+a*g
    return params,m,v, LB

def grad_KL(params, num_samples, num_particles):
    S = num_samples
    samples = sample_theta(params,S)
    #initialize KL to be this
    KL1 = gradient_log_variational(params,samples,0)
    KL1 *= log_variational(params,samples)-h_s(samples,num_particles)-c_i(params,0,S,num_particles)
    KL1 = np.sum(KL1)/S
    KL2 = gradient_log_variational(params,samples,1)
    KL2 *= log_variational(params,samples)-h_s(samples,num_particles)-c_i(params,1,S,num_particles)
    KL2 = np.sum(KL2)/S
    KL = np.array([KL1,KL2])
    return KL

def lower_bound(params,samples,num_particles):
    S = len(samples)
    #without the sum, which is a highly nonstandard convergence criteria, this performs VERY well
    return np.sum(log_variational(params,samples)-h_s(samples, num_particles))/S

def log_variational(params, theta):
    '''
    @summary: log-pdf of variational distribution
    '''
    a=params[0]
    b=params[1]
    return np.log(a*b*(theta**(a-1))*((1-theta**a)**(b-1)))

def kumaraswamy_pdf(theta,params):
    a=params[0]
    b=params[1]
    return a*b*(theta**(a-1))*((1-theta**a)**(b-1))

def gradient_log_variational(params,theta, i):
    a=params[0]
    b=params[1]
    x=theta
    if i==0:
        return -b*(x**(a-1))*((1-x**a)**(b-2))*(a*np.log(x)*(b*(x**a)-1)+x**a-1)
    else:
        return a*(x**(a-1))*((1-x**a)**(b-1))*(b*np.log(1-x**a)+1)

def gradient_check():
    params = np.array([2,2])
    h = np.array([1e-5,0])
    print (np.exp(log_variational(params+h,0.5))-np.exp(log_variational(params,0.5)))/h[0]
    h = np.array([0,1e-5])
    print (np.exp(log_variational(params+h,0.5))-np.exp(log_variational(params,0.5)))/h[1]
    print gradient_log_variational(params,0.5,0)
    print gradient_log_variational(params,0.5,1)

def h_s(theta,num_particles):
    h_s = np.log(prior_density(theta))+abc_log_likelihood(theta,num_particles)
    return h_s

def abc_log_likelihood(samples,num_particles):
    N=num_particles
    S = len(samples)
    log_kernels = np.zeros(N)
    ll = np.zeros(S)
    for s in range(S):
        theta = samples[s]
        x = simulator(theta,N)
        log_kernels = log_abc_kernel(x)
        ll[s] = misc.logsumexp(log_kernels)
        ll[s] = np.log(1./N)+ll[s]
    return ll

def simulator(theta,N):
    return np.random.binomial(n,theta,size=N)

def log_abc_kernel(x):
    '''
        @summary: kernel density, we use normal here
        @param y: observed data
        @param x: simulator output, often the mean of kernel density
        @param e: bandwith of density
        '''
    #e=std/np.sqrt(M)
    e = 0.001
    Sx = x
    return -np.log(e)-np.log(2*np.pi)/2-(Sy-Sx)**2/(2*(e**2))

def c_i(params,i,S,num_particles):
    return 0
    if S==1:
        return 0
    first = np.zeros(S)
    second = np.zeros(S)
    samples = sample_theta(params,S)
    first = h_s(samples,num_particles)*gradient_log_variational(params,samples,i)
    second = gradient_log_variational(params,samples,i)
    return np.cov(first,second)[0][1]/np.cov(first,second)[1][1]

def sample_theta(params,S):
    '''
        @param params: list of parameters for recognition model, gamma
        @param S: number of samples
        '''
    u = np.random.uniform(0,1,S)
    return generate_kumaraswamy(params,u)

def prior_density(theta):
    params = np.array([1,1])
    return 1
#np.random.beta(params[0],params[1],size=S)

def generate_kumaraswamy(params,u):
    a=params[0]
    b=params[1]
    return (1-(1-u)**(1./b))**(1./a)

def BBVI(params,num_samples,num_particles,K,convergence):
    m = np.array([0.,0.])
    v = np.array([0.,0.])
    lower_bounds = []
    iterating = 1
    i=0
    while iterating==1:
        params,m,v,LB = iterate(params,num_samples,num_particles,i,m,v)
        i+=1
        lower_bounds.append(-LB/M)
        if params[0]<=0 or params[1]<=0:
            i=0
            params = np.random.uniform(10,100,2)
            m = np.array([0.,0.])
            v = np.array([0.,0.])
        if i%100==0:
            print params
        if len(lower_bounds)>K+1:
            lb2= np.mean(np.array(lower_bounds[-K:]))
            lb1 = np.mean(np.array(lower_bounds[-K-1:-1]))
            if abs(lb2-lb1)<convergence:
                iterating = 0
    return params, lower_bounds, i


if __name__=='__main__':
    print gradient_check()
    params = np.random.uniform(10,100,2)
    m = np.array([0.,0.])
    v = np.array([0.,0.])
    lower_bounds = []
    num_samples = 10
    num_particles = 10
    K=10
    convergence = 1e-4
    lower_bounds = []
    iterating = 1
    params, lower_bounds, i = BBVI(params,num_samples,num_particles,K,convergence)
    print '%i iterations' % (i)
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
    fig, ax = plt.subplots(1, 1)
    plt.plot(x, beta.pdf(x, a,b),'r-', lw=5, label='beta pdf',color='blue')
    plt.plot(x,kumaraswamy_pdf(x,params),'r-', lw=5, label='kuma pdf',color='green')
    plt.legend()
    plt.show()
    plt.plot(lower_bounds)
    plt.show()
#
#    num_samples = 1
#    num_particles = 1
#    params = np.array([3,100])
#    for i in range(100):
#        all_gradients.append(grad_KL(params, num_samples,num_particles,n,k)[1])
#    print all_gradients
#    plt.hist(all_gradients,color='orange')
#    plt.title('BBVI with 10 samples, 10 particles, alpha-beta after 2k runs')
#    plt.show()

