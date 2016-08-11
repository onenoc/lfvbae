import autograd.numpy as np
from autograd import grad
from scipy import special
from scipy.stats import beta, gamma
from scipy import misc
import math
from matplotlib import pyplot as plt
import pdb
from vbil import BBVI, sample_theta
from vbil import lower_bound as lower_boundBBVI

all_gradients = []
M=15
Sy=1
e_method = 0
e_val = 0.5
def iterate(params,i,m,v,num_samples,num_particles):
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    U1=np.random.normal(0,1,num_samples)
    sn=np.random.uniform(0,1,num_particles*M)
    grad_lower_bound = grad(lower_bound)
    g = grad_lower_bound(params,U1,sn,i)
    samples = sample_theta(params,num_samples)
    LB = lower_boundBBVI(params,samples,num_particles)
    #lower_bound(params,U1,sn,i)
    all_gradients.append(g)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    a = 5*(num_samples**(1./2))*1e-2
    #a=0.001
    #a = 5*(num_samples)*1e-2
    params = params+a*m_h/(np.sqrt(v_h)+e)
    #params = params+a*g
    return params,m,v,LB

def gradient_lower_bound(params,num_samples,num_simulations):
    U1=np.random.normal(0,1,num_samples)
    sn=np.random.uniform(0,1,num_simulations*M)
    grad_lower_bound = grad(lower_bound)
    g = grad_lower_bound(params,U1,sn,i)
    return g

def lower_bound(params,U1,v,i):
    E = expectation(params,U1,v,i)
    KL = KL_via_sampling(params,U1)
    return E-KL

def expectation(params,U1,v,i):
    theta = generate_lognormal(params,U1)
    E=0
    for j in range(len(theta)):
        E+=abc_log_likelihood(theta[j],j,v,i)
    #E = np.log(likelihood(theta))
    #return  np.mean(E)
    return E/len(theta)

def likelihood(theta):
    log_like= M*np.log(theta)-theta*M*Sy
    like = np.exp(log_like)
    return like

def abc_log_likelihood(theta,j,v,i):
    N = len(v)
    x, std = simulator(theta,v)
    log_kernels = log_abc_kernel(x,i,std)
    if len(log_kernels)>1:
        log_kernels_max = log_kernels.max()
        ll = np.log(np.sum(np.exp(log_kernels-log_kernels_max)))+log_kernels_max
        ll = np.log(1./N)+ll
    else:
        ll = log_kernels
    return ll

def simulator(theta,v):
    #handle more than one simulation per sample
    v = v.reshape(-1,M)
    simulations =-np.log(v)/theta
    return np.mean(simulations,1), np.std(simulations,1)

def log_abc_kernel(x,i,std):
    '''
    @summary: kernel density, we use normal here
    @param y: observed data
    @param x: simulator output, often the mean of kernel density
    @param e: bandwith of density
    '''
    if e_method==0:
        e = std/np.sqrt(M)
    elif e_method==1:
        e = max(30./i,0.01)
    else:
        e = e_val
    Sx = x
    return -np.log(e)-np.log(2*np.pi)/2-(Sy-Sx)**2/(2*(e**2))

#Correct
def generate_lognormal(params,u):
    mu = params[0]
    sigma = params[1]
    Y = mu+sigma*u
    X = np.exp(Y)
    return X

#Correct
def lognormal_pdf(theta,params):
    mu=params[0]
    sigma=params[1]
    x=theta
    return np.exp(-(np.log(x)-mu)**2/(2*(sigma**2)))/(x*sigma*np.sqrt(2*np.pi))

#Correct
def KL_via_sampling(params,U):
    theta = generate_lognormal(params,U)
    alpha = 2
    beta = 0.5
    muPrior = np.log(alpha/beta)
    sigmaPrior = np.log(np.sqrt(alpha/(beta**2)))
    paramsPrior = np.array([muPrior,sigmaPrior])
    E = np.log(lognormal_pdf(theta,params)/lognormal_pdf(theta,paramsPrior))
    E = np.mean(E)
    return E

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_gradients(params,num_samples,num_particles):
    for i in range(100):
        all_gradients.append(gradient_lower_bound(params,10,10))
    plt.hist(all_gradients)
    plt.show()

def avabc(params,num_samples,num_particles,K,convergence):
    lower_bounds = []
    scaled_lower_bounds = []
    iterating = 1
    i=0
    m = np.array([0.,0.])
    v = np.array([0.,0.])
    while iterating==1:
        params,m,v,LB = iterate(params,i,m,v,num_samples,num_particles)
        #LB/=M
        if params[1]<=0 or np.isnan(params).any():
            params = np.random.uniform(0,1,2)
            m = np.array([0.,0.])
            v = np.array([0.,0.])
        i+=1
        lower_bounds.append(-LB)
        if len(lower_bounds)>K+1:
            lb2 = np.mean(np.array(lower_bounds[-K:]))
            lb1 = np.mean(np.array(lower_bounds[-K-1:-1]))
            scaled_lower_bounds.append(-LB)
            if abs(lb2-lb1)<convergence:
                iterating = 0
            if i%10==0:
                print abs(lb2-lb1)
            if np.isnan(abs(lb2-lb1)):
                lower_bounds=[]
        if i%10==0:
            print params, LB
    return params, scaled_lower_bounds,i

if __name__=='__main__':
    paramsStart = np.random.uniform(2,3,2)
    Kavabc = 50
    Kvbil = 50
    same = 10
    num_samplesAVABC = same
    num_particlesAVABC = same
    num_samplesVBIL = same
    num_particlesVBIL = same
    #plot_gradients(params,num_samples,num_particles)
    convergenceAVABC = 1e-5
    convergenceVBIL = 1e-5
    x = np.linspace(0.001,2,100)
    params, lower_bounds, i = avabc(paramsStart,num_samplesAVABC,num_particlesAVABC, Kavabc,convergenceAVABC)
    paramsBBVI, lower_boundsBBVI, iBBVI = BBVI(paramsStart,num_samplesVBIL,num_particlesVBIL,Kvbil,convergenceVBIL)
    plt.plot(lower_boundsBBVI,label='BBVI S=%i, sim=%i' % (num_samplesVBIL, num_particlesVBIL),color='red')
    plt.plot(lower_bounds,label='AVABC S=%i, sim=%i' % (num_samplesAVABC, num_particlesAVABC),color='blue')
    plt.legend(loc=4)
    plt.ylim((-400,0))
    plt.title('Exponential Problem Scaled Lower Bounds, $\lambda=1$')
    plt.show()
    print 'AVABC convergence after %i iterations' % (i)
    print 'BBVI convergence after %i iterations' % (iBBVI)

    plt.plot(x,gamma.pdf(x,M+1,scale=1./(Sy*M+1)),label='true posterior',color='green')
    plt.plot(x,lognormal_pdf(x,paramsBBVI),label='BBVI',color='red')
    plt.plot(x,lognormal_pdf(x,params),label='AVABC',color='blue')
    plt.legend()
    plt.title('AVABC vs BBVI vs true posterior, %i samples, %i particles' % (num_samplesAVABC,num_particlesAVABC))
    plt.show()
    #plt.plot(all_gradients)
    #plt.show()
    #all_gradients = np.asarray(all_gradients)
    #running_var = []
    #for i in range(1,len(all_gradients)):
    #    running_var.append(np.var(all_gradients[0:i])/i)
    #plt.plot(running_var)
    #print len(moving_average(lower_bounds,n=100))
    #plt.hist(all_gradients)
    #plt.plot(moving_average(lower_bounds,n=100))
    #plt.plot(lower_bounds)
    #plt.show()

#    S = 10
#    b_1 = 0.9
#    b_2 = 0.999
#    e = 10e-8
#    grad_lower_bound = grad(lower_bound)
#    all_gradients = []
#    #params = np.array([21.,81.,1.])
#    params = np.array([10.,10.])
#    params = np.append(params,1.)
#    for i in range(100):
#        U1=np.random.uniform(0,1,1)
#        U2=np.random.uniform(0,1,S)
#        sn=np.random.normal(0,1,1)
#        g = grad_lower_bound(params,n,k,U1,U2,sn)
#        all_gradients.append(g[0])
#    #plt.hist(all_gradients)
#    sns.distplot(all_gradients)#,hist=False)
#    plt.title('AVABC gradients with 10 samples, 10 particles, alpha-beta fixed after 5k iterations')
#    plt.show()
#        

    '''
    U = np.random.uniform(0,1,100000)
    beta_samples = np.random.beta(k+1,n-k+1,100000)
    kuma_samples = generate_kumaraswamy(params[0:2],U)
    sns.distplot(beta_samples)
    sns.distplot(kuma_samples)
    plt.show()
    '''

