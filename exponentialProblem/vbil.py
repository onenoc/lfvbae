import autograd.numpy as np
from autograd import elementwise_grad, grad
from scipy import special
from scipy import stats
from scipy.stats import beta, gamma, gaussian_kde
from scipy import misc
from matplotlib import pyplot as plt
import math
import pickle

all_gradients = []
lower_bounds = []
M=15
Sy =1
iteration = 1
def iterate(params,num_samples,num_particles,i,m,v):
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    samples = sample_theta(params,num_samples)
    LB = lower_bound(params,samples,num_particles)
    g = -grad_KL(params, samples,num_particles,LB)
    m = b_1*m+(1-b_1)*g
    v = b_2*v+(1-b_2)*(g**2)
    m_h = m/(1-(b_1**(i+1)))
    v_h = v/(1-(b_2**(i+1)))
    S = num_samples
    a = (S**(1./4))*1e-2
    #a = 5*(S)*1e-2
    all_gradients.append(g)
    params = params+a*m_h/(np.sqrt(v_h)+e)
    return params,m,v,LB

def lower_bound(params,samples,num_particles):
    S = len(samples)
    #without the sum, which is a highly nonstandard convergence criteria, this performs VERY well
    return np.sum(log_variational(params,samples)-h_s(samples, num_particles))/S

def grad_KL(params, samples, num_particles,LB):
    S = len(samples)
    #initialize KL to be this
    KL1 = gradient_log_variational(params,samples,0)*(LB-c_i(params,0,S,num_particles)/S)
    KL1 = np.sum(KL1)
    KL2 = gradient_log_variational(params,samples,1)*(LB-c_i(params,1,S,num_particles)/S)
    KL2 = np.sum(KL2)
    KL = np.array([KL1,KL2])
    return KL

def log_variational(params, theta):
    '''
    @summary: log-pdf of variational distribution
        '''
    mu=params[0]
    sigma=params[1]
    x = theta
    return loglognormal_np(np.log(x),mu,sigma)

#correct
def loglognormal_np( logx, mu, stddev ):
    
    log_pdf = np.log(np.exp(-(logx-mu)**2/(2*(stddev**2)))/(np.exp(logx)*stddev*np.sqrt(2*np.pi)))
    #log_pdf = -np.log(stddev) - 0.5*pow( (logx-mu)/stddev, 2.0 )-logx-0.5*np.log(2*np.pi)
    return log_pdf

#correct
def gradient_log_variational(params,theta, i):
    mu=params[0]
    sigma=params[1]
    x= theta
    if i==0:
        return (np.log(x)-mu)/(sigma**2)
    else:
        return (mu-np.log(x))**2/(sigma**3)-1/sigma

#correct
def gradient_check():
    params = np.array([2,2])
    h = np.array([1e-5,0])
    print (log_variational(params+h,0.5)-log_variational(params,0.5))/h[0]
    h = np.array([0,1e-5])
    print (log_variational(params+h,0.5)-log_variational(params,0.5))/h[1]
    print gradient_log_variational(params,0.5,0)
    print gradient_log_variational(params,0.5,1)

def h_s(theta,num_particles):
    h_s = log_prior_density(theta)+abc_log_likelihood(theta,num_particles)
    return h_s

#seems correct
def abc_log_likelihood(samples,num_particles):
    N=num_particles
    S = len(samples)
    log_kernels = np.zeros(N)
    ll = np.zeros(S)
    for s in range(S):
        theta = samples[s]
        x,std = simulator(theta,N)
        log_kernels = log_abc_kernel(x,std)
        ll[s] = misc.logsumexp(log_kernels)
        ll[s] = np.log(1./N)+ll[s]
    return ll

#correct
def simulator(theta,N):
    #get 500*N exponentials
    exponentials = np.random.exponential(1/theta,size=N*M)
    #reshape to Nx500
    exponentials = np.reshape(exponentials,(N,M))
    #get means of the rows
    summaries = np.mean(exponentials,1)
    std = np.std(exponentials,1)
    return summaries, std

#gets max likelihood at right point
def log_abc_kernel(x,std):
    '''
        @summary: kernel density, we use normal here
        @param x: simulator output, often the mean of kernel density
        @param e: bandwith of density
        '''
    
    e=std/np.sqrt(M)
    #e = 1
    Sx = x
    return -np.log(e)-np.log(2*np.pi)/2-(Sy-Sx)**2/(2*(e**2))

def c_i(params,i,S,num_particles):
    if S==1:
        return 0
    first = np.zeros(S)
    second = np.zeros(S)
    samples = sample_theta(params,S)
    first = h_s(samples,num_particles)*gradient_log_variational(params,samples,i)
    second = gradient_log_variational(params,samples,i)
    return np.cov(first,second)[0][1]/np.cov(first,second)[1][1]

#Correct
def sample_theta(params,S):
    '''
        @param params: list of parameters for recognition model, gamma
        @param S: number of samples
        '''
    return generate_lognormal(params,S)

#Correct
def log_prior_density(theta):
    alpha = 2
    beta = 0.5
    mu = np.log(alpha/beta)
    sigma = np.log(np.sqrt(alpha/(beta**2)))
    params = np.array([mu,sigma])
    return log_variational(params, theta)

#Correct
def generate_lognormal(params,S):
    mu=params[0]
    sigma=params[1]
    Y = np.random.normal(mu,sigma,S)
    X = np.exp(Y)
    return X

def plot_gradients(params,num_samples,num_particles):
    for i in range(100):
        samples = sample_theta(params,num_samples)
        LB = lower_bound(params,samples,num_particles)
        g = -grad_KL(params, samples,num_particles,LB)
        all_gradients.append(g)
    plt.hist(all_gradients)
    plt.show()

def BBVI(params,num_samples,num_particles,K,convergence):
    m = np.array([0.,0.])
    v = np.array([0.,0.])
    iterating = 1
    lower_bounds = []
    i=0
    while iterating==1:
        params,m,v,LB = iterate(params,num_samples,num_particles,i,m,v)
        i+=1
        lower_bounds.append(-LB)
        if params[1]<=0:
            params = np.random.uniform(0,1,2)
            m = np.array([0.,0.])
            v = np.array([0.,0.])
        if i%10==0:
            print params, LB
        if len(lower_bounds)>K+1:
            lb2= np.mean(np.array(lower_bounds[-K:]))
            lb1 = np.mean(np.array(lower_bounds[-K-1:-1]))
            if abs(lb2-lb1)<convergence:
                iterating = 0
    return params,lower_bounds,i

if __name__=='__main__':
    params = np.random.uniform(0,1,2)
    #params = np.array([5,5])
    num_samples=100
    num_particles=100

    K=10
    params,lower_bounds,i = BBVI(params,num_samples,num_particles,K)
    #plot_gradients(params,num_samples,num_particles)
    print params,i
    print "true mean"
    print (M+1.)/(Sy*M+1)
    samples = generate_lognormal(params,10000)
    print "estimated mean"
    print np.mean(samples)
    mu = params[0]
    sigma = params[1]
    x = np.linspace(0.01,3,100)
    fig, ax = plt.subplots(1, 1)
    plt.plot(x,np.exp(log_variational(params,x)),label='BBVI/VBIL')
    plt.plot(x,stats.gamma.pdf(x,M+1,scale=1./(Sy*M+1)),label='true posterior')
    plt.title('Posteriors for $\lambda=1$, AVABC and BBVI')
    plt.legend()
    plt.show()
    plt.plot(lower_bounds)
    plt.show()
    plt.plot(all_gradients)
    plt.show()

    #params = np.array([0.223545, 0.289477])
    #params = np.array([5,5])
    #AVABC_gradients_1 = pickle.load(open('gradients-sigma-single-correct-far.pkl','rb'))
    #AVABC_gradients_10 = pickle.load(open('gradients-single-sigma-far.pkl','rb'))
    #all_gradients_10=[]
    #all_gradients_1=[]
    #num_samples = 1
    #num_particles = 1
    #for i in range(1000):
    #    all_gradients_10.append(-grad_KL(params,10,num_particles)[1])
    #    all_gradients_1.append(-grad_KL(params,1,num_particles)[1])
    #    #print grad_KL(params, num_samples,num_particles)[1]
    #AVABC_gradients_1 = np.asarray(AVABC_gradients_1)
    #AVABC_gradients_10 = np.asarray(AVABC_gradients_10)
    #all_gradients_10 = np.asarray(all_gradients_10)
    #all_gradients_1 = np.asarray(all_gradients_1)
    #density_AVABC_10 = gaussian_kde(AVABC_gradients_10)
    #density_AVABC_1 = gaussian_kde(AVABC_gradients_1)
    #density_BBVI_10 = gaussian_kde(all_gradients_10)
    #density_BBVI_1 = gaussian_kde(all_gradients_1)
    #print np.std(AVABC_gradients_1)
    #print np.std(AVABC_gradients_10)
    #print np.std(all_gradients_1)
    #print np.std(all_gradients_10)
    ##xs = np.linspace(np.mean(AVABC_gradients_10)-2*np.std(AVABC_gradients_10), np.mean(AVABC_gradients_10)+2*np.std(AVABC_gradients_10),200)
    ##plt.plot(xs,density_AVABC_10(xs),label='AVABC 10 samples')
    ##plt.plot(xs,density_AVABC_1(xs),label='AVABC 1 sample')
    ##plt.plot(xs,density_BBVI_10(xs),label='BBVI 10 samples')
    ##plt.plot(xs,density_BBVI_1(xs),label='BBVI 1 sample')
    #plt.hist(AVABC_gradients_1)
    #plt.hist(AVABC_gradients_10)
    #plt.title(r'$\nabla_{\sigma}\mathcal{L}$, AVABC and BBVI, convergence parameters,$\lambda$=1')
    #plt.legend()
    #plt.show()


