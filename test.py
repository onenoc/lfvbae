import lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import math
'''
Run variational inference 
'''

def generate_data(m,n,weight_vector,bias,sigma_e):
    np.random.seed(50)
    X = np.random.uniform(0, 1,(m, n))
    np.random.seed(50)
    e = np.random.normal(0, sigma_e,(m,1))
    if bias:
        X = np.column_stack((X,np.ones(m)))
    dot = np.reshape(np.dot(X,weight_vector), (m,1))
    Y = dot+e
    return Y,X

def true_posterior_standard_normal(n, bias, sigma_e,X,y):
    beta = 1/(sigma_e**2)
    Sinv = np.identity(n+bias)+beta*np.dot(X.T,X)
    S = np.linalg.inv(Sinv)
    muTrue = beta*np.dot(S,np.dot(X.T,y))
    return muTrue,S

def plot_cost(encoder, start_percent=0):
    iterations = len(encoder.lowerBounds)
    plt.plot(encoder.lowerBounds[int(start_percent*iterations):])
    plt.show()

def run_VA(n, bias, m, sigma_e, iterations, batch, Lu=1):
    #dimX, dimTheta, m, n
    encoder = create_encoder(n, bias, m, sigma_e, iterations, batch, Lu)
    for i in range(iterations):
        encoder.iterate(batch)
    return encoder

def run_VA_five_times(n, bias, m, sigma_e, iterations, batch, Lu=1):
    mu_list = []
    sigma_list = []
    for i in range(5):
        encoder = run_VA(n, bias, m, sigma_e, iterations, batch, Lu)
        mu_list.append(encoder.params[0].get_value())
        sigma_list.append(np.exp(encoder.params[1].get_value()))
    return np.median(mu_list), np.median(sigma_list), encoder

def plot(muVar, sigmaVar, muSDTrue, sigmaSDTrue):
    xMin = min(muVar,muSDTrue)-2*max(sigmaVar,np.sqrt(varSDTrue))
    xMax = max(muVar,muSDTrue)+2*max(sigmaVar,np.sqrt(varSDTrue))
    x = np.linspace(xMin,xMax, 1000)
    plt.title('m= %f, sigma_e=%f, weight=2, no bias, %i iterations' % (m, sigma_e, iterations)) 
    plt.plot(x,mlab.normpdf(x,muVar,sigmaVar))
    plt.plot(x,mlab.normpdf(x,muSDTrue,sigmaSDTrue))
    plt.legend(('variational, sd=%f' % (sigmaVar), 'bayesian regression sd=%f' % (sigmaSDTrue)))
    plt.show()
    
def create_encoder(n, bias, m, sigma_e, iterations, batch, Lu=1):
    encoder = lfvbae.VA(n+bias, n+bias, m, 1, sigma_e, Lu)
    encoder.initParams()
    encoder.createObjectiveFunction()
    return encoder

def plot_cost_function(encoder, batch, muSDTrue, sigmaSDTrue):
    x = np.linspace(0,sigmaSDTrue+100*sigmaSDTrue, 20)
    costs = []
    for val in x:
        costs.append(encoder.changeParamsAndCalcCost(batch,muSDTrue, np.array([[val]])))
        print costs[-1]
    cost_true_posterior = encoder.changeParamsAndCalcCost(batch,muSDTrue,sigmaSDTrue)

    plt.plot(x, costs)
    plt.show()

if __name__=='__main__':
    m = 200
    n=1
    bias=0
    sigma_e=0.5
    Lu=1
   
    iterations = 1000
    y,X = generate_data(m,n,np.array([2]),bias, sigma_e)
    np.random.seed()
    muSDTrue, varSDTrue = true_posterior_standard_normal(n, bias, sigma_e,X,y)
    muSDTrue = muSDTrue[0][0]
    varSDTrue = varSDTrue[0][0]
    sigmaSDTrue = np.sqrt(varSDTrue)
    
    batch = np.column_stack((y,X))
 
    muVar, sigmaVar, encoder = run_VA_five_times(n, bias, m, sigma_e, iterations, batch,Lu=Lu)

    #plot_cost(encoder)
    print "final cost"
    print encoder.lowerBounds[-1]

    print "variational posterior"
    print encoder.params[0].get_value(), np.exp(encoder.params[1].get_value())

    print "true posterior"
    print muSDTrue, np.sqrt(varSDTrue)
    
    print "MLE sigma"
    print np.sqrt((sigma_e**2)*np.linalg.inv(np.dot(X.T,X)))
   
    plot(muVar, sigmaVar, muSDTrue, sigmaSDTrue)
    #plot_cost_function(encoder, batch, muSDTrue, sigmaSDTrue) 
