import lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

def generate_data(m,n,weight_vector,bias,sigma_e):
    X = np.random.uniform(0, 1,(m, n))
    e = np.random.normal(0, sigma_e,(m,1))
    if bias:
        X = np.column_stack((X,np.ones(m)))
    dot = np.reshape(np.dot(X,weight_vector), (m,1))
    Y = dot+e
    return Y,X

def get_true_posterior(muPrior, sigmaPrior,n,bias, sigma_e):
    alpha = 1./sigmaPrior
    Sinv = np.dot(alpha, sigma_e*np.identity(n+bias))+np.dot(X.T,X)
    S = np.linalg.inv(Sinv)
    muTrue = np.dot(S,np.dot(X.T,Y)) 
    return muTrue,np.sqrt(S)

m = 10
n=1
bias=0
sigma_e=1
#dimX, dimTheta, m, n
encoder = lfvbae.VA(n+bias, n+bias, m, 1, learning_rate=0.2)
encoder.initParams()
encoder.createGradientFunctions()

Y,X=generate_data(m,n,np.array([2]),0, sigma_e)

muPrior, sigmaPrior = encoder.params[0][0][0], np.exp(encoder.params[1][0][0])
muTrue,sigmaTrue = get_true_posterior(muPrior,sigmaPrior,n,bias,sigma_e)

X = np.column_stack((Y,X))
'''
posteriors = []
iterations = 3000
for i in range(iterations):
    if i%100==0:
        print "intercept mean, logSigma, logLambda"
        print encoder.params
    if i==iterations-1:
        muPosterior, sigmaPosterior = encoder.params[0][0][0], np.exp(encoder.params[1][0][0])
        posteriors.append((muPosterior, sigmaPosterior))
    encoder.iterateConjugate(X)
    #encoder.iterate(X)

print "variational inference posterior"
print posteriors[-1]

print "true posterior"
print muTrue
print sigmaTrue

print "times difference"
print posteriors[-1][1]/sigmaTrue
'''

'''
#muPosterior = posteriors[-1][0]
#sigmaPosterior = posteriors[-1][1]

#x=np.linspace(lstsqMu-2*lstsqSD, lstsqMu+2*lstsqSD)
#plt.plot(x,mlab.normpdf(x,lstsqMu,lstsqSD))
#plt.plot(x,mlab.normpdf(x,muPosterior,sigmaPosterior))

#x = np.linspace(minMu-2*maxSD, maxMu+2*maxSD, 1000)
#plt.plot(x,mlab.normpdf(x,muPrior,sigmaPrior))
#for i in range(6):
#    plt.plot(x,mlab.normpdf(x,posteriors[i][0], posteriors[i][1]))
#plt.legend(('1', '2', '3', '4', '5', '6'))

#plt.show()

#mu = encoder.params[0]
#print mu
X = np.matrix(X)
plt.plot(X[:,1],X[:,0],color='red')
#plt.plot(X[:,1],np.dot(X[:,1],mu),color='blue')
plt.show()
#at some point we will have 
'''
