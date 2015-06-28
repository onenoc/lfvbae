import lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

def generate_data(m,n,weight_vector,bias,sigma_e):
    X = np.random.uniform(0, 10,(m, n))
    e = np.random.normal(0, sigma_e,(m,1))
    if bias:
        X = np.column_stack((X,np.ones(m)))
    dot = np.reshape(np.dot(X,weight_vector), (m,1))
    Y = dot+e
    return Y,X

def get_true_posterior(muPrior, sigmaPrior,n,bias, sigma_e,X,y):
    alpha = 1./sigmaPrior
    Sinv = np.dot(alpha, sigma_e*np.identity(n+bias))+np.dot(X.T,X)
    S = np.linalg.inv(Sinv)
    muTrue = np.dot(S,np.dot(X.T,y))
    return muTrue,np.sqrt(S)

def plot_cost(encoder):
    plt.plot(encoder.lowerBounds[1500:])
    plt.show()

m = 200
n=1
bias=0
sigma_e=0.01

y,X = generate_data(m,n,np.array([-3]),bias, sigma_e)

batch = np.column_stack((y,X))

#dimX, dimTheta, m, n
encoder = lfvbae.VA(n+bias, n+bias, m, 1)

encoder.initParams()
encoder.createObjectiveFunction()

muPrior = np.array([[0]])
sigmaPrior = np.array([[1]])

print muPrior, sigmaPrior

muTrue,sigmaTrue = get_true_posterior(muPrior,sigmaPrior,n,bias,sigma_e,X,y)

'''
for i in range(100000):
    encoder.iterate(batch)

muVar = encoder.params[0].get_value()
sigmaVar = encoder.params[1].get_value()
'''

#plot_cost(encoder)

print "true posterior"
print muTrue,sigmaTrue
'''
print "minCost"
print encoder.minCost
print "minCost params"
print encoder.minCostParams
'''

print "MLE sigma"
print np.sqrt((0.01**2)*np.linalg.inv(np.dot(X.T,X)))

#xMin = min(muVar,muTrue)-2*max(sigmaVar,sigmaTrue)
#xMax = max(muVar,muTrue)+2*max(sigmaVar,sigmaTrue)

#x = np.linspace(xMin,xMax, 1000)

#plt.plot(x,mlab.normpdf(x,muVar,sigmaVar))
#plt.plot(x,mlab.normpdf(x,muTrue,sigmaTrue))
#plt.show()

#fix lambda for variational inference


