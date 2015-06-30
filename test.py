import lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

def generate_data(m,n,weight_vector,bias,sigma_e):
    X = np.random.uniform(0, 1,(m, n))
    e = np.random.normal(0, (sigma_e)**2,(m,1))
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

def plot_cost(encoder):
    plt.plot(encoder.lowerBounds[1500:])
    plt.show()

m = 20
n=1
bias=0
sigma_e=0.1

y,X = generate_data(m,n,np.array([2]),bias, sigma_e)
muSDTrue, varSDTrue = true_posterior_standard_normal(n, bias, sigma_e,X,y)

batch = np.column_stack((y,X))

#dimX, dimTheta, m, n
encoder = lfvbae.VA(n+bias, n+bias, m, 1, sigma_e)

encoder.initParams()
encoder.createObjectiveFunction()

for i in range(10000):
    encoder.iterate(batch)

'''
muVar = encoder.params[0].get_value()
varVar = encoder.params[1].get_value()
print "minCost"
print encoder.minCost
print "minCost params"
print encoder.minCostParams
'''

#plot_cost(encoder)
print "true posterior"
print muSDTrue, np.sqrt(varSDTrue)

print "MLE sigma"
print np.sqrt((sigma_e**2)*np.linalg.inv(np.dot(X.T,X)))

#xMin = min(muVar,muTrue)-2*max(sigmaVar,sigmaTrue)
#xMax = max(muVar,muTrue)+2*max(sigmaVar,sigmaTrue)

#x = np.linspace(xMin,xMax, 1000)

#plt.plot(x,mlab.normpdf(x,muVar,sigmaVar))
#plt.plot(x,mlab.normpdf(x,muTrue,sigmaTrue))
#plt.show()

#fix lambda for variational inference


