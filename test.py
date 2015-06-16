import lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
#make univariate data, univariate param

#dimX, dimTheta, m, n
'''
encoder = lfvbae.VA(1, 2, 2, 1, 10, 1)
encoder.createGradientFunctions()
encoder.initParams()
#The following test the lower bound for training data
#negKL between standard normal and standard normal should be 0
print "this should be 0"
print encoder.negKL([[0]], [[1]])

print "this should be -1"
print encoder.negKL([[1], [1]], [[1], [1]])
#this should be 5
#print "this should be 5"
#[X, theta, u]
print encoder.f([[10, 2, 1]], [[2], [0]], [[1]])

print "these should be approximately -0.92"
#[mu, sigma, lambd]
gradvariables = np.array([[[0], [0]], [[1], [1]], [[1]]])
u=np.array([[1]])
v=np.array([[1], [1]])
print encoder.logLike(*(gradvariables), X=[[2, 0, 1]], u=[[1]], v=[[1], [1]])
print encoder.lowerboundfunction(*(gradvariables), X=[[2, 0, 1]], u=[[1]], v=[[1], [1]])

'''
m = 500
#dimX, dimTheta, m, n
encoder = lfvbae.VA(1, 1, m, 1, learning_rate=0.0000001)
encoder.initParams()
encoder.createGradientFunctions()


X = np.random.uniform(0, 100,m)
e = np.random.normal(0, 0.001,m)
Y = 2*X+e
X = np.column_stack((Y,X))
#print "data y,x"
#print X
#we will need to add bias
for i in range(10000):
    if i%50==0:
        print "intercept mean, sigma, lambda"
        print encoder.params
    encoder.iterate(X)
'''
#mu = encoder.params[0]

#X = np.matrix(X)
#plt.plot(X[:,0],X[:,1],color='red')
#plt.plot(X[:,0],np.dot(X[:,0],mu),color='blue')
#plt.show()

#at some point we will have 

#q is normally distributed with mean mu and variance sigma
#take theta=mu, plot (X,X*theta) in one color,(X,Y) in another color

#generate 100 thetas
#mu = encoder.mu
'''
