import lfvbae
import theano as th
import numpy as np

#make univariate data, univariate param
encoder = lfvbae.VA(1, 2, 2, 1, 10, 1)

encoder.createGradientFunctions()

'''
#The following test the lower bound for training data
#negKL between standard normal and standard normal should be 0
print "this should be 0"
print encoder.negKL([[0]], [[1]])

#this should be 5
print "this should be 5"
print encoder.f([[2], [2]], [[1]])

print "these should be approximately -5.84"
#[mu, sigma, lambd]
gradvariables = [[[0], [0]], [[1], [1]], [[1]]]
print encoder.logLike(*(gradvariables), X=[[0, 0]], u=[[1]], v=[[1], [1]])
print encoder.lowerboundfunction(*(gradvariables), X=[[0, 0]], u=[[1]], v=[[1], [1]])
'''

Y = np.random.normal(0, 1,[1,100])
encoder.initParams()
encoder.initH(Y)
for i in range(100):
    encoder.iterate(Y)

#at some point we will have 

#how do we test whether this is right?

#q is normally distributed with mean mu and variance sigma

#generate 100 thetas
mu = encoder.mu
sigma = encoder.sigma
v = np.random.normal(0, 1,[encoder.dimTheta,1])
thetas = np.zeros(100)
#for i in range(100):

#setup the w's, x's
#see if the weights have correct variance
