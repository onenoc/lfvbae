import lfvbae
import theano as th
import numpy as np

#make univariate data, univariate param
encoder = lfvbae.VA(1, 2, 2, 1, 10, 1)

encoder.createGradientFunctions()

#The following test the lower bound for training data
'''
#negKL between standard normal and standard normal should be 0
print "this should be 0"
print encoder.negKL([[0]], [[1]])

#this should be 5
print "this should be 5"
print encoder.f([[2], [2]], [[1]])

#[X, mu, sigma, lambd, u, v]
print "this should be approximately -5.84"
print encoder.logLike([[0, 0]], [[0], [0]], [[1], [1]], [[1]], [[1]], [[1], [1]])
'''
X = np.array([[1, 1, 2, 5, 1]])
encoder.initParams()
encoder.initH(X)
encoder.iterate(X)

