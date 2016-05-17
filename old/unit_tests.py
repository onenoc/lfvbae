import lfvbaeold as lfvbae
import theano as th
import numpy as np

'''
#make univariate data, univariate param
#dimX, dimTheta, m, n
encoder = lfvbae.VA(1, 2, 2, 1, 10, 1)
encoder.createGradientFunctions()
encoder.initParams()
#The following test the lower bound for training data
#negKL between standard normal and standard normal should be 0
print "this should be 0"
print encoder.negKL([[0]], [[np.log(1)]])

print "this should be -1"
print encoder.negKL([[1], [1]], [[np.log(1)], [np.log(1)]])
#this should be 5
print "this should be 5"
#[mu, logSigma, logLambd]
gradvariables = np.array([[[2], [0]], [[np.log(1)], [np.log(1)]], [[1]]])
#[X, theta, u]
print encoder.f(*(gradvariables), X=[[10, 2, 1]], u=[[1]], v=[[0], [0]])

print "this should be approximately -0.92"
#[mu, logSigma, logLambd]
gradvariables = np.array([[[0], [0]], [[0], [0]], [[0]]])
u=np.array([[1]])
v=np.array([[1], [1]])
print encoder.logLike(*(gradvariables), X=[[2, 0, 1]], u=[[1]], v=[[1], [1]])
print "this should be approximately -0.92/2"
print encoder.lowerboundfunction(*(gradvariables), X=[[2, 0, 1]], u=[[1]], v=[[1], [1]])
'''

