import numpy as np
import theano as th
import theano.tensor as T

"""This class implements an auto-encoder with Variational Bayes"""

class VA:
    def __init__(self, dimX, dimTheta, batch_size, L, learning_rate = 0.01):
        self.dimX = dimX
        self.dimTheta = dimTheta
        self.L = L
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.sigmaInit = 0.01
        self.lowerbound = 0

        self.continuous = False

    def initParams(self):
        mu = np.random.normal(0, 1, (self.dimTheta, 1))
        sigma = np.random.normal(0, 1, (self.dimTheta, 1))
        lambd = np.random.normal(0, 1, (self.dimX, 1))
        self.params = [mu, sigma, lambd]
        self.h = [0.01] * len(self.params)
        
    def createGradientFunctions(self):
        #create
        mu,sigma,lambd,x,u,v,f = T.dmatrices("mu","sigma","lambd","X","u","v","f")
        
        negKL = 0.5 * T.sum(1 + 2*T.log(sigma) - mu ** 2 - sigma ** 2)
        theta = mu+sigma*v
        f = T.dot(theta[:self.dimTheta/2], theta[self.dimTheta/2:])+u
        #these are for testing
        self.negKL = th.function([mu, sigma], negKL)
        self.f = th.function([theta, u], f)
        #the log-likelihood depends on f and lambda
        #need to setup log-likelihood so dependent on all datapoints
        #this needs the 2 in denominator handled
        logLike = T.sum(-(0.5 * np.log(2 * np.pi) + T.log(lambd)) - 0.5 * ((x - f) / lambd)**2)
        self.logLike = th.function([x, mu, sigma, lambd, u, v], logLike)

        '''
        logp = negKL + logLike

        gradvariables = [mu, sigma, lambd]

        self.gradientfunction = th.function(gradvariables + [X, u, v], derivatives, on_unused_input='ignore')
        self.lowerboundfunction = th.function(gradvariables + [X, u, v], logp, on_unused_input='ignore')
        '''
