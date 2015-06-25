import numpy as np
import theano as th
import theano.tensor as T
from pylearn2.utils import sharedX
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent

class VA:
    def __init__(self, dimX, dimTheta, m, n, L=1):
        '''
        @param m: number of samples
        @param n: dimension
        @param L: number of samples to draw
        '''
        self.dimX = dimX
        self.dimTheta = dimTheta
        self.m = m
        self.n = n
        self.L = L
        self.iterations = 0
        self.lowerBounds = []

    def initParams(self):
        '''
        @description: parameters to learn
        '''
        mu = sharedX(np.random.normal(0, 10, (self.dimTheta, 1)), name='mu')
        logSigma = sharedX(np.random.uniform(0, 10, (self.dimTheta, 1)), name='logSigma')
        logLambda = sharedX(np.random.uniform(0, 10), name='logLambda')
        self.params = [mu, logSigma, logLambda]

    def createObjectiveFunction(self):
        '''
        @description: initialize objective function and minimization function
        '''
        X = T.dmatrix("X")
        y = T.vector("y")
        u, v, f = T.vectors("u", "v", "f")

        mu = self.params[0]
        logSigma = self.params[1]
        logLambda = self.params[2]

        negKL = 0.5*self.dimTheta+T.sum(2*logSigma - mu ** 2 - T.exp(logSigma) ** 2)
        theta = mu + T.exp(logSigma)*v.dimshuffle(0,'x')
        f = T.dot(X,theta)+u.dimshuffle(0,'x')

        logLike = -self.m*(0.5 * np.log(2 * np.pi) + logLambda)-0.5*T.sum((y.dimshuffle(0,'x')-f)**2)/(T.exp(logLambda)**2)

        logp = (negKL + logLike)/self.m
        obj = -logp
        self.minimizer = BatchGradientDescent(objective = obj,params = self.params,inputs = [X,y,u,v],max_iter=1,conjugate=1)
       
    def iterate(self,batch):
        X = batch[:,1:]
        y = batch[:,0]
        v = np.random.normal(0, 1,self.dimTheta)
        u = np.random.normal(0, 0.01,self.m)
        print "iteration"
        print self.minimizer.minimize(X,y,u,v)
        print self.params[0].get_value()
        print self.params[1].get_value()
        print self.params[2].get_value()
