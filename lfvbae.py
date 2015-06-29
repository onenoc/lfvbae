import numpy as np
import theano as th
import theano.tensor as T
from pylearn2.utils import sharedX
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent

class VA:
    def __init__(self, dimX, dimTheta, m, n, sigma_e):
        '''
        @param m: number of samples
        @param n: dimension
        @param L: number of samples to draw
        '''
        self.dimX = dimX
        self.dimTheta = dimTheta
        self.m = m
        self.n = n
        self.sigma_e = sigma_e
        self.iterations = 0
        self.minCostParams = []
        self.lowerBounds = []

    def initParams(self):
        '''
        @description: parameters to learn
        '''
        mu = sharedX(np.random.normal(0, 10, (self.dimTheta, 1)), name='mu')
        logSigma = sharedX(np.random.uniform(0, 10, (self.dimTheta, 1)), name='logSigma')
        logLambda = sharedX(np.random.uniform(0, 10), name='logLambda')
        self.params = [mu,logSigma, logLambda]

    def createObjectiveFunction(self):
        '''
        @description: initialize objective function and minimization function
        '''
        X = T.dmatrix("X")
        y = T.vector("y")
        u, v, f = T.vectors("u", "v", "f")

        mu = self.params[0]
        #mu = sharedX(np.array([[-3]]),name='mu')
        logSigma = self.params[1]
        logLambda = sharedX(np.log(self.sigma_e),name='logLambda')
        #logLambda = self.params[2]

        negKL = 0.5*self.dimTheta+0.5*T.sum(2*logSigma - mu ** 2 - T.exp(logSigma) ** 2)
        theta = mu + T.exp(logSigma)*v.dimshuffle(0,'x')
        #the only thing that could be wrong here is the T.exp(logSigma)*v.dimshuffle(0,'x')
        f = T.dot(X,theta)+u.dimshuffle(0,'x')
        #or the u.dimshuffle(0,'x')

        logLike = -self.m*(0.5 * np.log(2 * np.pi) + logLambda)-0.5*T.sum((y.dimshuffle(0,'x')-f)**2)/(T.exp(logLambda)**2)

        elbo = (negKL + logLike)
        obj = -elbo
        self.minimizer = BatchGradientDescent(objective = obj,params = self.params,inputs = [X,y,u,v],max_iter=1,conjugate=1)
        #derivatives = T.grad(obj,self.params)
           
    def iterate(self,batch):
        X = batch[:,1:]
        y = batch[:,0]
        v = np.random.normal(0, 1,self.dimTheta)
        u = np.random.normal(0, self.sigma_e**2,self.m)
        cost = self.minimizer.minimize(X,y,u,v)
        #keep track of min cost and its parameters
        if self.iterations == 0 or cost < self.minCost:
            self.minCost = cost
            self.minCostParams = [self.params[0].get_value(), np.exp(self.params[1].get_value()),np.exp(self.params[2].get_value())]
        self.lowerBounds.append(cost)
        if self.iterations % 300 == 0:
            print "theta"
            self.print_parameters()
        self.iterations += 1

    def print_parameters(self):
        print "\n"
        print "cost"
        print self.lowerBounds[-1]
        print "mu"
        print self.params[0].get_value()
        print "sigma"
        print np.exp(self.params[1].get_value())
        print "lambda"
        print np.exp(self.params[2].get_value())

