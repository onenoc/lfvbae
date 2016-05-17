import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX

class VA:
    def __init__(self, dimX, dimTheta, m, n, sigma_e, Lu=1, learning_rate=0.01):
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
        self.Lu = Lu
        self.minCostParams = []
        self.lowerBounds = []
        self.converge = 0
        self.learning_rate = learning_rate

    def initParams(self):
        '''
        @description: parameters to learn
        '''
        mu = sharedX(np.random.normal(0, 10, (self.dimTheta, 1)), name='mu')
        logSigma = sharedX(np.random.uniform(0, 1, (self.dimTheta, 1)), name='logSigma')
        logLambda = sharedX(np.random.uniform(0, 10), name='logLambda')
        self.params = [mu,logSigma]

    def createObjectiveFunction(self):
        '''
        @escription: initialize objective function and minimization function
        @X,y data matrix/vector
        @u random noise for simulator
        @v standard normal for reparametrization trick
        '''
        X,u = T.dmatrices("X","u")
        f, y, v = T.dcols("f", "y", "v")
        
        mu = self.params[0]
        logSigma = self.params[1]
        logLambda = sharedX(np.log(self.sigma_e),name='logLambda')
        #logLambda = self.params[2]

        negKL = 0.5*self.dimTheta+0.5*T.sum(2*logSigma - mu ** 2 - T.exp(logSigma) ** 2)
        f = self.regression_simulator(X,u,v,mu,logSigma)

        logLike = -self.m*(0.5 * np.log(2 * np.pi) + logLambda)-0.5*T.sum((y-f)**2)/(T.exp(logLambda)**2)/self.Lu

        elbo = (negKL + logLike)
        obj = -elbo
        self.lowerboundfunction = th.function([X, y, u, v], obj, on_unused_input='ignore')
        derivatives = T.grad(obj,self.params)
        self.gradientfunction = th.function([X,y,u,v], derivatives, on_unused_input='ignore')

    def regression_simulator(self,X,u,v,mu,logSigma):
        theta = mu + T.exp(logSigma)*v
        predval = T.dot(X,theta)
        predval = T.addbroadcast(predval,1)
        #+u
        return predval

    def iterate(self,batch):
        X = batch[:,1:]
        y = np.matrix(batch[:,0]).T
        v = np.random.normal(0, 1,(self.dimTheta,1))
        u = np.random.normal(0, self.sigma_e,(self.m,self.Lu))
        cost = self.lowerboundfunction(X=X,y=y,u=u,v=v)
        gradients = self.getGradients(batch)
        self.updateParams(gradients)
        self.lowerBounds.append(cost)
        change = 0
        if len(self.lowerBounds) > 11:
            l2 = sum(self.lowerBounds[-10:])/(self.m*10)
            l1 = sum(self.lowerBounds[-11:-1])/(self.m*10)
            change = abs((l2-l1)/l1)
            if change<0.0000000025:
                self.converge = 1
                print "convergence change"
                print change
        if self.iterations % 300 == 0:
            print change
            self.print_parameters()
        self.iterations += 1

    def print_parameters(self):
        print "\n"
        #print "cost"
        #print self.lowerBounds[-1]
        print "mu"
        print self.params[0].get_value()
        print "sigma"
        print np.exp(self.params[1].get_value())
        #print "lambda"
        #print np.exp(self.params[2].get_value())

    def getGradients(self,batch):
        """Compute the gradients for one minibatch and check if these do not contain NaNs"""
        X = batch[:,1:]
        y = np.matrix(batch[:,0]).T
        v = np.random.normal(0, 1,(self.dimTheta,1))
        u = np.random.normal(0, self.sigma_e,(self.m,self.Lu))
        gradients = self.gradientfunction(X=X,y=y,u=u,v=v)
        return gradients

    def updateParams(self,gradients):
        """Update the parameters, taking into account AdaGrad and a prior"""
        for i in xrange(len(self.params)):
            self.params[i].set_value(self.params[i].get_value()-gradients[i]/(1/self.learning_rate+self.iterations))
