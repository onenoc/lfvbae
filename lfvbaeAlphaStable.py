import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX

class VA:
    def __init__(self, dimTheta, m, n, learning_rate=0.01):
        '''
        @param m: number of samples
        @param n: dimension
        @param L: number of samples to draw
        '''
        self.dimTheta = dimTheta
        self.m = m
        self.n = n
        self.iterations = 0
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
        self.params = [mu,logSigma,logLambda]

    def createObjectiveFunction(self):
        '''
        @escription: initialize objective function and minimization function
        @X,y data matrix/vector
        @u random noise for simulator
        @v standard normal for reparametrization trick
        '''
        X = T.dmatrices("X")
        W, U = T.dvectors("W","U")

        mu = self.params[0]
        logSigma = self.params[1]
        logLambda = self.params[2]

        negKL = 0.5*self.dimTheta+0.5*T.sum(2*logSigma - mu ** 2 - T.exp(logSigma) ** 2)

        result,updates = th.map(fn=self.alpha_stable,sequences=[W,U],n_steps=self.m)
        f = results

        logLike = -self.m*(0.5 * np.log(2 * np.pi) + logLambda)-0.5*T.sum((y-f)**2)/(T.exp(logLambda)**2)/self.Lu

        elbo = (negKL + logLike)
        obj = -elbo
        self.lowerboundfunction = th.function([X,W,U], obj, updates=updates,on_unused_input='ignore')
        derivatives = T.grad(obj,self.params)
        self.gradientfunction = th.function([X,W,U], derivatives, updates=updates,on_unused_input='ignore')

    def alpha_stable(self,w,u,v):
        mu = self.params[0]
        logSigma = self.params[1]
        alpha = mu[0]+T.exp(logSigma[0])v[0]
        beta = mu[1]+T.exp(logSigma[1])v[1]
        gamma = mu[2]+T.exp(logSigma[2])v[2]
        delta = mu[3]++T.exp(logSigma[3])v[3]
        S_var = S(alpha,beta)
        B_var = B(alpha,beta)
        first = T.sin(alpha*(u+B_var)/(T.cos(u)**(alpha/2)))
        second = T.cos(u-alpha*(u+B_var))/w
        return S_var*first*(second**((1-alpha)/alpha))

    def S(self, alpha, beta):
        inner = 1+(beta**2)*(T.tan(T.pi*alpha/2)**2)
        S = inner**(-1/(2*alpha))
        return S

    def B(self, alpha, beta):
        return T.arctan(beta*T.tan(T.pi*alpha/2))

    def iterate(self,batch):
        X = batch[:,1:]
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

