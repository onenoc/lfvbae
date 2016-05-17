import numpy as np
import theano as th
import theano.tensor as T
import math
import sys
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
        self.pi = sharedX(3.14159265359,name='pi')
        mu = sharedX(np.random.uniform(0, 4, (self.dimTheta, 1)), name='mu')
        logSigma = sharedX(np.random.uniform(0, 4, (self.dimTheta, 1)), name='logSigma')
        logLambda = sharedX(np.random.uniform(0, 10), name='logLambda')
        print "initial parameters"
        print mu.get_value()
        print logSigma.get_value()
        print logLambda.get_value()
        #ADD LOGLAMBDA
        self.params = [mu]

    def createObjectiveFunction(self):
        '''
        @escription: initialize objective function and minimization function
        @X,y data matrix/vector
        @u random noise for simulator
        @v standard normal for reparametrization trick
        '''
        y = T.dvector("y")
        W, U = T.dvectors("W","U")
        V = T.dscalar("V")

        mu = self.params[0]
        #logSigma = self.params[1]
        logSigma = sharedX(0.6)
        logLambda = sharedX(0)
        #self.params[2]

        negKL = 0.5*self.dimTheta+0.5*T.sum(2*logSigma - mu ** 2 - T.exp(logSigma) ** 2)

        results,updates = th.map(fn=self.alpha_stable,sequences=[W,U],non_sequences=[V])
        f = results
        results2,updates2 = th.map(fn=self.alpha_perfect,sequences=[W,U])
        f2 = results2

        #SSE = T.sum((y-f)**2)
        logLike = -self.m*(0.5 * np.log(2 * np.pi) + logLambda)-0.5*T.sum((T.flatten(y)-T.flatten(f))**2)/(T.exp(logLambda)**2)
        #logLike2 = -self.m*(0.5 * np.log(2 * np.pi) + logLambda)-0.5*T.sum((y-f2)**2)/(T.exp(logLambda)**2) 

        elbo = (negKL + logLike)
        #elbo2 = (negKL + logLike2)
        obj = -elbo
        #obj = SSE

        self.f = th.function([y,W,U,V],f,updates=updates,on_unused_input='ignore')
        self.lowerboundfunction = th.function([y,W,U,V], obj, updates=updates,on_unused_input='ignore')
        derivatives = T.grad(obj,self.params)
        self.gradientfunction = th.function([y,W,U,V], derivatives, updates=updates,on_unused_input='ignore')

    def alpha_perfect(self, w, u):
        alpha = sharedX(1.5)
        beta = sharedX(0.5)
        S_var = self.S(alpha,beta)
        B_var = self.B(alpha,beta)
        first = T.sin(alpha*(u+B_var)/(T.cos(u)**(1/alpha)))
        second = T.cos(u-alpha*(u+B_var))/w
        return S_var*first*(second**((1-alpha)/alpha))

    def alpha_stable(self,w,u,v):
        mu = self.params[0]
        logSigma = sharedX(0.6)
        #logSigma = self.params[1]
        #alpha = mu+T.exp(logSigma)*v
        alpha=sharedX(1.9)
        beta = mu+T.exp(logSigma)*v
        #alpha = (2*T.exp(alpha)+1.1)/(T.exp(alpha)+1)
        beta = (T.exp(beta)-1)/(T.exp(beta)+1)
        #beta = sharedX(0.5)
        #mu[1]+T.exp(logSigma[1])*v[1]
        #gamma = mu[2]+T.exp(logSigma[2])*v[2]
        #delta = mu[3]+T.exp(logSigma[3])*v[3]
        S_var = self.S(alpha,beta)
        B_var = self.B(alpha,beta)
        first = T.sin(alpha*(u+B_var)/(T.cos(u)**(1/alpha)))
        second = T.cos(u-alpha*(u+B_var))/w
        return S_var*first*(second**((1-alpha)/alpha))

    def S(self, alpha, beta):
        inner = 1+(beta**2)*(T.tan(self.pi*alpha/2)**2)
        S = inner**(1/(2*alpha))
        return S

    def B(self, alpha, beta):
        return (1/alpha)*T.arctan(beta*T.tan(self.pi*alpha/2))

    def iterate(self,batch):
        y = batch
        W = np.random.exponential(size=self.m)
        U = np.random.uniform(-np.pi/2,np.pi/2,self.m)
        V = np.random.normal(0,1)
        cost = self.lowerboundfunction(y=y,W=W,U=U,V=V)
        gradients = self.getGradients(batch)
        self.updateParams(gradients)
        self.lowerBounds.append(cost)
        change = 0
        if len(self.lowerBounds) > 11:
            l2 = sum(self.lowerBounds[-10:])/(self.m*10)
            l1 = sum(self.lowerBounds[-11:-1])/(self.m*10)
            change = abs((l2-l1)/l1)
            if change<0.00000000025:
                self.converge = 1
                print "convergence change"
                print change
        if self.iterations % 100 == 0:
            print "change"
            print change
            self.print_parameters()
        if math.isnan(cost):
            #print "f"
            print "hello there"
            #print self.f(y=y,W=W,U=U,V=V)
            mu = self.params[0].get_value()
            sigma = np.exp(self.params[1].get_value())
            print "mu, sigma"
            print mu, sigma
            print "V"
            print V
            sys.exit()
        self.iterations += 1

    def print_parameters(self):
        print "\n"
        print "cost"
        print self.lowerBounds[-1]
        mu = self.params[0]
        #alpha = mu.get_value()
        beta = mu.get_value()
        #print "alpha"
        #print (2*np.exp(alpha)+1.1)/(np.exp(alpha)+1)
        print "beta"
        print (np.exp(beta)-1)/(np.exp(beta)+1)
        #print "sigma"
        #print np.exp(self.params[1].get_value())
        #print "lambda"
        #print np.exp(self.params[2].get_value())

    def getGradients(self,batch):
        """Compute the gradients for one minibatch and check if these do not contain NaNs"""
        y = batch
        W = np.random.exponential(size=self.m)
        U = np.random.uniform(-np.pi/2,np.pi/2,self.m)
        V = np.random.normal(0,1)
        gradients = self.gradientfunction(y=y,W=W,U=U,V=V)
        #print "gradients"
        #print gradients
        if math.isnan(gradients[0][0][0]):
            print "f"
            print self.f(y=y,W=W,U=U,V=V)
            mu = self.params[0].get_value()
            sigma = np.exp(self.params[1].get_value())
            print "mu, sigma"
            print mu, sigma
            print "V"
            print V
            sys.exit()

        return gradients

    def updateParams(self,gradients):
        """Update the parameters, taking into account AdaGrad and a prior"""
        for i in xrange(len(self.params)):
            self.params[i].set_value(self.params[i].get_value()-gradients[i]/(1/self.learning_rate+self.iterations))

