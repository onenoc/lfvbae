import sys
import numpy as np
import theano as th
import theano.tensor as T
import math
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX

class VA:
    def __init__(self, dimTheta, m, learning_rate=0.01, i=100, N_fw=2000):
        '''
        @param m: number of samples
        @param n: dimension
        @param L: number of samples to draw
        '''
        self.dimTheta = dimTheta
        self.m = m
        self.iterations = 0
        self.minCostParams = []
        self.lowerBounds = []
        self.converge = 0
        self.learning_rate = learning_rate
        self.i = i
        self.N_fw = N_fw

    def initParams(self):
        '''
        @description: parameters to learn
        '''
        mu = sharedX(np.random.uniform(0, 5, (self.dimTheta, 1)), name='mu')
        logSigma = sharedX(np.random.uniform(0, 0.25, (self.dimTheta, 1)), name='logSigma')
        logLambda = sharedX(np.random.uniform(0, 10), name='logLambda')
        self.params = [mu, logSigma,logLambda]

    def createObjectiveFunction(self):
        '''
        @escription: initialize objective function and minimization function
        @X,y data matrix/vector
        @u random noise for simulator
        @v standard normal for reparametrization trick
        '''
        y = T.dmatrices("y")
        i = T.iscalar("i")
        v = T.dscalar("i")
        xStart = T.dvector("xStart")

        mu = self.params[0]
        #logSigma = sharedX(np.random.uniform(0, 1, (self.dimTheta, 1)), name='logSigma')
        logSigma = self.params[1]
        #logLambda = sharedX(np.random.uniform(0, 10), name='logLambda')
        logLambda = self.params[2]

        negKL = 0.5*self.dimTheta+0.5*T.sum(2*logSigma - mu ** 2 - T.exp(logSigma) ** 2)
        self.k = mu+T.exp(logSigma)*v
        V1 = T.dvector("V2")
        V2 = T.dvector("V2")
        results, updates = th.scan(fn=self.fisher_wright_normal_approx, outputs_info=[{'initial':xStart,'taps':[-1]}],sequences=[V1,V2], n_steps=i)
        f = results

        logLike = -self.m*(0.5 * np.log(2 * np.pi) + logLambda)-0.5*T.sum((y-f)**2)/(T.exp(logLambda)**2)
        part2 = f
        #0.5*T.sum((y-f)**2)
        #/(T.exp(logLambda)**2)
        elbo = (negKL + logLike)
        obj = -elbo
        test1 = y[0:self.i/4,:].sum(axis=0)/(self.i/4)
        test2 = y[self.i/4:self.i/2].sum(axis=0)/(self.i/4)
        self.test = th.function([xStart, i, y, v, V1, V2],test,on_unused_input='ignore')
        self.part2 = th.function([xStart, i, y, v, V1, V2], part2, updates=updates, on_unused_input='ignore')
        self.logLike = th.function([xStart, i, y, v, V1, V2], logLike, updates=updates, on_unused_input='ignore')
        self.lowerboundfunction = th.function([xStart, i, y, v, V1, V2], obj, updates=updates, on_unused_input='ignore')
        derivatives = T.grad(obj, self.params)
        self.gradientfunction = th.function([xStart, i, y, v, V1, V2], derivatives, updates=updates, on_unused_input='ignore')

    def summary_statistics(self, y):
        s1 = y[0:25,:].sum(axis=0)/(0.25*self.i)

    def create_trajectory(self,xStartInput,kInput):
        U1 = T.dmatrix("U2")
        U2 = T.dmatrix("U2")
        xStart = T.vector('xStart')
        results, updates = th.scan(fn=self.fisher_wright, outputs_info=[{'initial':xStart,'taps':[-1]}],sequences=[U1,U2], n_steps=self.i)
        f = th.function([xStart, U1,U2,k],results)
        U1 = np.random.uniform(0, 1, (self.i, self.N_fw))
        U2 = np.random.uniform(0, 1, (self.i, self.N_fw))
        return f(xStartInput,U1,U2,kInput)

    def fisher_wright(self, un1, un2, x):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        N = sharedX(self.N_fw,name='N')
        p1 = sharedX(0.1,name='N')
        k = (6*T.exp(self.k)+1)/(T.exp(self.k)+1)
        p0 = 1/(1+self.k*x2/N)
        q = x0*p0/(x0+x1)
        qhat = (x0*(1-p0)+x1*p1)/((x0+x1)*(1-q))
        x0n = self.binomial(q, un1, T.cast(N, 'int32'))
        x1n = self.binomial(qhat, un2, T.cast(N-x0n, 'int32'))
        x2n = N-x0n-x1n
        xOut = T.stack(x0n,x1n,x2n)
        return T.flatten(xOut)

    def fisher_wright_normal_approx(self,v1,v2,x):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        N = sharedX(self.N_fw,name='N')
        p1 = sharedX(0.1,name='N')
        k = (6*T.exp(self.k)+1)/(T.exp(self.k)+1)
        p0 = 1/(1+k*x2/N)
        q = x0*p0/(x0+x1)
        qhat = (x0*(1-p0)+x1*p1)/((x0+x1)*(1-q))
        x0n = N*q+v1*T.sqrt((N*q*(1-q)))
        x1n = (N-x0n)*qhat+v2*T.sqrt(((N-x0n)*qhat*(1-qhat)))
        x2n = N-x0n-x1n
        xOut = T.stack(x0n,x1n,x2n)
        return T.flatten(xOut)

    def bernoulli(self, u,p):
        return T.le(u,p)

    def binomial(self, p, un, n):
        results, updates = th.scan(fn=self.bernoulli, outputs_info=None, sequences=un[0:n], non_sequences=p)
        binomial = results.sum()
        return binomial

    def iterate(self,xStart, y):
        v = np.random.normal(0, 1)
        V1 = np.random.normal(0,1,self.i)
        V2 = np.random.normal(0,1,self.i)
        cost = self.lowerboundfunction(xStart,self.i,y,v,V1,V2)
        gradients = self.getGradients(xStart,y)
        self.updateParams(gradients)
        self.lowerBounds.append(cost)
        change = 0
        if len(self.lowerBounds) > 11:
            l2 = sum(self.lowerBounds[-10:])/(self.m*10)
            l1 = sum(self.lowerBounds[-11:-1])/(self.m*10)
            change = abs((l2-l1)/l1)
            if change<0.00000001:
                self.converge = 1
                print "convergence change"
                print change
        if self.iterations %100==0:
            print "test"
            print self.test(xStart,self.i,y,v,V1,V2)
            print "gradients"
            print gradients
            #print change
            self.print_parameters()
        if math.isnan(cost):
            print "logLike"
            print self.logLike(xStart,self.i,y,v,V1,V2)
            print "part2"
            print self.part2(xStart,self.i,y,v,V1,V2)
            mu = self.params[0].get_value()
            sigma = np.exp(self.params[1].get_value())
            print "mu, sigma"
            print mu, sigma
            k_tild = mu+v*sigma
            print "k"
            print (6*np.exp(k_tild)+1)/(np.exp(k_tild)+1)
            sys.exit()
        self.iterations += 1

    def print_parameters(self):
        print "\n"
        print "cost"
        print self.lowerBounds[-1]
        print "mu"
        mu = self.params[0].get_value()
        print (6*np.exp(mu)+1)/(np.exp(mu)+1)
        print "sigma"
        sigma = np.exp(self.params[1].get_value())
        print np.exp(self.params[1].get_value())
        print "95% interval"
        print (6*np.exp(mu+2*sigma)+1)/(np.exp(mu+2*sigma)+1), (6*np.exp(mu-2*sigma)+1)/(np.exp(mu-2*sigma)+1)
        print "lambda"
        print np.exp(self.params[2].get_value())

    def getGradients(self,xStart, y):
        v = np.random.normal(0, 1)
        V1 = np.random.uniform(0,1,self.i)
        V2 = np.random.uniform(0,1,self.i)
        gradients = self.gradientfunction(xStart,self.i,y,v,V1,V2)
        return gradients

    def updateParams(self,gradients):
        """Update the parameters, taking into account AdaGrad and a prior"""
        for i in xrange(len(self.params)):
            self.params[i].set_value(self.params[i].get_value()-gradients[i]/(1/self.learning_rate+self.iterations)) 
