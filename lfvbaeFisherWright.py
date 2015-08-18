import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX

class VA:
    def __init__(self, dimTheta, m, learning_rate=0.01, i=100):
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

    def initParams(self):
        '''
        @description: parameters to learn
        '''
        mu = sharedX(np.random.normal(0, 10, (self.dimTheta, 1)), name='mu')
        logSigma = sharedX(np.random.uniform(0, 1, (self.dimTheta, 1)), name='logSigma')
        logLambda = sharedX(np.random.uniform(0, 10), name='logLambda')
        self.params = [mu,logSigma, logLambda]

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
        logSigma = self.params[1]
        logLambda = self.params[2]
        #logLambda = sharedX(np.log(self.sigma_e),name='logLambda')

        negKL = 0.5*self.dimTheta+0.5*T.sum(2*logSigma - mu ** 2 - T.exp(logSigma) ** 2)
        k = mu+T.exp(logSigma)*v
        f, updates = th.scan(fn=self.fisher_wright, outputs_info=[{'initial':xStart,'taps':[-1]}],non_sequences=k, n_steps=i)
        #f, updates = th.scan(fn=self.fisher_wright, outputs_info=[{'initial':xStart,'taps':[-1]}],non_sequences=k, n_steps=i)
        self.f2 = th.function(inputs=[xStart, i, v], outputs=f, updates=updates)
        #self.f2 = th.function(inputs=[xStart], outputs=xStart)
        test = self.fisher_wright(xStart, k)
        self.test_f = th.function([xStart, v], test)
        #logLike = T.sum(y-f)
        #-self.m*(0.5 * np.log(2 * np.pi) + logLambda)-0.5*T.sum((y-f)**2)/(T.exp(logLambda)**2)

        #elbo = (negKL + logLike)
        #obj = -elbo
        #self.lowerboundfunction = th.function([xStart, i, y, v], obj, updates=updates, on_unused_input='ignore')
        #derivatives = T.grad(obj, self.params)
        #self.gradientfunction = th.function([xStart, i, y, v], derivatives, updates=updates, on_unused_input='ignore')

    def fisher_wright(self, x, k):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        N = sharedX(2000.0,name='N')
        p1 = sharedX(0.1,name='N')
        p0 = 1/(1+k*x2/N)
        q = x0*p0/(x0+x1)
        qhat = (x0*(1-p0)+x1*p1)/((x0+x1)*(1-q))
        srng = RandomStreams()
        x0n = srng.binomial(n=N,p=q)
        #srng2 = RandomStreams(seed=234)
        x1n = srng.binomial(n=N-x0n,p=qhat)
        x2n = N-x0n-x1n
        xOut = T.stack(x0n,x1n,x2n)
        return T.flatten(xOut)
