import numpy as np
import theano as th
import theano.tensor as T


class VA:
    def __init__(self, dimX, dimTheta, m, n, L=1, learning_rate = 0.01):
        self.dimX = dimX
        self.dimTheta = dimTheta
        self.m = m
        self.n = n
        self.L = L
        self.iterations = 0
        self.learning_rate = learning_rate

    def initParams(self):
        mu = np.random.normal(10, 1, (self.dimTheta, 1))
        sigma = np.random.uniform(0, 0.01, (self.dimTheta, 1))
        lambd = np.matrix(np.random.uniform(0.95, 1.05))
        self.params = [mu, sigma, lambd]
        
    def createGradientFunctions(self):
        #create
        X = T.dmatrices("X")
        mu, sigma, u, v, f, R = T.dcols("mu", "sigma", "u", "v", "f", "R")
        lambd = T.patternbroadcast(T.dmatrix("lambd"),[1,1])
        negKL = 0.5 * T.sum(1 + 2*T.log(abs(sigma)) - mu ** 2 - sigma ** 2)
        theta = mu+sigma*v
        W=theta
        y=X[:,0]
        X_sim=X[:,1:]
        f = (T.dot(X_sim,W)+u).flatten()
        
        gradvariables = [mu, sigma, lambd]
        
        
        logLike = T.sum(-(0.5 * np.log(2 * np.pi) + T.log(abs(lambd))) - 0.5 * ((y-f)/(lambd))**2)

        logp = negKL + logLike
        
        self.negKL = th.function([mu, sigma], negKL, on_unused_input='ignore')
        self.f = th.function(gradvariables + [X,u,v], f, on_unused_input='ignore')
        self.logLike = th.function(gradvariables + [X, u, v], logLike,on_unused_input='ignore')
        derivatives = T.grad(logp,gradvariables)
        derivatives.append(logp)

        self.gradientfunction = th.function(gradvariables + [X, u, v], derivatives, on_unused_input='ignore')
        self.lowerboundfunction = th.function(gradvariables + [X, u, v], logp, on_unused_input='ignore')

    def iterate(self, data):
        '''''Main method, slices data in minibatches and performs an iteration'''''
        totalGradients = self.getGradients(data)
        self.iterations += 1
        self.updateParams(totalGradients)

    def getGradients(self, batch):
        totalGradients = [0] * len(self.params)
        #in our case, we only use a single sample
        v = np.random.normal(0, 1,[self.dimTheta,1]) 
        u = np.random.normal(0, 0.001,[self.m,1])
        gradients = self.gradientfunction(*(self.params),X=batch,u=u,v=v)
        '''
        print "log-likelihood"
        print self.logLike(*(self.params),X=batch,u=u,v=v)
        y = batch[:,0]
        print "(y-f)^2"
        print np.sum(((y-self.f(*(self.params), X=batch,u=u,v=v)))**2)
        print y
        print self.f(*(self.params), X=batch,u=u,v=v)
        print y-self.f(*(self.params), X=batch,u=u,v=v).flatten()
        print "simulator output"
        print self.f(*(self.params), X=batch,u=u,v=v)
        print "neg-kl"
        print self.negKL(self.params[0], self.params[1])
        '''
        if self.iterations % 100 == 0:
            print "lower bound"
            print self.lowerboundfunction(*(self.params),X=batch,u=u,v=v)
        for i in xrange(len(self.params)):
            totalGradients[i] += gradients[i]
        '''
        print "gradients"
        print totalGradients
        '''
        return totalGradients

    def updateParams(self, totalGradients):
        for i in xrange(len(self.params)):
            self.params[i] += self.learning_rate * totalGradients[i]

