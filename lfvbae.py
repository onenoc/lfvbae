import numpy as np
import theano as th
import theano.tensor as T


class VA:
    def __init__(self, dimX, dimTheta, m, n, batch_size, L, learning_rate = 0.01):
        self.dimX = dimX
        self.dimTheta = dimTheta
        self.m = m
        self.n = n
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

    def initH(self, batch):
        totalGradients = self.getGradients(batch)
        #why are we initializing h with the squared gradients for each param?
        for i in range(len(totalGradients)):
            self.h[i] += totalGradients[i]*totalGradients[i]
        
    def createGradientFunctions(self):
        #create
        X = T.dmatrices("X")
        mu, sigma, u, v, f, lambd = T.dcols("mu", "sigma", "u", "v", "f", "lambd")
        
        negKL = 0.5 * T.sum(1 + 2*T.log(sigma) - mu ** 2 - sigma ** 2)
        theta = mu+sigma*v
        f = T.dot(theta[:self.dimTheta/2], theta[self.dimTheta/2:])+u
        #these are for testing
        self.negKL = th.function([mu, sigma], negKL)
        self.f = th.function([theta, u], f)
        
        #the log-likelihood depends on f and lambda
        #the reason we can divide like this is because we assume p(x|f) isotropic
        logLike = T.sum(-(0.5 * np.log(2 * np.pi) + T.log(lambd)) - 0.5 * ((X - f) / lambd)**2)
        self.logLike = th.function([X, mu, sigma, lambd, u, v], logLike)

        logp = negKL + logLike

        gradvariables = [mu, sigma, lambd]

        derivatives = T.grad(logp,gradvariables)
        derivatives.append(logp)

        self.gradientfunction = th.function(gradvariables + [X, u, v], derivatives, on_unused_input='ignore')
        self.lowerboundfunction = th.function(gradvariables + [X, u, v], logp, on_unused_input='ignore')

    def iterate(self, data):
        '''''Main method, slices data in minibatches and performs an iteration'''''
        [N, dimX] = data.shape
        totalGradients = self.getGradients(data)
        self.updateParams(totalGradients,N,data.shape[0])

    def getGradients(self, batch):
        totalGradients = [0] * len(self.params)
        #in our case, we only use a single sample
        v = np.random.normal(0, 1,[self.dimTheta,1]) 
        u = np.random.normal(0, 1,[self.dimTheta/2,1])
        gradients = self.gradientfunction(*(self.params),X=batch,u=u,v=v)
        self.lowerbound += gradients[-1]
        for i in xrange(len(self.params)):
            totalGradients[i] += gradients[i]
        print totalGradients
        return totalGradients

    def updateParams(self, totalGradients,N,current_batch_size):
        """Update the parameters, taking into account AdaGrad and a prior"""
        for i in xrange(len(self.params)):
            self.h[i] += totalGradients[i]*totalGradients[i]
            if i < 5 or (i < 6 and len(self.params) == 12):
                prior = 0.5 * self.params[i]
            else:
                prior = 0

            self.params[i] += self.learning_rate/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(current_batch_size/N))
