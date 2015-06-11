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
        
    def createGradientFunctions(self):
        #create
        X = T.dmatrices("X")
        mu, sigma, u, v, f, lambd = T.dcols("mu", "sigma", "u", "v", "f", "lambd")
        
        negKL = 0.5 * T.sum(1 + 2*T.log(sigma) - mu ** 2 - sigma ** 2)
        theta = mu+sigma*v
        self.negKL = th.function([mu, sigma], negKL)
        W=theta
        #R=theta[-1]
        #eps=R*u
        y=X[:-1,0]
        X_sim=X[:,1:]
        f = T.dot(W.T,X_sim)+u
        #these are for testing
        
        self.f = th.function([X, theta, u], f)
        
        #the log-likelihood depends on f and lambda
        #the reason we can divide like this is because we assume p(x|f) isotropic
        
        logLike = T.sum(-(0.5 * np.log(2 * np.pi) + T.log(lambd)) - 0.5 * ((y - f) / lambd)**2)

        logp = negKL + logLike

        gradvariables = [mu, sigma, lambd]
        
        self.logLike = th.function(gradvariables + [X, u, v], logLike)

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
        self.lowerbound += gradients[-1] #not sure about this line
        print "params"
        print self.params
        #print self.logLike(*(self.params),X=batch,u=u,v=v)
        for i in xrange(len(self.params)):
            totalGradients[i] += gradients[i]
        return totalGradients

    def updateParams(self, totalGradients,N,current_batch_size):
        for i in xrange(len(self.params)):
            self.params[i] += self.learning_rate * totalGradients[i]
'''
    def getLowerBound(self, data):
        lowerbound = 0
        v = np.random.normal(0, 1,[self.dimTheta,1])
        u = np.random.normal(0, 1,[self.dimTheta/2,1])
        return lowerbound
'''
