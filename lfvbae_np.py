import numpy as np

class VA:
    def __init__(self, dimX, dimTheta, m, n, sigma_e, Lu=1, learning_rate=0.01):
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
        mu = np.random.normal(0, 10, (self.dimTheta, 1))
        logSigma = np.random.uniform(0, 1, (self.dimTheta, 1))
        logLambda = np.random.uniform(0, 10)
        self.params = [mu,logSigma]

    def lowerBoundFunction(self, X,y,u,v):
        mu = self.params[0]
        logSigma = self.params[1]
        logLambda = np.log(self.sigma_e)

        negKL = 0.5*self.dimTheta+0.5*np.sum(2*logSigma - mu ** 2 - np.exp(logSigma) ** 2)
        f = self.regression_simulator(X,u,v,mu,logSigma)
        logLike = -self.m*(0.5 * np.log(2 * np.pi) + logLambda)-0.5*np.sum((y-f)**2)/(np.exp(logLambda)**2)

        elbo = (negKL + logLike)
        obj = -elbo
        return obj

    def regression_simulator(self,X,u,v,mu,logSigma):
        theta = mu+np.exp(logSigma)*v
        predval = np.reshape(np.dot(X,theta),(self.m,1))
        return predval


