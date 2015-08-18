import numpy as np

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
        mu = np.random.normal(0, 10, (self.dimTheta, 1))
        logSigma = np.random.uniform(0, 1, (self.dimTheta, 1))
        logLambda = np.random.uniform(0, 10)
        self.params = [mu,logSigma]

    def objectiveFunction(X,y,u,v):
        
