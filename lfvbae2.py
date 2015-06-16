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
        self.learning_rate = learning_rate

    def initParams(self):
        muW = np.random.normal(2, 0.01, (self.dimTheta, 1))
        sigmaW = np.random.uniform(0, 0.01, (self.dimTheta, 1))
        lambd = np.matrix(np.random.uniform(0.95, 1.05))
        muR = np.matrix(np.random.normal(0, 1), (self.m, 1))
        sigmaR = np.matrix(np.random.norma(0, 1), (self.m, 1))
        self.params = [mu, sigma, lambd, alpha, beta]

    def createGradientFunctions(self):
        X = T.dmatrices("X")
        muW, sigmaW, u, v1, v2, f, muR, sigmaR = T.dcols("muW", "sigmaW", "u", "v1", "v2", "f", "muR", "sigmaR")
        lambd = T.patternbroadcast(T.dmatrix("lambd"),[1,1])

        negKLW = 0.5 * T.sum(1 + 2*T.log(abs(sigmaW)) - muW ** 2 - sigmaW ** 2)
        negKLR = 
        
        theta = muW+sigmaW*v1
