import numpy as np
import theano as th
import theano.tensor as T
from pylearn2.utils import sharedX
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent

class VA:
    def __init__(self, dimX, dimTheta, m, n, L=1):
        self.dimX = dimX
        self.dimTheta = dimTheta
        self.m = m
        self.n = n
        self.L = L
        self.iterations = 0
        self.lowerBounds = []

    def createGradientFunctions(self):
        X = T.dmatrix("X")
        y = T.vector("y")
        a = T.scalar("a")
        self.theta = sharedX(np.zeros(self.n),name='theta')
        b = sharedX(5,name='b')
        obj = T.sum(y-T.dot(X,self.theta))**2
        self.minimizer = BatchGradientDescent(objective = obj,params = [self.theta],inputs = [X,y])
       
    def iterate(self,batch):
        X = batch[:,1:]
        y = batch[:,0]
        print self.minimizer.minimize(X,y)
        print self.theta.get_value()
