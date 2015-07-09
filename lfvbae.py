import numpy as np
import theano as th
import theano.tensor as T
from pylearn2.utils import sharedX
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent

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
        self.sigmas = []
        self.learning_rate = learning_rate
        print "learning rate"
        print self.learning_rate

    def initParams(self):
        '''
        @description: parameters to learn
        '''
        mu = sharedX(np.random.normal(0, 10, (self.dimTheta, 1)), name='mu')
        logSigma = sharedX(np.random.uniform(0, 1, (self.dimTheta, 1)), name='logSigma')
        logLambda = sharedX(np.random.uniform(0, 10), name='logLambda')
        self.params = [mu,logSigma]

    def createObjectiveFunction(self):
        '''
        @description: initialize objective function and minimization function
        @X,y data matrix/vector
        @u random noise for simulator
        @v standard normal for reparametrization trick
        '''
        X,u = T.dmatrices("X","u")
        f, y, v = T.dcols("f", "y", "v")
        
        mu = self.params[0]
        logSigma = self.params[1]
        logLambda = sharedX(np.log(self.sigma_e),name='logLambda')
        #logLambda = self.params[2]

        negKL = 0.5*self.dimTheta+0.5*T.sum(2*logSigma - mu ** 2 - T.exp(logSigma) ** 2)
        f = T.dot(X,mu + T.exp(logSigma)*v)
        #self.regression_simulator(X,u,v,mu,logSigma)

        logLike = -self.m*(0.5 * np.log(2 * np.pi) + logLambda)-0.5*T.sum((y-f)**2)/(T.exp(logLambda)**2)/self.Lu

        #logLike = -T.sum((y-T.dot(X,mu + T.exp(logSigma)*v))**2)

        elbo = (negKL + logLike)
        obj = -elbo
        self.lowerboundfunction = th.function([X, y, u, v], obj, on_unused_input='ignore')
        derivatives = T.grad(obj,self.params)
        self.gradientfunction = th.function([X,y,u,v], derivatives, on_unused_input='ignore')
        self.minimizer = BatchGradientDescent(objective = obj,params = self.params,inputs = [X,y,u,v],max_iter=1,conjugate=1)

    def changeParamsAndCalcCost(self, batch, mu='same', sigma='same'):
        if mu!='same':
            mu = sharedX(mu, name='mu')
            self.params[0] = mu
        if sigma!='same':
            logSigma = sharedX(np.log(sigma), name='logSigma')
            self.params[1] = logSigma
        self.createObjectiveFunction()
        X = batch[:,1:]
        y = np.matrix(batch[:,0]).T
        u = np.random.normal(0, self.sigma_e,(self.m,1))
        np.random.seed(seed=10)
        v = np.random.normal(0, 1,(self.dimTheta,1))
        #np.random.seed(seed=50)
        ret_val = self.lowerboundfunction(X,y,u,v)
        np.random.seed()
        return ret_val
        
    def regression_simulator(self,X,u,v,mu,logSigma):
        theta = mu + logSigma*v
        predval = T.dot(X,theta)
        predval = T.addbroadcast(predval,1)
        #+u
        return predval

    def fisher_wright(x0, x1, x2, k):
        N = sharedX(2000,name='N')
        p1 = sharedX(0.1,name='N')
        p0 = 1/(1+k*x2/N)
           
    def iterate(self,batch):
        X = batch[:,1:]
        y = np.matrix(batch[:,0]).T
        v = np.random.normal(0, 1,(self.dimTheta,1))
        u = np.random.normal(0, self.sigma_e,(self.m,self.Lu))
        #cost = self.minimizer.minimize(X,y,u,v)
        cost = self.lowerboundfunction(X=X,y=y,u=u,v=v)
        old_params = [i.get_value() for i in self.params]
        gradients = self.getGradients(batch)
        self.updateParams(gradients)
        new_params = [i.get_value() for i in self.params]
        change = []
        for i in range(len(self.params)):
            change.append((new_params[i]-old_params[i])/old_params[i])
        if abs(max(change)) < 0.0000000005:
            self.converge=1
        #keep track of min cost and its parameters
        self.lowerBounds.append(cost)
        '''
        if self.iterations > 2 and abs((self.lowerBounds[-1]-self.lowerBounds[-2])/self.lowerBounds[-2]) < 0.001:
            self.converge = 1
        '''
        '''
        if self.iterations % 300 == 0:
            print change
            self.print_parameters()
        '''
        '''
        self.sigmas.append(np.exp(self.params[1].get_value()))
        '''
        self.iterations += 1

    def print_parameters(self):
        print "\n"
        #print "cost"
        #print self.lowerBounds[-1]
        print "mu"
        print self.params[0].get_value()
        print "sigma"
        print np.exp(self.params[1].get_value())
        #print "lambda"
        #print np.exp(self.params[2].get_value())

    def getGradients(self,batch):
        """Compute the gradients for one minibatch and check if these do not contain NaNs"""
        X = batch[:,1:]
        y = np.matrix(batch[:,0]).T
        v = np.random.normal(0, 1,(self.dimTheta,1))
        u = np.random.normal(0, self.sigma_e,(self.m,self.Lu))
        gradients = self.gradientfunction(X=X,y=y,u=u,v=v)
        return gradients

    def updateParams(self,totalGradients):
        """Update the parameters, taking into account AdaGrad and a prior"""
        for i in xrange(len(self.params)):
            self.params[i].set_value(self.params[i].get_value()-self.learning_rate*totalGradients[i])
