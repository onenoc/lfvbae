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
