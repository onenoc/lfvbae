import numpy as np

dim_st = 1
dim_Obs = 1

'''
What do we need?
Need to fit a variational distribution (normal) to A
'''

def generate_observations(trajectory, C, Sigma, startState,T,u):
    '''
    @summary: generate observations
    @param C: linear transformation matrix for mean of observations
    @param Sigma: covariance matrix for observations
    @return: trajectory matrix, first column is observations, second column is states
    '''
    #initialize matrix of observations
    #observations = np.zeros(T)
    observations = []
    #for t in range(T):
        #observations[t] = np.dot(C,trajectory[t])+Sigma*u[t]
    #    observations.append(np.dot(C,trajectory[t])+Sigma*u[t])
    #return np.asarray(observations,dtype='f')
    return np.array([1.1, 2.1])


def generate_trajectory(A,Gamma,startState,T,u):
    '''
    @summary: generate trajectory
    @param A: linear transformation matrix for mean of transitions
    @param Gamma: covariance matrix for transitions
    '''
    #trajectory = np.zeros(T)
    #trajectory[0] = startState
    #for each time step in the chain
    trajectory = [startState]
    #for t in range(1,T):
    #    trajectory.append(np.dot(A,trajectory[t-1])+Gamma*u[t])
    #trajectory[t] = np.dot(A,trajectory[t-1])+Gamma*u[t]
    #trajectory = np.asarray(trajectory,dtype='f')
    #return trajectory
    return np.array([1.1, 2.1])

def log_likelihood_estimator(A,Gamma,C,Sigma,all_observations, trajectories, nSamples):
    '''
    approximates likelihood via simulation samples
    '''
    log_likelihood = 1
    trajectory=trajectories[0]
    T = len(trajectory)
    for i in range(nSamples):
        trajectory = trajectories[i]
        observations = all_observations[i]
        #we assume first state is fixed
        x = trajectory[0]
        y = observations[0]
        log_likelihood += np.log(norm_pdf(y,np.dot(C,x),Sigma))
        for t in range(1,T):
            #y is observation, x is latent variable
            y = observations[t]
            x = trajectory[t]
            x_prev = trajectory[t-1]
            print norm_pdf(y,np.dot(C,x),Sigma)
            print norm_pdf(x,np.dot(A,x_prev),Gamma)
            log_likelihood += np.log(norm_pdf(y,np.dot(C,x),Sigma))+np.log(norm_pdf(x,np.dot(A,x_prev),Gamma))
    return log_likelihood

def norm_pdf(x,mu,Sigma):
    return np.exp(-(x-mu)**2/(2*(Sigma**2)))/(Sigma*np.sqrt(2*np.pi))

def unit_test():
    #generate trajectory and observations from true value of A
    A = 1.2
    Gamma = 0.5
    C = 1
    Sigma = 0.5
    T = 10
    startState = 1
    u1 = np.random.normal(0,1,T)
    u2 = np.random.normal(0,1,T)
    trajectory = generate_trajectory(A,Gamma, startState,T,u1)
    observations = generate_observations(trajectory, C, Sigma, startState,T,u2)
    #generate trajectory from some other value of A
    trajectoryFalse = generate_trajectory(1,Gamma,startState,T,u1)
    #get log-likelihood for true trajectory
    all_observations = {}
    trajectories = {}
    all_observations[0] = observations
    trajectories[0] = trajectory
    nSamples = 1
    print 'log-likelihood true trajectory'
    print log_likelihood_estimator(A,Gamma,C,Sigma,all_observations, trajectories, nSamples)
    #get log-likelihood for other trajectory
    trajectories[0] = trajectoryFalse
    print 'log-likelihood false trajectory'
    print log_likelihood_estimator(A,Gamma,C,Sigma,all_observations, trajectories, nSamples)


if __name__=='__main__':
    A = 1.2
    Gamma = 0.1
    C = 1
    Sigma = 0.1
    T = 10
    startState = 1
    u1 = np.random.normal(0,1,T)
    u2 = np.random.normal(0,1,T)
    trajectory = generate_trajectory(A,Gamma, startState,T,u1)
    observations = generate_observations(trajectory, C, Sigma, startState,T,u2)
    print trajectory
    print observations
    all_observations = {}
    trajectories = {}
    all_observations[0] = observations
    trajectories[0] = trajectory
    nSamples = 1
    print log_likelihood_estimator(A,Gamma,C,Sigma,all_observations, trajectories, nSamples)
    unit_test()
