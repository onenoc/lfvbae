# first line: 124
@memory.cache
def data():
    A = 1.2
    Gamma = 0.5
    C = 1
    Sigma = 0.1
    T = 100
    startState = 1
    u1 = np.random.normal(0,1,T)
    u2 = np.random.normal(0,1,T)
    trajectory = generate_trajectory(A,Gamma, startState,T,u1)
    observations = generate_observations(trajectory,C,Sigma,startState,T,u2)
    return summary_statistics(observations)
