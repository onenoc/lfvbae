import numpy as np
import matplotlib.pyplot as plt

def S(alpha,beta):
    inner = 1+(beta**2)*(np.tan(np.pi*alpha/2)**2)
    S = inner**(1/(2*alpha))
    return S

def B(alpha, beta):
    return (1/alpha)*np.arctan(beta*np.tan(np.pi*alpha/2))

def fy(alpha,beta,w,u):
    S_var = S(alpha,beta)
    B_var = B(alpha,beta)
    first = np.sin(alpha*(u+B_var)/(np.cos(u)**(1/alpha)))
    second = np.cos(u-alpha*(u+B_var))/w
    return S_var*first*(second**((1-alpha)/alpha))

if __name__ == '__main__':
    samples = []
    #alpha = 0.5
    alpha = 1.9
    beta = 1.59
    for i in range(500):
        w = np.random.exponential()
        u = np.random.uniform(-np.pi/2,np.pi/2)
        samples.append(fy(alpha,beta,w,u))
    #print samples    
    plt.hist(samples,100)
    plt.show()
