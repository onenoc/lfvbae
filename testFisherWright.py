import lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import math

if __name__=='__main__':
    n = 1
    bias = 0
    m = 5
    sigma_e = 0.1
    Lu = 1
    learning_rate = 0.01
    encoder = lfvbae.VA(n+bias, n+bias, m, 1, sigma_e, Lu,learning_rate=learning_rate)
    trajectory = encoder.start_fisher_wright(k=1)
    plt.plot(trajectory[:,0],label='x1')
    plt.plot(trajectory[:,1],label='x2')
    plt.plot(trajectory[:,2],label='x3')
    plt.legend()
    plt.show()
