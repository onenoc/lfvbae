import lfvbaeFisherWright as lfvbae
import theano as th
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import math
import sys
import alpha_stable.fy as alphaStable

if __name__=='__main__':
    alpha = 0.5
    beta = 0
    

    m = 20
    n = 1
    learning_rate = 0.0000001
    dimTheta = 4
    encoder = lfvbae.VA(dimTheta, m, n, learning_rate)
    encoder.initParams()
    encoder.createObjectiveFunction()

    W = np.random.exponential(size=m)
    U = np.random.uniform(-np.pi/2,np.pi/2,m)
    encoder.lowerboundfunction()
