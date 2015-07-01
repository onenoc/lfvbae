import numpy as np
import scipy as sp
import pylab as pp

import pdb
from blowfly import *

problem_params = default_params()
problem_params["epsilon"] = 2
problem_params["blowfly_filename"] = "blowfly-data.txt"
problem = BlowflyProblem( problem_params, force_init = True )

nbr_samples = 10

y = problem.observations
sy = problem.statistics_function(y)

X = []
S = []
for t in range(nbr_samples):
  theta = problem.theta_prior_rand()
  x = problem.simulation_function(theta)
  s = problem.statistics_function(x)
  
  X.append(x)
  S.append(s)
  
X = np.array(X).T
S = np.array(S)

pp.figure()
pp.plot( y, 'k-', lw=4)
pp.plot( X, '-', lw=2)
pp.ylim(0,10000)
pp.show()