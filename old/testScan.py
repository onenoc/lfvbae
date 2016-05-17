import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX
import matplotlib.pyplot as plt

def return_row(u,n):
    return u[0:n]

U=T.dmatrix('U')
n = T.iscalar('i')

results, updates = th.scan(fn=return_row, sequences=U,non_sequences=n)
f = th.function([U,n],outputs=results, updates=updates)

Ur = np.random.uniform(0,1,(20, 2))

print f(Ur, 1)

