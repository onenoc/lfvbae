import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse
from pylearn2.utils import sharedX

def bernoulli(p, u):
    return T.le(u,p)

def binomial(p, un, n):
    results, updates = th.scan(fn=bernoulli, outputs_info=None, sequences=un[0:n], non_sequences=p)
    binomial = results.sum()
    return binomial

p = T.dscalar('p')
un = T.dvector('un')
n = T.iscalar('n')

binomial = binomial(p, un, n)

f = th.function([p, un, n], binomial)

def row(u, prev):
    return u
    

U = T.dmatrix("U")
outputs_info = T.as_tensor_variable(np.zeros(5),U.dtype)
results, updates = th.scan(fn=add_row, ]

#print f(0.5, np.asarray([0.8, 0.8, 0.8, 0.8]), 2)

