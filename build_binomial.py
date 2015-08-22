import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse
from pylearn2.utils import sharedX

def bernoulli(p, u):
    return T.le(p,u)

def binomial(p, un, n):
    results, updates = th.scan(fn=bernoulli, outputs_info=None, sequences=un[0:n], non_sequences=p)
    binomial = results.sum()
    return binomial

p = T.dscalar('p')
un = T.dvector('un')
n = T.iscalar('n')

binomial = binomial(p, un, n)

f = th.function([p, un, n], binomial)

samples = []

uniform_samples = np.random.uniform(0, 1, 5)
f(0.01, uniform_samples, 5)

plt.plot(samples)
plt.show()

print f(0.1, np.asarray([0.8, 0.8, 0.8, 0.8]), 2)

