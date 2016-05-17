import theano as th
import theano.tensor as T
import numpy
import theano
from theano import tensor

def add_next(x,k):
    return x+k

xStart = T.dscalar('xStart')
i = T.iscalar('i')
k = T.iscalar('k')

results, updates = th.scan(fn=add_next, outputs_info=[{'initial':xStart,'taps':[-1]}],non_sequences=k, n_steps=i)

total = T.sum(results**2)

derivative = T.grad(T.sum(results**2),k)
f = th.function([xStart,i,k],outputs=results,updates=updates)
df_dk = th.function([xStart,i,k],outputs=derivative,updates=updates)
print f(2,3,2)
print df_dk(2,3,1)

