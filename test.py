import lfvbae
import theano as th

#make univariate data, univariate param
encoder = lfvbae.VA(1, 2, 1, 10, 1)

encoder.createGradientFunctions()

#negKL between standard normal and standard normal should be 0
print "this should be 0"
print encoder.negKL([[0]], [[1]])

#this should be 5
print "this should be 5"
print encoder.f([[2], [2]], [[1]])

#[X, mu, sigma, lambd, u, v]
#should be approximately -2.92
print encoder.logLike([[0, 0]], [[0], [0]], [[1], [1]], [[1]], [[1]], [[1], [1]])
