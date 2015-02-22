__author__ = 'darshanhegde'
"""
k-max pooling example.
"""

import numpy as np

import theano
from theano import tensor as T
from theano.sandbox import neighbours

k = 3
# instantiate 4D tensor for input
input = T.tensor4(name='input')

neighborsForPooling = neighbours.images2neibs(input, (1, 5), mode='valid')
neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
kNeighborsArg = neighborsArgSorted[:, -k:]
kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)
ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
jj = kNeighborsArgSorted.flatten()
k_pooled_2D = neighborsForPooling[ii, jj].reshape((3, k))
k_pooled = neighbours.neibs2images(k_pooled_2D, (1, 3), (1, 3, 1, 3))

k_max = theano.function([input], k_pooled)

input = np.array([[2, 4, 1, 6, 8], [12, 3, 5, 7, 1], [-8, 6, -12, 4, 1]], dtype=np.float32)
input = input.reshape(1, 3, 1, 5)
print "input shape: ", input.shape
print "input: ", input
output = k_max(input)
print "output shape: ", output.shape
print "output : ", output
