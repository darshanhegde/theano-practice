__author__ = 'darshanhegde'
"""
1-d convolution layer example.
"""

import numpy as np

import theano
from theano import tensor as T
from theano.tensor.nnet import conv

import numpy

rng = numpy.random.RandomState(23455)

# instantiate 4D tensor for input
input = T.tensor4(name='input')

# initialize shared variable for weights.
W = theano.shared(np.array([[[0.3]*4, [0.2]*4, [0.1]*4], [[0.1]*4, [0.1]*4, [0.1]*4], [[0.2]*4, [0.2]*4, [0.2]*4]],
                           dtype=input.dtype).reshape(3, 3, 1, 4), name='W')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv.conv2d(input, W, border_mode='full')

# create theano function to compute filtered images
conv_1d = theano.function([input], conv_out)

input = np.array([np.arange(10), 2*np.arange(10), 3*np.arange(10)], dtype=np.float32)
output = conv_1d(input.reshape(1, 3, 1, 10))
print output.shape
print output