"""
A convolutional neural network using theano.  This script was created by
Nicholas Zufelt as a part of the London Machine Learning Practice meetup.

This is NOT YET IMPLEMENTED, as doing so is one of the exercises for
tonight's meetup.  In order to make things run without error, I put sections
into triple quotes.  Delete the triple quotes for each section as you get to
it, so that you can keep checking your work.

Calling this script with an example (may not converge):
$ python cnn_nyi.py 5 5 3 2 25 100 20 1000 128 .01 .01

Parameters:
    filter_height -- int, convolution layer filter size (with width)
    filter_width -- int
    n_filters -- int, number of copies of the filter
    pool_size -- int, size of pooling window.  This gets turned into a tuple
                     of two ints, to make a square pool
    n_hidden -- int, number of nodes in the hidden (fully-connected) layer
    epochs -- int, number of training epochs (iterations) through the whole data
    print_every -- int, on which epoch should the current cost print
    n_samples -- int, number of training samples.  Currently, this is reduced
                     to be a multiple of the batch size, because conv2d
                     seems to require all batches are the same size.  This
                     reduces the training list to help speed along the process.
                     In practice, you would want to use the whole training set.
    batch -- int, size of batch for SGD
    reg -- float, regularization strength
    alpha -- float, learning rate

Givens:
    image_height, image_width -- ints, input image size
    n_inputs -- number of nodes in the input layer of the FULLY-CONNECTED
                    LAYER.  Note that this is determined by the other
                    hyperparameters.
    n_outputs -- classes of digits
    n_channels -- since MNIST is greyscale, this is 1.  Another common option
                      is 3 for RGB pictures
"""
import sys
import numpy as np
import numpy.random as rng
import pandas as pd
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import relu,conv2d

# Read in parameters
(filter_height,filter_width,n_filters,
    pool_size,n_hidden,epochs,print_every,
    n_samples,batch) = (int(i) for i in sys.argv[1:10])
(reg,alpha) = (float(i) for i in sys.argv[10:])
pool_size = (pool_size,pool_size)
n_samples = n_samples - (n_samples % batch)
# TODO: I'm going to save you some annoyance and give you the above line.
#       Essentially, conv2d() wants all your batches to be the same size, so
#       this ensures you don't have a tiny last batch in the epoch. After
#       you've got everything else working, try to implement conv2d without
#       the above line.  On your first time through, ignore this todo.

# Givens
image_height, image_width = 28, 28
n_inputs = None # TODO: Figure out what expression should go here.  It is
                #       determined by the output of the convolution layer.
n_outputs = 10
n_channels = 1

# Input and training values
#   TODO: Initialize your training inputs (X and y) using theano tensors.
#         X will be a 4-tensor, and y will be a matrix (of one-hot vectors)

# Convolution layer
#    TODO: Here, you implement the convolution layer.  I've left the function
#          calls in with None in place of the parameters so that you know what
#          to look for in the theano docs.  Important note: in December, theano
#          refactored the old convolution layer implementation, it was in:
#              theano.tensor.nnet.conv.conv2d
#          which is not what we're using.  I just make note of that here if you
#          used theano prior to December 2015.
"""
noise_conv = None # this is a small noise parameter to scale your rng
W_shape = (None,None,None,None) # 4-tuple of ints, to plug into next line
W_conv = theano.shared(None) # initialize with small, random weights
conv_out_layer = conv2d(None)
"""
# Pooling layer
"""
pooled_out = pool_2d(None)
"""
# Implement the bias term and nonlinearity
"""
b_conv = theano.shared(None) # a 1-d vector which determines if each filter fires
conv_out = relu(None) # you'll need to use b_conv.dimshuffle() to make sure
                      #     that b_conv gets broadcasted to the relevant dims
conv_out_flat = None # Need to flatten at least one pair of dimensions to feed
                     #     into the fully-connected layer
"""

# Fully-connected layer
#    TODO: Here, you implement the fully-connected layer.  This time I've
#          removed the function calls, so you can try to implement this section
#          from scratch.  You'll need a full MLP here; that is, there is a
#          hidden layer and a round of weights and biases before and after the
#          hidden layer.

# The shared variables I used are W1_full, b1_full, W2_full, b2_full, and the
#     final theano expression is output:

"""
output = T.nnet.softmax(None)
prediction = np.argmax(output,axis=1)
crossent = T.nnet.categorical_crossentropy(output,y).mean()
"""
# I had better luck with the mean.  Do you think I should use something
#     else here?  Experiment!
"""
cost = crossent + None # add the regularization term
"""
# gradients and update statements.
# TODO: params is a list of all your parameters, and grads is a tuple of
#       their gradients (that is, the gradient of cost with respect
#       to each one)
"""
params = [None]
grads = T.grad(None)
updates = [(param,param - alpha * grad) for param, grad in zip(params,grads)]
"""

# build theano functions
"""
epoch = theano.function(inputs = [X,y],
                        outputs = [],
                        updates = updates)
predict = theano.function(inputs=[X],outputs=prediction)
compute_cost = theano.function(inputs = [X,y],
                               outputs = [cost,crossent.sum()])
"""

# Read in MNIST data.
# TODO: Implement reading in the data.  I highly recommend using n_samples
#       to limit the number of examples you read in, as otherwise your
#       training epochs will take a long time. Note that the first row
#       of train.csv is a collection of column names, and the first column
#       is the value of each row (the response digit).  I used pandas here,
#       but that's overkill.  If you're working over AWS on the free tier,
#       beware your memory usage!  Also, don't forget to one-hot your y_data
#       and reshape your X_data into the correct 4-tensor


# Preprocessing
# TODO: Give the data mean 0 and scale the range -1 to 1.  I found that to
#       be sufficient


# Train the model
# TODO: When I say epochs, I mean iterating through the entire dataset once.
#       This way, if you change the size of your batches, you still get about
#       the same amount of training (smaller batch = more, noisier iterations)
for i in range(epochs):
    # TODO: iterate through the dataset by batch, applying epoch() to each batch
    if i % print_every == 0:
        # TODO: Compute cost and crossent.  I recommend batching this because
        #       I think conv2d requires it (maybe there's a way around this?),
        #       but also to not explode memory
        current_cost, current_crossent = 0, 0
        sentence = 'Cost after epoch {} is {}, with crossentropy {}'
        print(sentence.format(i, current_cost, current_crossent))

# Compute accuracy.
# TODO: Check some kind of accuracy on the final model.  If you read in and
#       used only a portion of the dataset, you could compute validation
#       accuracy by reading in a validation set above, when you read in the
#       data.  Alternatively, you can just check your accuracy on your dataset,
#       which in practice you would avoid because it tends to overestimate test
#       accuracy.
wrong,score = 0,0
sentence = 'Our model made {} errors, for a training accuracy of {}'
print(sentence.format(wrong,score))
