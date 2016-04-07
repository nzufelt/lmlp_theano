"""
A convolutional neural network using theano.

This script was created by Nicholas Zufelt as a part of the London Machine Learning
Practice meetup.

Calling this script with an example (may not converge):
$ python cnn.py 5 5 3 2 25 100 20 1000 128 .01 .01

Parameters:
    filter_height -- int, convolution layer filter size (with width)
    filter_width -- int
    n_filters -- int, number of copies of the filter
    pool_size -- int, size of pooling window.  This gets turned into a tuple
                     of two ints, to make a square pool
    n_hidden -- int, number of nodes in the hidden (fully-connected) layer
    epochs -- int, number of training epochs
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

# Givens
image_height, image_width = 28, 28
n_inputs = n_filters*((image_height - filter_height + 1) // pool_size[0])**2
# Convince yourself that the above is correct! (or check that it is :-)
n_outputs = 10
n_channels = 1

# Input and training values
X = T.tensor4(name='X')
X_shape = (batch,n_channels,image_height,image_width)
y = T.dmatrix('y') # one-hot vectors

# Convolution layer
noise_conv = 2/(np.sqrt(n_filters*filter_height*filter_width))
W_shape = (n_filters,n_channels,filter_height,filter_width)
W_conv = theano.shared(noise_conv*rng.randn(*W_shape),
                       name='W_conv')
conv_out_layer = conv2d(X,
                        W_conv,
                        input_shape=X_shape,
                        filter_shape=W_shape,
                        border_mode='valid')
# NOTE:
# output_shape = (batch, 1, output_rows, output_columns)

# Pooling layer
pooled_out = pool_2d(input=conv_out_layer,
                     ds=pool_size,
                     ignore_border=True)
# NOTE:
# ignore_border ==> round down if convolution_output / pool_size is not int

# Implement the bias term and nonlinearity
b_conv = theano.shared(np.zeros(n_filters,), name='b_conv')
conv_out = relu(pooled_out + b_conv.dimshuffle('x',0,'x','x'))
conv_out_flat = conv_out.flatten(2)

# Fully-connected layers
noise_full = 1/(np.sqrt(n_inputs))
W1_full = theano.shared(noise_full*rng.randn(n_inputs,n_hidden), name='W1_full')
b1_full = theano.shared(np.zeros(n_hidden), name='b1_full')
W2_full = theano.shared(noise_full*rng.randn(n_hidden,n_outputs), name='W2_full')
b2_full = theano.shared(np.zeros(n_outputs), name='b2_full')

z1 = conv_out_flat.dot(W1_full) + b1_full
hidden = relu(z1)
z2 = hidden.dot(W2_full) + b2_full
output = T.nnet.softmax(z2)
prediction = np.argmax(output,axis=1)
crossent = T.nnet.categorical_crossentropy(output,y)/n_samples

cost = crossent.sum() + reg*((W_conv**2).sum()+(W1_full**2).sum()+(W2_full**2).sum())

# gradients and update statements
params = [W_conv,b_conv,W1_full,b1_full,W2_full,b2_full]
grads = T.grad(cost,[*params])
updates = [(param,param - alpha * grad) for param, grad in zip(params,grads)]

# build theano functions
epoch = theano.function(inputs = [X,y],
                        outputs = [],
                        updates = updates)
predict = theano.function(inputs=[X],outputs=prediction)
compute_cost = theano.function(inputs = [X,y],
                               outputs = [cost,crossent.sum()])

# Read in MNIST data
train_df = pd.read_csv('train.csv')[0:n_samples]
X_data = train_df.values
del train_df # free up some memory
I = np.identity(10)
y_data = np.array([I[i] for i in X_data.T[0].T]) # one-hot the y's
# strip off response variable and make into a 4-tensor:
X_data = np.reshape(X_data.T[1:].T,(n_samples,n_channels,
                                    image_height,image_width))

# Give the data mean 0 and range -1 to 1:
X_data = X_data.astype(float)
X_data -= 128.0
X_data /= 128.0

# Train the model
for i in range(epochs):
    for ind in range(0,n_samples,batch):
        rows = list(range(ind,min(ind+batch,n_samples)))
        epoch(X_data[rows],y_data[rows])
    if i % print_every == 0:
        # Compute cost and crossent.  We batch this to not explode memory
        current_cost = 0
        current_crossent = 0
        for ind in range(0,n_samples,batch):
            rows = list(range(ind,min(ind+batch,n_samples+1)))
            new_cost,new_crossent = compute_cost(X_data[rows],y_data[rows])
            current_cost += new_cost
            current_crossent += new_crossent
        sentence = 'Cost after epoch {} is {}, with crossentropy {}'
        print(sentence.format(i, current_cost, current_crossent))

# Compute accuracy.  We batch this to not explode memory
wrong = 0
I = np.identity(n_outputs)
for ind in range(0,n_samples,batch):
    rows = list(range(ind,min(ind+batch,n_samples+1)))
    preds = np.array([I[i] for i in predict(X_data[rows])])
    wrong += (preds != y_data[rows]).sum() / 2
score = (n_samples*1.0 - wrong)/n_samples
sentence = 'Our model made {} errors, for a training accuracy of {}'
print(sentence.format(wrong,score))
