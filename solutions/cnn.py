"""
Implement a convolutional neural network using theano.  This script was
created by Nicholas Zufelt as a part of the London Machine Learning
Practice meetup.
"""
import sys
import numpy as np
import numpy.random as rng
import pandas as pd
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import relu,conv2d

#### Parameters
# For the whole network
reg,alpha = .01,.2 #(float(i) for i in sys.argv[4:6])
minibatch = 128 #(int(i) for i in sys.argv[1:4])
n_samples = 2000 - (2000 % minibatch) # make this better!
n_channels = 1 # greyscale
epochs,print_every = 200,20 #(int(i) for i in sys.argv[6:8])
# For the convolutional layer
image_height,image_width = 28,28
filter_height,filter_width = 5,5 
pool_size,n_filters = (2,2),3 # n_filters is the number of copies of the filter
# For the Fully-connected layer
n_inputs = n_filters*((image_height - filter_height + 1) // pool_size[0])**2
# figure out why above is correct! Note that we assume square everything
# for MNIST, with 5x5 filter and 2x2 pooling, this gives 13x13=169
n_hidden,n_outputs = 90,10

# TODO: change the name of this
n_conv = 2/(np.sqrt(n_filters*filter_height*filter_width))

# Input and training values
X = T.tensor4(name='X')
X_shape = (minibatch,n_channels,image_height,image_width)
y = T.dmatrix('y') # one-hot vectors

# Convolution layer
W_shape = (n_filters,n_channels,filter_height,filter_width)
W_conv = theano.shared(n_conv*rng.randn(*W_shape),
                       name='W_conv')
conv_out_layer = conv2d(X, 
                        W_conv,
                        input_shape=X_shape,
                        filter_shape=W_shape,
                        border_mode='valid')
# NOTE:
# output_shape = (minibatch, 1, output_rows, output_columns)

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
n_full = 1/(np.sqrt(n_inputs))
W1_full = theano.shared(n_full*rng.randn(n_inputs,n_hidden), name='W1_full')
b1_full = theano.shared(np.zeros(n_hidden), name='b1_full')
W2_full = theano.shared(n_full*rng.randn(n_hidden,n_outputs), name='W2_full')
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
X_data = np.reshape(X_data.T[1:].T,(n_samples,n_channels,image_height,image_width))  
 
# Give the data mean 0 and range -1 to 1:
X_data = X_data.astype(float)
X_data -= 128.0
X_data /= 128.0

# train the model
for i in range(epochs):
    for ind in range(0,n_samples,minibatch):
        rows = list(range(ind,min(ind+minibatch,n_samples)))
        epoch(X_data[rows],y_data[rows])
    if i % print_every == 0:
        current_cost = 0
        current_crossent = 0
        for ind in range(0,n_samples,minibatch):
            rows = list(range(ind,min(ind+minibatch,n_samples+1)))
            new_cost,new_crossent = compute_cost(X_data[rows],y_data[rows]) 
            current_cost += new_cost
            current_crossent += new_crossent
        print('Cost after epoch {} is {}, with crossentropy {}'.format(i,
                                                                    current_cost,
                                                                    current_crossent))

# Accuracy testing
wrong = 0
I = np.identity(n_outputs)
for ind in range(0,n_samples,minibatch):
    rows = list(range(ind,min(ind+minibatch,n_samples+1)))
    preds = np.array([I[i] for i in predict(X_data[rows])])
    wrong += (preds != y_data[rows]).sum() / 2  
score = (n_samples*1.0 - wrong)/n_samples
print("Our model made {} errors, for an accuracy of {}".format(wrong, score))
