"""
Implement a feedforward neural network using theano.

Calling this script with an example:
$ python ff_nn noclass.py 3 6 .01 .01 10000 1000 61
"""
import numpy as np
import sys
import theano
import theano.tensor as T

# Receive inputs from user
n_inputs,n_hidden = (int(i) for i in sys.argv[1:3])
n_outputs = 1                                       # binary classification ! 
reg,alpha = (float(i) for i in sys.argv[3:5])
epochs,print_every = (int(i) for i in sys.argv[5:7])
n_samples = int(sys.argv[7])

# Need to initialize the parameters to a small, random number
n = 1/(np.sqrt(n_inputs * n_outputs * n_hidden))

# Weights and biases
W1 = theano.shared(n*np.random.randn(n_inputs,n_hidden), name='W1')
W2 = theano.shared(n*np.random.randn(n_hidden,n_outputs), name='W2')
b1 = theano.shared(np.zeros(n_hidden), name='b1')
b2 = theano.shared(np.zeros(n_outputs), name='b2')

x = T.dmatrix('x')
y = T.dvector('y')

# forward prop
z1 = x.dot(W1)+b1
hidden = T.tanh(z1)
z2 = hidden.dot(W2) + b2
output = T.nnet.sigmoid(z2)
prediction = output > 0.5

# cost
crossent = -y.dot(T.log(output)) - (1-y).dot(T.log(1-output))
cost = crossent.sum() + reg * ((W1**2).sum()+(W2**2).sum())

# gradients
gW1,gb1,gW2,gb2 = T.grad(cost,[W1,b1,W2,b2])

# build theano functions
epoch = theano.function(inputs = [x,y],
                        outputs = [output, crossent.sum()],
                        updates = ((W1,W1-alpha*gW1),
                                   (b1,b1-alpha*gb1),
                                   (W2,W2-alpha*gW2),
                                   (b2,b2-alpha*gb2)))
predict = theano.function(inputs=[x],outputs=prediction)

# generate toy data
N = n_samples
D = (np.random.randn(N,n_inputs),np.random.randint(size=N,low=0,high=2)) 

# train the model
for i in range(epochs):
    pred,err = epoch(D[0],D[1])
    if i % print_every == 0:
        print('Error after epoch {}: {}'.format(i,err))

# check accuracy
preds = predict(D[0]).T[0]
wrong = (preds != D[1]).sum()
score = (N*1.0 - wrong)/N
print("Our model made {} errors, for an accuracy of {}".format(wrong, score))







