README for lmlp-theano
----------------------
This repo contains several files of interest:

* This `README` which contains the 8 exercises for our 7th April 2016 meetup on theano and CNNs.
* `ffnn.py` serves as an introduction to theano using a feedforward (*i.e.* traditional) neural network (exercises 1-4).
* `cnn_nyi.py` contains the sketched-out architecture of a CNN (via comments), which you could choose to implement (exercise 7).
* `solutions/cnn.py` is an implemented CNN which you could use for exercises 5, 6, or 8.
* `solutions/parameters.txt` contains some parameters that I found allowed the CNN to converge.

Exercises
---------
1. Tune hyperparameters to get `ffnn.py` to converge
2. Extend `ffnn.py` to classify MNIST digits: first extend to multiclassification using one-hot vectors for y, implement SGD, then read in `train.csv` and tune hyperparameters for convergence
3. (advanced) Implement _Dropout_: during each training step, pick some subset of random nodes to drop, and scale the rest of the weights accordingly
4. (advanced) Make the network `ffnn.py` deeper and faster: change to ReLU and add an extra hidden layer
5. **Tune hyperparameters to get `cnn.py` to converge**
6. **(advanced) Add more convolutional layers, either in line with the first one for a "deeper" first layer, or after the first one, for a "deeper" overall network**
7. **(advanced) Make the network `ffnn.py` deeper and faster: change to ReLU and add an extra hidden layer*(advanced) Implement the CNN from my blocked-out code `cnn_nyi.py`**
8. **(advanced) Change our CNN to be used for NLP**

If you're running AWS, here's some details you'll need to get jupyter notebooks running:  from the home directory (`cd ~`), type

`jupyter notebook --certfile=~/certs/mycert.pem --keyfile ~/certs/mycert.key`

then you can go to your webbrowser and navigate to

http://<your-ec2-instance-public-ip-address>:8888/

and enter the password: lmlp

Your EC2 public ip address is located on your EC2 console page.
