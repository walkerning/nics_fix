# Fixed Point Training Simulation Framework

This is a fixed-point training simulation framework based on tensorflow.

Design
------------
We have four things to be quantitized potentially:

* weights
* gradients of weights
* activations
* gradients of activations

For transparent conversion to fixed point simulation, we supply a context manager, the weights and gradients of weights in the models created in this context manager will be handled transparently. The activations and the gradients of activations should be handled manually by insert `quantitize` operation into necessary places.