# Fixed Point Training Simulation Framework

This is a fixed-point training simulation framework based on tensorflow.

Design
------------

### Basics

We have four things to be quantitized potentially:

* weights
* gradients of weights
* activations
* gradients of activations

For transparent conversion to fixed point simulation, we supply a context manager for managing fixed point configuration: `nics_scope.context.fixed_scope`, and wrapper of nn operations. Eg. `Dense` for `tf.layers.dense`, `Conv2d` for tf.layers.conv2d.

Using these wrappers inside the context manager, the above four things in the models created will be handled transparently. You can also manually insert `.apply(quantitize)` operation into necessary places.

### Strategy

**INTERFACE NOT IMPLEMENTED**

Examples
------------
See `examples/mlp.py` for a simple example of a 1-hidden-layer fixed-point MLP tested on Mnist. 

See `examples/config_fix_wag.yaml.sample` for a example of fixed-point configuration.

Try `python mlp.py --cfg config_fix_wag.yaml.sample`.

Others
------------

### Logging

Use the environment variable `NICS_FIX_LOGLEVEL` to control the python logging module. Avaiable log levels are `{"debug", "info", "warning", "error", "fatal"}`. By default, the log level is set to "warning".