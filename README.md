# Fixed Point Training Simulation Framework

This is a fixed-point training simulation framework based on tensorflow.

Design
------------

There are two phases that we care about in deploying NN to hardware with only fixed-point calculation units:
* TSDH: Training on Software, Deploy a fixed-point model on Hardware.
* THDH: fixed-point Training on Hardware directly, Deployed on Hardware.

In this documentation, unless otherwise specified, "fixed-point training" are refering to both of these phases in this documentation.

### Basics

We have four things to be quantitized potentially:

* weights
* gradients of weights
* activations
* gradients of activations

For transparent conversion to fixed point simulation, we supply a context manager for managing fixed point configuration: `nics_scope.fixed_scope`, and wrapper of nn operations. Eg. `Dense` for `tf.layers.dense`, `Conv2d` for tf.layers.conv2d.

Using these wrappers inside the context manager, the above four things in the models created will be handled transparently. You can also manually insert `.apply(quantitize)` operation into necessary places.

#### Fixed Config

**TODO**: Describe the format of the fixed configuration file.

### Strategy

In the above section, using the context manager with suitable quantization configs, you can simulate the fixed-point computation on the hardware. This is a simulation of hardware constraints.

However, to do better training, or to explore what training techniques can be applied to enable better fixed-point training, we supply a strategy interface, and some pre-defined strategies, see [Strategies](#strategies) for the pre-defined strategies.

You can implement your own strategy by creating a class inheriting `nics_fix.Strategy`, and implement its methods:
* `pre_weight`
* `post_weight`
* `pre_weight_grad`
* `post_weight_grad`
* `pre_activation`
* `post_activation`
* `pre_activation_grad`
* `post_activation_grad`

These methods all receive a tensor as input, and should return a tensor. By default, they just return their input tensor. The `pre_*` functions's output tensor will be fed into its corresponding quantitize operation, and the output tensor of the quantitize operation will be fed into the `post_*` functions.

When using strategy, call `nics_scope.fixed_scope` with a strategy config. You should specify where these strategies should be used in the strategy config.

You can nested multiple strategies, the first strategy's `pre_*` methods will be first executed among all the strategies, while its `post_*` methods will be last executed.

<a name="strategies"></a>
#### Strategies

* `NoiseStrategy`: Insert noise into points. See the class' documentation string for details.

#### StrategyConfig

**TODO**: Describe the format of the strategy configuration file.

Examples
------------

See `examples/mlp.py` for a simple example of a 1-hidden-layer fixed-point MLP tested on Mnist. 

See `examples/config_fix_wag.yaml.sample` for a example of fixed-point configuration.

See `examples/config_strategy_noise.yaml.sample` for a example of strategy configuration, using the pre-defined `NoiseStrategy`.

Try `python mlp.py --cfg config_fix_wag.yaml.sample`.

Other
------------

### Logging

Use the environment variable `NICS_FIX_LOGLEVEL` to control the python logging module. Avaiable log levels are `{"debug", "info", "warning", "error", "fatal"}`. By default, the log level is set to "warning".