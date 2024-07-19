# Quickstart

Inspired by the PyTorch [example](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) of the same name, we go through a simple example of implementing a model and a training/testing loop.

## Prerequisites
This tutorial requires Inferno (and its dependencies, including PyTorch) and TorchVision for working with the dataset. We'll start with the necessary import statements.

```{code} python
import inferno
import torch
from inferno import neural, learn
from torch import nn
from torch.utils.data import Dataloader
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Lambda
```

Then we set some values used throughout this example.

```{code} python
shape = (10, 10)
step_time = 1.0
batch_size = 20

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu
```

## Retrieving and Loading Data
Inferno works directly with PyTorch, and therefore we can use the same methods to import and transform the data with which we want to work. Unlike with artificial neural networks, we'll need to convert each sample into a spike train. In this example, we'll be using {py:class}`~inferno.neural.HomogeneousPoissonEncoder` which takes a tensor of floats in the range $[0, 1]$.

```{code} python
train_set = MNIST(
    root="data",
    train=True,
    transform=Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Lambda(lambda x: x.squeeze(0))
        ]
    )
    download=True,
)

test_set = MNIST(
    root="data",
    train=False,
    transform=Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Lambda(lambda x: x.squeeze(0))
        ]
    )
    download=True,
)
```

## Constructing Neural Components
Inferno divides the simulation of spiking neural networks into three basic types of components: neurons ({py:class}`~inferno.neural.Neuron`), synapses ({py:class}`~inferno.neural.Synapse`), and connections ({py:class}`~inferno.neural.Connection`). We'll start with neurons, which consume real-valued input representing electrical current and produce boolean output representing action potentials (also called spikes). In terms of artificial neural networks, these serve a role similar to that of activation functions, but neuron models are generally far more complex. They are also stateful and need to maintain their state for each sample in a batch.

Here we are creating two groups of neurons, `exc` and `inh`, which respectively correspond to an excitatory and an inhibitory group of neurons. The output of the model will come from the former, but the latter will be used to regulate it. The inhibitory group is made up of leaky-integrate and fire (LIF) neurons and the excitatory group is made up of adaptive leaky-integrate and fire (ALIF) neurons.

```{code} python
exc = neural.ALIF(
    shape,
    step_time,
    rest_v=-65.0,
    reset_v=-60.0,
    thresh_eq_v=-52.0,
    refrac_t=5.0,
    tc_membrane=100.0,
    tc_adaptation=1e7,
    spike_increment=0.05,
    resistance=1.0,
    batch_size=batch_size,
)

inh = neural.LIF(
    shape,
    step_time,
    rest_v=-60.0,
    reset_v=-45.0,
    thresh_v=-40.0,
    refrac_t=2.0,
    time_constant=75.0,
    resistance=1.0,
    batch_size=batch_size,
)
```

Then connections take these boolean spikes and produce real-valued currents which have been transformed by some trainable mapping. The process of transforming spikes into currents is performed by synapses. Because certain internal properties of synapses depend upon properties of the connection with which they're associated, Inferno has synapses constructed by the connection and composed within it.

Here we're creating three connections: `enc2exc`, `inh2exc`, and `exc2inh`—and are building a model similar to the one described in [this](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full) paper. Each connection does the following.

- `enc2exc`: trainable mapping from the encoded inputs to the excitatory neurons.
- `inh2exc`: applies "lateral inhibition", where the firing of every excitatory neuron other than itself inhibits other excitatory neurons from firing.
- `exc2inh`: triggers the lateral inhibition.

In order to do this, we use the three different types of linear connections offered by Inferno. {py:class}`~inferno.neural.LinearDense` acts exactly like a normal {py:class}`~torch.nn.Linear` connection, providing a linear all-to-all mapping between inputs and outputs. {py:class}`~inferno.neural.LinearLateral` masks the parameters along the diagonal with zeros, making an all-to-"all but self" mapping. {py:class}`~inferno.neural.LinearDirect` provides a one-to-one mapping.

```{code} python
enc2exc = neural.LinearDense(
    (28, 28),
    shape,
    step_time,
    synapse=neural.DeltaCurrent.partialconstructor(100.0),
    weight_init=lambda x: inferno.rescale(inferno.uniform(x), 0.0, 0.3),
)

inh2exc = neural.LinearLateral(
    shape,
    step_time,
    synapse=neural.DeltaCurrent.partialconstructor(100.0),
    weight_init=lambda x: inferno.full(x, -180.0),
)

exc2inh = neural.LinearDirect(
    shape,
    step_time,
    synapse=neural.DeltaCurrent.partialconstructor(75.0),
    weight_init=inferno.full(x, 22.5),
)
```

## Creating the Model
With all of the separate components created, we can now put them together into a single model. Inferno models a set of connections and a set of neurons which take input from those connections with the
{py:class}`~inferno.neural.Layer` class. As in the model here, neuron groups can be shared across layers (connections normally will not be and care will need to be taken to avoid complications beyond the scope of this example, but Inferno doesn't explicitally prohibit it). The mapping between connections is customizable, although Inferno provides some common layer architectures for convenience. {py:class}`~inferno.neural.Serial` maps a single connection to a single group of neurons.

Implementing the `Model` class follows the normal PyTorch pattern. Here, we use the previously built components to create the layers. We then write the `forward()` function such that it takes in the tensor of spikes from the encoder, applies the inputs at each step, and returns the concatenated output. It will also update the connection if a trainer is passed in (more on this in the next section) and the module is in `training` mode.

```{code} python
class Model(inferno.Module):

    def __init__(self, exc, inh, enc2exc, inh2exc, exc2inh):

        # call superclass constructor
        inferno.Module.__init__(self)

        # construct the layers
        self.feedfwd = snn.Serial(enc2exc, exc)
        self.inhibit = snn.Serial(inh2exc, exc)
        self.trigger = snn.Serial(exc2inh, inh)


    def forward(self, inputs, trainer=None):

        # compute for each time step
        def step(x):

            # infer
            res = self.feedfwd(step)
            _ = self.inhibit(self.trigger(res))

            # train
            if self.training and trainer:
                trainer()
                self.feedfwd.connection.update()

            return res

        return torch.stack([step(x) for x in inputs], dim=0)

model = Model(exc, inh, enc2exc, inh2exc, exc2inh)
model.to(device=device)
```

## Configuring the Trainer
Unlike in artificial neural networks where parameters are *optimized*, typically with a variant of gradient descent, parameters in spiking neural networks are trained using a variety of methods (including optimization). Inferno's development to date has focused on the use of plasticity rules such as [spike-timing dependent plasticity (STDP)](<zoo/learning-stdp:Spike-Timing Dependent Plasticity (STDP)>).

In this example, we will use {py:class}`~inferno.learn.STDP` itself. When constructing `STDP`, it will take in a set of default hyperparameters which will apply to the training for each {py:class}`~inferno.neural.Cell` we register with it, although it can be overridden on a cell-by-cell base. Inferno uses the concept of a `Cell` for defining the structure of a trainable model. Each cell represents a tuple of a {py:class}`~inferno.neural.Connection` and a {py:class}`~inferno.neural.Neuron`, where the latter takes its output from the former.

A limit imposed on cells is that the output shape, {py:attr}`~inferno.neural.Connection.outshape`, of any connection must be the same as the {py:attr}`~inferno.neural.Neuron.shape` of the neurons. These are constructed by {py:class}`~inferno.neural.Layer` classes, where {py:class}`~inferno.neural.Serial` does so automatically. Any custom layer implementation should create cells along the "trainable path".

```{code} python
trainer = learn.STDP(
        step_time=step_time,
        lr_post=5e-4,
        lr_pre=-5e-6,
        tc_post=30.0,
        tc_pre=30.0,
        delayed=False,
        interp_tolerance=0.0,
        trace_mode="cumulative",
        batch_reduction="mean",
    )
```

If we look at the `STDP` class, we'll see that it inherits from {py:class}`~inferno.learn.IndependentCellTrainer`, which has some implementation to help with developing classes for training methods where the training of each cell is independent of every other cell. This inherits from {py:class}`~inferno.learn.CellTrainer`, which provides a more limited implementation and should be used for training methods where some dependency between cells does exist.

Before we can add the trainable cell in our model to our trainer, we need to make it trainable. This means we need to give it an {py:class}`~inferno.neural.Updater`. The updater is responsible for accumulating and applying updates to the trainable parameters of a connection. The behavior of how updates are applied can then be controlled (more on this in the next section). Here, we'll add the default updater to our connection, then register the cell we want to train.

```{code} python
model.feedfwd.connection.updater = model.feedfwd.connection.defaultupdater()
trainer.register_cell("feedfwd", model.feedfwd.cell)
```

## Adding Update Hooks
```{code} python
# add bounding hooks
```

```{code} python
# add normalization hook
```

## Training/Testing Loop

```{code} python
# create the classifier
```

```{code} python
encoder = neural.HomogeneousPoissonEncoder(250, step_time, 128.0)
```

