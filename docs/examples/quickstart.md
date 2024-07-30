# Quickstart

Inspired by the PyTorch [example](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) of the same name, we go through a simple example of implementing a model and a training/testing loop.

```{eval-rst}
.. only:: builder_html

    Download this example as a
    :download:`Python script <./code/quickstart.py>`
    or as a
    :download:`Jupyter notebook <./code/quickstart.ipynb>`
    .
```

## Prerequisites

This tutorial requires Inferno (and its dependencies, including PyTorch) and TorchVision for working with the dataset. We'll start with the necessary import statements.

```{code} python
import inferno
import torch
from inferno import functional, neural, learn
from torch.utils.data import DataLoader
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
    device = "cpu"

print(f"using device: {device}\n")
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
            Lambda(lambda x: x.squeeze(0)),
        ],
    ),
    download=True,
)

test_set = MNIST(
    root="data",
    train=False,
    transform=Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Lambda(lambda x: x.squeeze(0)),
        ],
    ),
    download=True,
)

train_data = DataLoader(train_set, batch_size=batch_size)
test_data = DataLoader(test_set, batch_size=batch_size)
```

In this example, we'll be classifying examples from the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset of handwritten digits using a model based on the one described [here](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full).

## Constructing Neural Components

Inferno divides the simulation of spiking neural networks into three basic types of components: neurons ({py:class}`~inferno.neural.Neuron`), synapses ({py:class}`~inferno.neural.Synapse`), and connections ({py:class}`~inferno.neural.Connection`). We'll start with neurons, which consume real-valued input representing electrical current and produce boolean output representing action potentials (also called spikes). In terms of artificial neural networks, these serve a role similar to that of activation functions, but neuron models are generally far more complex. They are also stateful and need to maintain their state for each sample in a batch.

Here we are creating two groups of neurons, `exc` and `inh`, which respectively correspond to an excitatory and an inhibitory group of neurons. The output of the model will come from the former, but the latter will be used to regulate it. The inhibitory group is made up of leaky-integrate and fire ({py:class}`~inferno.neural.LIF`) neurons and the excitatory group is made up of adaptive leaky-integrate and fire ({py:class}`~inferno.neural.ALIF`) neurons.

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
    batch_size=batch_size,
)

inh2exc = neural.LinearLateral(
    shape,
    step_time,
    synapse=neural.DeltaCurrent.partialconstructor(100.0),
    weight_init=lambda x: inferno.full(x, -180.0),
    batch_size=batch_size,
)

exc2inh = neural.LinearDirect(
    shape,
    step_time,
    synapse=neural.DeltaCurrent.partialconstructor(75.0),
    weight_init=lambda x: inferno.full(x, 22.5),
    batch_size=batch_size,
)
```

## Creating the Model

With all the separate components created, we can now put them together into a single model. Inferno models a set of connections and a set of neurons which take input from those connections with the
{py:class}`~inferno.neural.Layer` class. As in the model here, neuron groups can be shared across layers (connections normally will not be and care will need to be taken to avoid complications beyond the scope of this example, but Inferno doesn't explicitly prohibit it). The mapping between connections is customizable, although Inferno provides some common layer architectures for convenience. {py:class}`~inferno.neural.Serial` maps a single connection to a single group of neurons.

Implementing the `Model` class follows the normal PyTorch pattern. Here, we use the previously built components to create the layers. We then write the `forward()` function such that it takes in the tensor of spikes from the encoder, applies the inputs at each step, and returns the concatenated output. It will also update the connection if a trainer is passed in (more on this in the next section) and the module is in `training` mode.

```{code} python
class Model(inferno.Module):

    def __init__(self, exc, inh, enc2exc, inh2exc, exc2inh):

        # call superclass constructor
        inferno.Module.__init__(self)

        # construct the layers
        self.feedfwd = neural.Serial(enc2exc, exc)
        self.inhibit = neural.Serial(inh2exc, exc)
        self.trigger = neural.Serial(exc2inh, inh)

    def forward(self, inputs, trainer=None):
        # clears the model state
        def clear(m):
            if isinstance(m, neural.Neuron | neural.Connection):
                m.clear()

        # compute for each time step
        def step(x):

            # inference
            res = self.feedfwd(x)
            _ = self.inhibit(self.trigger(res))

            # training
            if self.training and trainer:
                trainer()
                self.feedfwd.connection.update()

            return res

        res = torch.stack([step(x) for x in inputs], dim=0)
        self.apply(clear)

        return res

model = Model(exc, inh, enc2exc, inh2exc, exc2inh)
model.to(device=device)
```

While this model is being trained, we will want to keep the weights of `enc2exc` normalized such that the $\ell_1$-norm of the weight vector corresponding to each output has a fixed value. Inferno extends PyTorch's _forward hooks_. While custom hooks can still be set with {py:meth}`~torch.nn.Module.register_forward_hook` and {py:meth}`~torch.nn.Module.register_forward_pre_hook`, Inferno provides a {py:class}`~inferno.Hook` class along with multiple extensions for managing forward hooks. Hooks can be set to trigger conditionally based on if the module is in `training` mode, which is set with {py:meth}`~torch.nn.Module.train` and {py:meth}`~torch.nn.Module.eval`. Here, we'll normalize the weights when training only (since the weights won't change during inference).

```{code} python
norm_hook = neural.Normalization(
    model,
    "feedfwd.connection.weight",
    order=1,
    scale=78.4,
    dim=-1,
    eval_update=False,
)
norm_hook.register()
norm_hook()
```

By default, {py:class}`~inferno.neural.Normalization` (like every hook by convention), will run as a post hook (after {py:meth}`~torch.nn.Module.forward` is run) and will run regardless of the `training` mode. Here, we're specifying that when `model.training` is `False`, the hook shouldn't be run. We're also configuring it such that `norm_hook` is a hook on `model`, so it is only run once per batch and not once per simulation step. The attribute on which the hook is applied can be specified using dot notation to target nested attributes. We then register the hook, so it will be called automatically. Finally, because `Normalization` is a {py:class}`~inferno.StateHook`, we can call the hook to apply its transformation even without `model.forward()` being called.

## Configuring the Trainer

Unlike in artificial neural networks where parameters are _optimized_, typically with a variant of gradient descent, parameters in spiking neural networks are trained using a variety of methods (including optimization). Inferno's development to date has focused on the use of plasticity rules such as [spike-timing dependent plasticity (STDP)](<zoo/learning-stdp:Spike-Timing Dependent Plasticity (STDP)>).

In this example, we will use {py:class}`~inferno.learn.STDP` itself. When constructing `STDP`, it will take in a set of default hyperparameters which will apply to the training for each {py:class}`~inferno.neural.Cell` we register with it, although it can be overridden on a cell-by-cell base. Inferno uses the concept of a `Cell` for defining the structure of a trainable model. Each cell represents a tuple of a {py:class}`~inferno.neural.Connection` and a {py:class}`~inferno.neural.Neuron`, where the latter takes its output from the former.

A limit imposed on cells is that the output shape, {py:attr}`~inferno.neural.Connection.outshape`, of any connection must be the same as the {py:attr}`~inferno.neural.Neuron.shape` of the neurons. These are constructed by {py:class}`~inferno.neural.Layer` classes, where {py:class}`~inferno.neural.Serial` does so automatically. Any custom layer implementation should create cells along the "trainable path".

```{code} python
trainer = learn.STDP(
    step_time=step_time,
    lr_post=5e-4,
    lr_pre=-5e-6,
    tc_post=30.0,
    tc_pre=30.0,
)
```

If we look at the `STDP` class, we'll see that it inherits from {py:class}`~inferno.learn.IndependentCellTrainer`, which has some implementation to help with developing classes for training methods where the training of each cell is independent of every other cell. This inherits from {py:class}`~inferno.learn.CellTrainer`, which provides a more limited implementation and should be used for training methods where some dependency between cells does exist.

Before we can add the cell in our model to our trainer, we need to make it trainable. This means we need to give the connection in it an {py:class}`~inferno.neural.Updater`. The updater is responsible for accumulating and applying updates to the trainable parameters of a connection. The behavior of how updates are applied can then be controlled (more on this in the next section). Here, we'll add the default updater to our connection, then register the cell.

```{code} python
updater = model.feedfwd.connection.defaultupdater()
model.feedfwd.connection.updater = updater

trainer.register_cell("feedfwd", model.feedfwd.cell)
trainer.to(device=device)
```

We've already specified that the weights should be normalized, but we also want to specify that they stay within a certain range. The simplest way of doing this is by clamping (also called clipping) the values within a range. This can also be done with [parameter dependence](<guide/concepts:Parameter Dependence>). For this example, the lower bound will be clamped, and the upper bound will be enforced with multiplicative parameter dependence.

Clamping the weights are done in much the same way as normalizing them, except this time we want it to trigger after each call of the updater. Therefore, we'll register the hook to the updater itself and define the target using the {py:meth}`~inferno.neural.Updater.parent` property.

```{code} python
clamp_hook = neural.Clamping(
    updater,
    "parent.weight",
    min=0.0,
)
clamp_hook.register()
```

Adding parameter dependence is done by accessing the {py:class}`~inferno.neural.Accumulator` for the associated parameter. To only bound the upper limit, {py:meth}`~inferno.neural.Accumulator.upperbound` is called and a function which follows the {py:class}`~inferno.functional.HalfBounding` protocol.

```{code} python
updater.weight.upperbound(
    functional.bound_upper_multiplicative,
    1.0,
)
```

It is important to note that STDP is an _unsupervised_ training method, but we're using it for a classification task. To this end, we'll need a classifier. In this case, we're using {py:class}`~inferno.learn.MaxRateClassifier`. It will assign each neuron a label corresponding to the class for which it fired most frequently.

```{code} python
classifier = learn.MaxRateClassifier(
    shape,
    num_classes=10,
    decay=1e-6,
)
classifier.to(device=device)
```

## Training/Testing Loop

Spiking neural networks require that we convert the real-valued data we normally work with into _spike trains_, a sequence of zeros and ones representing if a spike occurred at a given time step. Inferno represents these as boolean tensors, typically where samples over the first dimension indicate the spikes at each time step. Most commonly this is done by representing spike generation as a Poisson point process. The intervals between spikes follow an exponential distribution with a mean equal to the expected frequency. More on Poisson spike generation can be found in this [excellent write-up](https://github.com/btel/python-in-neuroscience-tutorials/blob/master/poisson_process.ipynb).

```{code} python
encoder = neural.HomogeneousPoissonEncoder(
    250,
    step_time,
    frequency=128.0,
)
```

As previously mentioned, the convention in Inferno is for encoders to take inputs scaled between zero and one. Here, the frequency (in Hertz) is a linear scaling factor for determining the expected spike rate for a given input.

We can then write the functions for the training/testing loop. Note that the classifier isn't updated every batch, but in this case every tenth batch.

```{code} python
classify_interval = 10
train_cutoff = None

def train(data, encoder, model, trainer, classifier):
    size = len(data.dataset)
    rates, labels = [], []
    correct, current = 0, 0

    for batch, (X, y) in enumerate(data, start=1):
        X, y = X.to(device=device), y.to(device=device)

        rates.append(model(encoder(X), trainer).float().mean(dim=0))
        labels.append(y)

        if batch % classify_interval == 0:
            rates = torch.cat(rates, dim=0)
            labels = torch.cat(labels, dim=0)

            pred = classifier(rates, labels)
            nc = (pred == labels).sum().item()
            correct += nc
            current += rates.size(0)

            print(f"acc: {(nc / rates.size(0)):.4f} [{current:>5d}/{size:>5d}]")
            rates, labels = [], []

        if train_cutoff is not None and current >= train_cutoff:
            break

    print(f"Training Accuracy:\n    {(correct / current):.4f}")
```

The function for testing is very similar, except it excludes the trainer as it won't be updated. The classifier is still passed in for computing the prediction accuracy. Unlike in the training function, the true labels aren't given to the classifier, and therefore it will only infer.

```{code} python
test_cutoff = None

def test(data, encoder, model, classifier):
    correct, current = 0, 0

    for batch, (X, y) in enumerate(data, start=1):
        X, y = X.to(device=device), y.to(device=device)

        rates = model(encoder(X), None).float().mean(dim=0)
        pred = classifier(rates, None)

        correct += (pred == y).sum().item()
        current += rates.size(0)

        if test_cutoff is not None and current >= test_cutoff:
            break

    print(f"Testing Accuracy:\n    {(correct / current):.4f}\n")
```

For the main loop, we need to set both the model and the trainer into the correct `training` mode. This will manage the hook triggers. When the trainer is set into evaluation mode, it will also deregister any hooks it creates in order to avoid runtime tests of the `training` mode.

```{code} python
epochs = 1

with torch.no_grad():
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------")
        model.train(), trainer.train()
        train(train_data, encoder, model, trainer, classifier)
        model.eval(), trainer.eval()
        test(test_data, encoder, model, classifier)
    print("Completed")
```

