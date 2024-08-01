# Inferno Documentation

```{toctree}
:hidden:

reference/index
guide/index
examples/index
zoo/index
```

Inferno is an extensible library for simulating spiking neural networks. Built on top of [PyTorch](https://github.com/pytorch/pytorch), it is designed with machine learning practitioners in mind. This project is still an early release and features may be subject to change.

## Installation

At this time, installation is only supported via pip. Future support for installation with conda is planned.

### With PyTorch (CPU Only)

```
pip install inferno-ai[torch]
```

Installation with the "torch" flag will install PyTorch 2.3.1 and the corresponding version of TorchVision.

### Without PyTorch

```
pip install inferno-ai
```

_Note: Inferno still requires PyTorch and a version of it must be installed. PyTorch will need to be installed separately if CUDA support is required. See the [instructions for installing PyTorch](https://pytorch.org/get-started/locally/) for details._

## Getting Started

For an example of walking through building, training, and evaluating a spiking neural network, see the [quickstart example](<examples/quickstart:Quickstart>). A comprehensive documentation of the classes and functions provided by Inferno are available [here](<reference/index:API Reference>).
