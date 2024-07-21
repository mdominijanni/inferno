![Inferno Header](https://raw.githubusercontent.com/mdominijanni/inferno/dev/misc/assets/inferno-header-github.png)

## About
Inferno is an extensible library for simulating spiking neural networks. It is built on top of [PyTorch](https://github.com/pytorch/pytorch) and is designed with machine learning practitioners in mind. This project is still an early release and features may be subject to change.

## Installation
### Pip
#### With PyTorch 2.3.1 (CPU Only)
```
pip install inferno-ai[torch]
```

#### Without PyTorch
```
pip install inferno-ai
```
*Note: Inferno still requires PyTorch and a version of it must be installed. PyTorch will need to be installed separately if CUDA support is required. See the [instructions for installing PyTorch](https://pytorch.org/get-started/locally/) for details.*