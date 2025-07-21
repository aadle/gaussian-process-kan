# Kolmogorov-Arnold Networks with Gaussian Process activations

This is the codebase of my masters project where I explored Gaussian
Processes as a replacement for the learnable splines activation function
provided in the original implementation of Liu et al[1].

This implementation is built entirely from scratch with JAX, with the exception
of the Gaussian Processes where we use the
[GPJax](https://github.com/JaxGaussianProcesses/GPJax) package. We have focused
on functional correctness rather than efficiency, hence the limited
implementation.

Not all of the experiments conducted in the thesis is included in this
repository.

## Abstract

The Kolmogorov-Arnold Network (KAN) is a recent addition to the family of neural
networks, proposed as an alternative to the Multi-Layer Perceptron (MLP). KANs,
unlike MLPs, do not have learnable weights but instead possess learnable
activation functions parameterized as univariate splines[1]. In this work, we
change the activation functions from splines to Gaussian Processes (GP),
allowing the KAN model to also model with uncertainty.

## Usage

Please refer to one of Jupyter Notebooks in the example folder.

## References

[1] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljaic, T. Y.
Hou, and M. Tegmark. KAN: Kolmogorov-Arnold Networks. url:
[https://arxiv.org/abs/2404.19756](https://arxiv.org/abs/2404.19756). 2025
