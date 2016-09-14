# Neural Autoregressive Distribution Estimation

This repository contains code for NADE and its extensions: R-NADE, Deep (Orderless) NADE and Convolutional NADE.

## Paper

#### Abstract
We present Neural Autoregressive Distribution Estimation (NADE) models,
which are neural network architectures applied to the problem of
unsupervised distribution and density estimation. They leverage
the probability product rule and a weight sharing scheme inspired
from restricted Boltzmann machines, to yield an estimator that
is both tractable and has good generalization performance.
We discuss how they achieve competitive performance in modeling both binary and
real-valued observations. We also present how deep NADE models can
be trained to be agnostic to the ordering of input dimensions used
by the autoregressive product rule decomposition. Finally, we also show
how to exploit the topological
structure of pixels in images using a deep convolutional architecture for NADE.

#### arXiv
https://arxiv.org/abs/1605.02226