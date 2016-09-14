# Deep NADE and r-NADE

Deep (Orderless) extension and real-value extension of the NADE model.

## Installation
#### Dependencies
- Python 2
- numpy
- h5py
- [Theano](https://github.com/Theano/Theano)

#### Install
1. `git clone https://github.com/MarcCote/NADE.git`
2. `cd NADE/deepnade`

## Usage
Training a Deep NADE model on binarized MNIST.
```
source run_orderless_nade_binary.sh
```

Training a Deep NADE model on RedWine dataset.
```
source run_orderless_nade.sh
```


## Datasets
The datasets are already included in this repository as HDF5 files. Available datasets are:

- binarized_mnist.hdf5 (219 Mb)
- red_wine.hdf5 (923 Kb)
