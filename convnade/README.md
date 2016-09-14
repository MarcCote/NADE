# Convolutional NADE

Convolution extension of the NADE model.

## Installation
#### Dependencies
- Python 3
- numpy
- [Theano](https://github.com/Theano/Theano)
- [Smartlearner](https://github.com/SMART-Lab/smartlearner)

#### Install
1. `git clone https://github.com/MarcCote/NADE.git`
2. `cd NADE/convnade`
3. `pip install --process-dependency-links .`

## Usage

#### Training
Training a Convolutional NADE model on binarized MNIST.
```
python scripts/train_convnade.py --batch-size 100 --Adam 1e-4 --name best_convnade convnade --convnet-blueprint "64@8x8(valid)->64@4x4(valid)->64@6x6(valid)->64@7x7(valid)->64@7x7(full)->64@6x6(full)->64@4x4(full)->1@8x8(full)" --fullnet-blueprint "500->500->784" --use-mask-as-input --hidden-activation hinge --weights-initialization uniform binarized_mnist
```

#### Evaluation
Given a trained ConvNADE model, we can estimate (faster) or compute the exact (slower) NLL.

Estimating the NLL is done as follow:
```
python scripts/estimate_nll.py experiments/best_convnade
```

Computing the exact NLL is not as straightforward. This type of evaluation is highly computationally intensive (especially when using ensemble). To speedup the process, this evaluation can be easily parallelized by splitting the testset in chunks and treating each ordering independently. For instance, computing the exact NLL of the 20th batch (from a total of 100 batches) for the 50th ordering (from a total of 128 orderings) would be achieved using this command:
```
python scripts/compute_nll.py experiments/best_convnade testset 20 50 --batch-size 1000 --nb-orderings 128
```

Once all batches and all orderings have been process, that is 128*100 different `compute_nll.py` comands have been executed, we are ready to merge all partial NLL results in order to obtain our exact NLL of the testset.
```
python scripts/merge_nll.py experiments/best_convnade
```

#### Sampling
Generating 16 binarized MNIST digits images sampled from a trained model.
```
python scripts/sample_convnade.py experiments/best_convnade 16 --view
```

## Datasets
The datasets are automatically downloaded and processed. Available datasets are:

- binarized MNIST
