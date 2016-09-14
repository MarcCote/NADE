#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

from os.path import join as pjoin
import argparse

import pickle
import numpy as np

import smartlearner.utils as smartutils
from convnade.utils import Timer


def buildArgsParser():
    DESCRIPTION = "Generate samples from a Conv Deep NADE model."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('experiment', type=str, help='folder where to find a trained ConvDeepNADE model')
    p.add_argument('count', type=int, help='number of samples to generate.')
    p.add_argument('--out', type=str, help='name of the samples file')

    # General parameters (optional)
    p.add_argument('--seed', type=int, help='seed used to generate random numbers. Default: 1234', default=1234)
    p.add_argument('--view', action='store_true', help="show samples.")

    p.add_argument('-v', '--verbose', action='store_true', help='produce verbose output')
    p.add_argument('-f', '--force',  action='store_true', help='permit overwriting')

    return p


def load_model(experiment_path):

    with Timer("Loading model"):
        from convnade import DeepConvNadeUsingLasagne, DeepConvNadeWithResidualUsingLasagne
        from convnade import DeepConvNADE, DeepConvNADEWithResidual

        for model_class in [DeepConvNadeUsingLasagne, DeepConvNadeWithResidualUsingLasagne, DeepConvNADE, DeepConvNADEWithResidual]:
            try:
                model = model_class.create(experiment_path)
                return model
            except Exception as e:
                print (e)
                pass

    raise NameError("No model found!")
    return None


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    # Load experiments hyperparameters
    try:
        hyperparams = smartutils.load_dict_from_json_file(pjoin(args.experiment, "hyperparams.json"))
    except:
        hyperparams = smartutils.load_dict_from_json_file(pjoin(args.experiment, '..', "hyperparams.json"))

    model = load_model(args.experiment)
    print(str(model))

    with Timer("Generating {} samples from Conv Deep NADE".format(args.count)):
        sample = model.build_sampling_function(seed=args.seed)
        samples, probs = sample(args.count, return_probs=True, ordering_seed=args.seed)

    if args.out is not None:
        outfile = pjoin(args.experiment, args.out)
        with Timer("Saving {0} samples to '{1}'".format(args.count, outfile)):
            np.save(outfile, samples)

    if args.view:
        import pylab as plt
        from convnade import vizu
        if hyperparams["dataset"] == "binarized_mnist":
            image_shape = (28, 28)
        else:
            raise ValueError("Unknown dataset: {0}".format(hyperparams["dataset"]))

        plt.figure()
        data = vizu.concatenate_images(samples, shape=image_shape, border_size=1, clim=(0, 1))
        plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
        plt.title("Samples")

        plt.figure()
        data = vizu.concatenate_images(probs, shape=image_shape, border_size=1, clim=(0, 1))
        plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
        plt.title("Probs")

        plt.show()

if __name__ == '__main__':
    main()
