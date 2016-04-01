#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import re
import pickle
import argparse
import numpy as np
from os.path import join as pjoin

from smartlearner import views
from smartlearner.status import Status
from smartlearner import utils as smartutils

from convnade import utils
from convnade import datasets
from convnade.utils import Timer

from convnade.batch_schedulers import MiniBatchSchedulerWithAutoregressiveMask
from convnade.losses import BinaryCrossEntropyEstimateWithAutoRegressiveMask

DATASETS = ['binarized_mnist']


def build_argparser():
    DESCRIPTION = "Evaluate the NLL estimate of a ConvNADE model."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('experiment', type=str, help='folder where to find a trained ConvDeepNADE model')
    p.add_argument('--seed', type=int,
                   help="Seed used to choose the orderings. Default: 1234", default=1234)
    p.add_argument('--batch-size', type=int,
                   help='if specified, will use try this batch_size first and will reduce it if needed.')

    # Optional parameters
    p.add_argument('-f', '--force',  action='store_true', help='overwrite evaluation results')

    return p


def estimate_NLL(model, dataset, seed=1234, batch_size=None):
    if batch_size is None:
        batch_size = len(dataset)

    loss = BinaryCrossEntropyEstimateWithAutoRegressiveMask(model, dataset)
    status = Status()

    batch_scheduler = MiniBatchSchedulerWithAutoregressiveMask(dataset,
                                                               batch_size=len(dataset),
                                                               use_mask_as_input=model.nb_channels == 2,
                                                               keep_mask=True,
                                                               seed=seed)

    nll = views.LossView(loss=loss, batch_scheduler=batch_scheduler)

    # Try different size of batch size.
    while batch_size >= 1:
        print("Estimating NLL using batch size of {}".format(batch_size))
        try:
            batch_scheduler.batch_size = min(batch_size, len(dataset))
            return {"mean": float(nll.mean.view(status)),
                    "stderror": float(nll.stderror.view(status))}

        except MemoryError as e:
            # Probably not enough memory on GPU
            #print("\n".join([l for l in str(e).split("\n") if "allocating" in l]))
            pass

        except RuntimeError as e:
            # Probably RuntimeError: BaseGpuCorrMM: Failed to allocate output of
            if "allocate" not in str(e):
                raise e

        print("*An error occured while estimating NLL. Will try a smaller batch size.")
        batch_size = batch_size // 2

    raise RuntimeError("Cannot find a suitable batch size to estimate NLL. Try using CPU instead or a GPU with more memory.")


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
    parser = build_argparser()
    args = parser.parse_args()

    # Load experiments hyperparameters
    try:
        hyperparams = smartutils.load_dict_from_json_file(pjoin(args.experiment, "hyperparams.json"))
    except:
        hyperparams = smartutils.load_dict_from_json_file(pjoin(args.experiment, '..', "hyperparams.json"))

    model = load_model(args.experiment)
    print(str(model))

    with Timer("Loading dataset"):
        trainset, validset, testset = datasets.load(hyperparams['dataset'], keep_on_cpu=True)
        print(" (data: {:,}; {:,}; {:,}) ".format(len(trainset), len(validset), len(testset)), end="")

    # Result files.
    result_file = pjoin(args.experiment, "results_estimate.json")

    if not os.path.isfile(result_file) or args.force:
        with Timer("Evaluating NLL estimate"):
            results = {"seed": args.seed}
            results['trainset'] = estimate_NLL(model, trainset, seed=args.seed, batch_size=args.batch_size)
            results['validset'] = estimate_NLL(model, validset, seed=args.seed, batch_size=args.batch_size)
            results['testset'] = estimate_NLL(model, testset, seed=args.seed, batch_size=args.batch_size)
            utils.save_dict_to_json_file(result_file, {"NLL_estimate": results})
    else:
        print("Loading saved results... (use --force to re-run evaluation)")
        results = utils.load_dict_from_json_file(result_file)['NLL_estimate']

    for dataset in ['trainset', 'validset', 'testset']:
        print("NLL estimate on {}: {:.2f} ± {:.2f}".format(dataset, results[dataset]['mean'], results[dataset]['stderror']))

if __name__ == '__main__':
    main()
