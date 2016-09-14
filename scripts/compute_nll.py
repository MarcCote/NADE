#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import argparse
import numpy as np
from os.path import join as pjoin

from smartlearner import views
from smartlearner.status import Status
from smartlearner import utils as smartutils

from convnade import datasets
from convnade.utils import Timer

from convnade.batch_schedulers import BatchSchedulerWithAutoregressiveMasks
from convnade.losses import NllUsingBinaryCrossEntropyWithAutoRegressiveMask


DATASETS = ['binarized_mnist']


def build_argparser():
    DESCRIPTION = "Evaluate the exact NLL of a ConvNADE model."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('experiment', type=str, help='folder where to find a trained ConvDeepNADE model')
    p.add_argument('subset', type=str, choices=['testset', 'validset'],
                   help='evaluate a specific subset (either "testset" or "validset").')

    p.add_argument('batch_id', type=int, help='evaluate only a specific batch.')
    p.add_argument('ordering_id', type=int, help='evaluate only a specific input ordering.')
    p.add_argument('--batch-size', type=int, help='size of the batch to use when evaluating the model.', default=100)
    p.add_argument('--nb-orderings', type=int, help='evaluate that many input orderings. Default: 128', default=128)
    # p.add_argument('--subset', type=str, choices=['valid', 'test'],
    #                help='evaluate only a specific subset (either "testset" or "validset") {0}]. Default: evaluate both subsets.')

    p.add_argument('--seed', type=int,
                   help="Seed used to choose the orderings. Default: 1234", default=1234)

    # Optional parameters
    p.add_argument('-f', '--force',  action='store_true', help='overwrite evaluation results')

    return p


def compute_NLL(model, dataset, batch_size, batch_id, ordering_id, seed):
    status = Status()
    batch_scheduler = BatchSchedulerWithAutoregressiveMasks(dataset,
                                                            batch_size=batch_size,
                                                            batch_id=batch_id,
                                                            ordering_id=ordering_id,
                                                            use_mask_as_input=model.nb_channels == 2,
                                                            seed=seed)
    loss = NllUsingBinaryCrossEntropyWithAutoRegressiveMask(model, dataset, mod=batch_scheduler.mod)

    nll = views.LossView(loss=loss, batch_scheduler=batch_scheduler)
    nlls_xod_given_xoltd = nll.losses.view(Status())
    nlls = np.sum(nlls_xod_given_xoltd.reshape(-1, batch_size), axis=0)
    return nlls


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
        # print(" (data: {:,}; {:,}; {:,}) ".format(len(trainset), len(validset), len(testset)), end="")

    # Result files.
    evaluation_dir = smartutils.create_folder(pjoin(args.experiment, "evaluation"))
    evaluation_file = pjoin(evaluation_dir, "{}_batch{}_ordering{}.npz".format(args.subset, args.batch_id, args.ordering_id))

    if not os.path.isfile(evaluation_file) or args.force:
        with Timer("Computing exact NLL on {} for batch #{} and ordering #{}".format(args.subset, args.batch_id, args.ordering_id)):
            dataset = validset
            if args.subset == "testset":
                dataset = testset

            nlls = compute_NLL(model, dataset, args.batch_size, args.batch_id, args.ordering_id, args.seed)
            results = {"nlls": nlls,
                       "subset": args.subset,
                       "batch_id": args.batch_id,
                       "nb_batches": int(np.ceil(len(dataset)/float(args.batch_size))),
                       "ordering_id": args.ordering_id,
                       "nb_orderings": args.nb_orderings,
                       "batch_size": args.batch_size,
                       "seed": args.seed}
            np.savez(evaluation_file, **results)

    else:
        print("Loading saved losses... (use --force to re-run evaluation)")
        nlls = np.load(evaluation_file)['nlls']

    avg_nlls = nlls.mean()
    stderror_nlls = nlls.std(ddof=1) / np.sqrt(len(nlls))

    print("NLL on {} for batch #{} and ordering #{}: {:.2f} ± {:.2f}".format(args.subset, args.batch_id, args.ordering_id, avg_nlls, stderror_nlls))

if __name__ == '__main__':
    main()
