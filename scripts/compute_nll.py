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

from convnade.batch_schedulers import BatchSchedulerWithAutoregressiveMasks
from convnade.losses import NllUsingBinaryCrossEntropyWithAutoRegressiveMask


DATASETS = ['binarized_mnist']


def build_argparser():
    DESCRIPTION = "Evaluate the exact NLL of a ConvNADE model."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', type=str, help='name/path of the experiment.')
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


# def evaluate(args):
#     evaluation_folder = pjoin(args.experiment, "evaluation")
#     if not os.path.isdir(evaluation_folder):
#         os.mkdir(evaluation_folder)

#     no_part, nb_parts = 1, 1
#     if args.part is not None:
#         no_part, nb_parts = map(int, args.part.split("/"))

#     def _compute_nll(subset):
#         #eval model_name/ eval --dataset testset --part [1:11]/10 --ordering [0:8]
#         nll_evaluation = EvaluateDeepNadeNLLParallel(convnade, subset,
#                                                      batch_size=args.batch_size, no_part=no_part, nb_parts=nb_parts,
#                                                      no_ordering=args.ordering, nb_orderings=args.nb_orderings, orderings_seed=args.orderings_seed)
#         nlls = nll_evaluation.view()

#         # Save [partial] evaluation results.
#         name = "{subset}_part{no_part}of{nb_parts}_ordering{no_ordering}of{nb_orderings}"
#         name = name.format(subset=subset,
#                            no_part=no_part,
#                            nb_parts=nb_parts,
#                            no_ordering=args.ordering,
#                            nb_orderings=args.nb_orderings)
#         filename = pjoin(evaluation_folder, name + ".npy")
#         np.save(filename, nlls)

#     if args.subset == "valid" or args.subset is None:
#         _compute_nll(dataset.validset_shared)

#     if args.subset == "test" or args.subset is None:
#         _compute_nll(dataset.testset_shared)

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


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Get experiment folder
    experiment_path = args.name
    if not os.path.isdir(experiment_path):
        # If not a directory, it must be the name of the experiment.
        experiment_path = pjoin(".", "experiments", args.name)

    if not os.path.isdir(experiment_path):
        parser.error('Cannot find experiment: {0}!'.format(args.name))

    if not os.path.isdir(pjoin(experiment_path, "DeepConvNADE")):
        parser.error('Cannot find model for experiment: {0}!'.format(experiment_path))

    if not os.path.isfile(pjoin(experiment_path, "hyperparams.json")):
        parser.error('Cannot find hyperparams for experiment: {0}!'.format(experiment_path))

    # Load experiments hyperparameters
    hyperparams = utils.load_dict_from_json_file(pjoin(experiment_path, "hyperparams.json"))

    with Timer("Loading dataset"):
        trainset, validset, testset = datasets.load(hyperparams['dataset'], keep_on_cpu=True)
        print(" (data: {:,}; {:,}; {:,}) ".format(len(trainset), len(validset), len(testset)), end="")

    with Timer("Loading model"):
        if hyperparams["model"] == "convnade":
            from convnade import DeepConvNADE
            model_class = DeepConvNADE

        # Load the actual model.
        model = model_class.create(experiment_path)  # Create new instance
        model.load(experiment_path)  # Restore state.
        print(str(model))

    # Result files.
    evaluation_dir = smartutils.create_folder(pjoin(experiment_path, "evaluation"))
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
                       "batch_size": batch_size,
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
