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

from convnade import utils
from convnade import datasets
from convnade.utils import Timer

from convnade.batch_schedulers import MiniBatchSchedulerWithAutoregressiveMask
from convnade.losses import BinaryCrossEntropyEstimateWithAutoRegressiveMask

# from smartpy.misc import utils
# from smartpy.misc.dataset import UnsupervisedDataset as Dataset
# from smartpy.misc.utils import load_dict_from_json_file

# from smartpy.models.convolutional_deepnade import generate_blueprints, DeepConvNADEBuilder
# from smartpy.models.convolutional_deepnade import EvaluateDeepNadeNLLParallel


DATASETS = ['binarized_mnist']


def build_argparser():
    DESCRIPTION = "Evaluate a ConvNADE model."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', type=str, help='name/path of the experiment.')

    # p.add_argument('--batch_size', type=int, help='size of the batch to use when evaluating the model.', default=64)
    # p.add_argument('--subset', type=str, choices=['valid', 'test'], help='evaluate only a specific subset (either "testset" or "validset") {0}]. Default: evaluate both subsets.')
    # p.add_argument('--part', metavar="<no_part>/<nb_parts>", type=str, help='evaluate only a specific part of the dataset (e.g. 5/10). Default: evaluate the whole dataset.')
    # p.add_argument('--ordering', type=int, help='evaluate only a specific input ordering. Default: evaluate all orderings.')
    # p.add_argument('--nb_orderings', type=int, help='evaluate that many input orderings. Default: 8.', default=8)
    # p.add_argument('--ordering_seed', type=int, help='evaluate random ordering(s) generate using this seed. Default: will use the 8 trivial orderings.')

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

def estimate_NLL(model, dataset, seed=1234):
    loss = BinaryCrossEntropyEstimateWithAutoRegressiveMask(model, dataset)
    status = Status()

    batch_scheduler = MiniBatchSchedulerWithAutoregressiveMask(dataset,
                                                               batch_size=len(dataset),
                                                               concatenate_mask=model.nb_channels == 2,
                                                               keep_mask=True,
                                                               seed=seed)

    nll = views.LossView(loss=loss, batch_scheduler=batch_scheduler)

    # Try different size of batch size.
    batch_size = len(dataset)
    while batch_size >= 1:
        print("Estimating NLL using batch size of {}".format(batch_size))
        try:
            batch_scheduler.batch_size = batch_size
            return float(nll.mean.view(status)), float(nll.stderror.view(status))

        except MemoryError as e:
            # Probably not enough memory on GPU
            #print("\n".join([l for l in str(e).split("\n") if "allocating" in l]))
            pass

        print("*An error occured while estimating NLL. Will try a smaller batch size.")
        batch_size = batch_size // 2

    raise RuntimeError("Cannot find a suitable batch size to estimate NLL. Try using CPU instead or a GPU with more memory.")


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

    if not os.path.isdir(pjoin(experiment_path, "model")):
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
        model = model_class.create(pjoin(experiment_path, "model"))  # Create new instance
        model.load(pjoin(experiment_path, "model"))  # Restore state.

        print(str(model.convnet))
        print(str(model.fullnet))

    # Result files.
    result_file = pjoin(experiment_path, "results.json")

    if not os.path.isfile(result_file) or args.force:
        with Timer("Evaluating NLL estimate"):
            results = {}
            results['NLL_est._trainset'] = estimate_NLL(model, trainset, seed=args.seed)
            results['NLL_est._validset'] = estimate_NLL(model, validset, seed=args.seed)
            results['NLL_est._testset'] = estimate_NLL(model, testset, seed=args.seed)
            utils.save_dict_to_json_file(result_file, results)
    else:
        print("Loading saved results... (use --force to re-run evaluation)")
        results = utils.load_dict_from_json_file(result_file)

    print("NLL estimate on trainset: {:.2f} ± {:.2f}".format(*results['NLL_est._trainset']))
    print("NLL estimate on validset: {:.2f} ± {:.2f}".format(*results['NLL_est._validset']))
    print("NLL estimate on testset:  {:.2f} ± {:.2f}".format(*results['NLL_est._testset']))

if __name__ == '__main__':
    main()


# def load_model(args):
#     with utils.Timer("Loading model"):
#         hyperparams = load_dict_from_json_file(pjoin(args.experiment, "hyperparams.json"))

#         image_shape = tuple(hyperparams["image_shape"])
#         hidden_activation = [v for k, v in hyperparams.items() if "activation" in k][0]
#         builder = DeepConvNADEBuilder(image_shape=image_shape,
#                                       nb_channels=hyperparams["nb_channels"],
#                                       ordering_seed=hyperparams["ordering_seed"],
#                                       consider_mask_as_channel=hyperparams["consider_mask_as_channel"],
#                                       hidden_activation=hidden_activation)

#         # Read infos from "command.pkl"

#         command = pickle.load(open(pjoin(args.experiment, "command.pkl")))

#         blueprint_seed = None
#         if "--blueprint_seed" in command:
#             blueprint_seed = int(command[command.index("--blueprint_seed") + 1])

#         convnet_blueprint = None
#         if "--convnet_blueprint" in command:
#             convnet_blueprint = int(command[command.index("--convnet_blueprint") + 1])

#         fullnet_blueprint = None
#         if "--fullnet_blueprint" in command:
#             fullnet_blueprint = int(command[command.index("--fullnet_blueprint") + 1])

#         if blueprint_seed is not None:
#             convnet_blueprint, fullnet_blueprint = generate_blueprints(blueprint_seed, image_shape[0])
#             builder.build_convnet_from_blueprint(convnet_blueprint)
#             builder.build_fullnet_from_blueprint(fullnet_blueprint)
#         else:
#             if convnet_blueprint is not None:
#                 builder.build_convnet_from_blueprint(convnet_blueprint)

#             if fullnet_blueprint is not None:
#                 builder.build_fullnet_from_blueprint(fullnet_blueprint)

#         print convnet_blueprint
#         print fullnet_blueprint
#         convnade = builder.build()
#         convnade.load(args.experiment)

#     return convnade


# def report(args):
#     load_model(args)  # Just so the model blueprint are printed.

#     evaluation_folder = pjoin(args.experiment, "evaluation")
#     evaluation_files = os.listdir(evaluation_folder)
#     _, nb_parts, _, nb_orderings = map(int, re.findall("part([0-9]+)of([0-9]+)_ordering([0-9]+)of([0-9]+)", evaluation_files[0])[0])

#     template = "{subset}_part{no_part}of{nb_parts}_ordering{no_ordering}of{nb_orderings}.npy"

#     # Check if we miss some evaluation results
#     orderings = {"validset": [], "testset": []}
#     nb_results_missing = 0
#     for subset in ["validset", "testset"]:
#         for no_ordering in range(nb_orderings):
#             is_missing_part = False
#             for no_part in range(1, nb_parts+1):
#                 name = template.format(subset=subset,
#                                        no_part=no_part, nb_parts=nb_parts,
#                                        no_ordering=no_ordering, nb_orderings=nb_orderings)

#                 if name not in evaluation_files:
#                     is_missing_part = True
#                     nb_results_missing += 1
#                     print "Missing: ", name

#             if not is_missing_part:
#                 orderings[subset].append(no_ordering)

#     if nb_results_missing > 0:
#         print "Missing {} result(s). Terminating...".format(nb_results_missing)
#         if not args.ignore:
#             return

#     # Merge results
#     def _nll_mean_stderr(subset):
#         nlls = []
#         # Examples have been split in multiple parts.
#         for no_part in range(1, nb_parts+1):
#             nlls_part = []
#             for no_ordering in orderings[subset]:
#                 name = template.format(subset=subset,
#                                        no_part=no_part, nb_parts=nb_parts,
#                                        no_ordering=no_ordering, nb_orderings=nb_orderings)

#                 # Load the NLLs for a given part and a given ordering.
#                 nlls_part.append(np.load(pjoin(evaluation_folder, name)))

#             # Average the probabilities across the orderings independently for each example.
#             nlls_part = -np.logaddexp.reduce(-np.array(nlls_part), axis=0)
#             nlls_part += np.log(len(orderings[subset]))  # Average across all orderings
#             nlls.append(nlls_part)

#         # Concatenate every part together.
#         nlls = np.hstack(nlls)
#         return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

#     def _nll_per_ordering(subset, no_ordering):
#         nlls = []
#         # Examples have been split in multiple parts.
#         for no_part in range(1, nb_parts+1):
#             name = template.format(subset=subset,
#                                    no_part=no_part, nb_parts=nb_parts,
#                                    no_ordering=no_ordering, nb_orderings=nb_orderings)

#             # Load the NLLs for a given part and a given ordering.
#             nlls_part = np.load(pjoin(evaluation_folder, name))
#             nlls.append(nlls_part)

#         # Concatenate every part together.
#         nlls = np.hstack(nlls)
#         return nlls.mean()

#     # Compute NLL mean and NLL stderror on validset and testset.
#     validset_mean, validset_stderr = _nll_mean_stderr("validset")
#     print "\nUsing {} orderings.".format(len(orderings["validset"]))
#     print "Validation NLL:", validset_mean
#     print "Validation NLL std:", validset_stderr
#     if args.verbose:
#         nlls_mean_per_ordering = map(lambda o: _nll_per_ordering("validset", o), orderings["validset"])
#         print "NLLs mean per ordering (validset):"
#         print "\n".join(["#{}: {}".format(o, nll_mean) for nll_mean, o in zip(nlls_mean_per_ordering, orderings["validset"])])

#     testset_mean, testset_stderr = _nll_mean_stderr("testset")
#     print "\nUsing {} orderings.".format(len(orderings["testset"]))
#     print "Testing NLL:", testset_mean
#     print "Testing NLL std:", testset_stderr
#     if args.verbose:
#         nlls_mean_per_ordering = map(lambda o: _nll_per_ordering("testset", o), orderings["testset"])
#         print "NLLs mean per ordering (testset):"
#         print "\n".join(["#{}: {}".format(o, nll_mean) for nll_mean, o in zip(nlls_mean_per_ordering, orderings["testset"])])

