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


def build_argparser():
    DESCRIPTION = "Merge NLL evaluations."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', type=str, help='name/path of the experiment.')

    p.add_argument('--ignore-missing-batches', action="store_true",
                   help='ignore batches that are missing.')
    p.add_argument('--ignore-missing-orderings', action="store_true",
                   help='ignore orderings that are missing.')
    p.add_argument('--subset', type=str, choices=['valid', 'test'],
                   help='merge nll evaluations only for a specific subset (either "testset" or "validset") {0}]. Default: evaluate both subsets.')

    # Optional parameters
    p.add_argument('-f', '--force',  action='store_true', help='overwrite evaluation results')

    return p


def merge_NLL_evaluations(evaluation_dir):
    evaluation_files = os.listdir(evaluation_dir)
    first_evaluation = np.load(pjoin(evaluation_dir, evaluation_files[0]))
    nb_batches = first_evaluation['nb_batches']
    nb_orderings = first_evaluation['nb_orderings']

    results = {"nb_batches": int(nb_batches),
               "nb_orderings": int(nb_orderings),
               "batch_size": int(first_evaluation['batch_size']),
               "seed": int(first_evaluation['seed'])}

    for subset in ["validset", "testset"]:
        nlls = []
        incomplete = False
        for batch_id in range(nb_batches):
            batch_nlls = []
            for ordering_id in range(nb_orderings):
                evaluation_file = "{}_batch{}_ordering{}.npz".format(subset, batch_id, ordering_id)
                if evaluation_file not in evaluation_files:
                    print("Missing '{}'".format(evaluation_file))
                    incomplete = True
                    continue

                evaluation = np.load(pjoin(evaluation_dir, evaluation_file))
                batch_nlls.append(evaluation['nlls'])

            if len(batch_nlls) == 0:
                continue

            # Average the probabilities across the orderings independently for each example.
            batch_nlls = -np.logaddexp.reduce(-np.array(batch_nlls), axis=0)
            batch_nlls += np.log(nb_orderings)  # Average across all orderings
            nlls.append(batch_nlls)

        if len(batch_nlls) == 0:
            print("Incomplete dataset evaluations: {}".format(subset))
            results[subset] = {'mean': float(-1),
                               'stderror': float(-1),
                               'incomplete': incomplete}
            continue

        nlls = np.concatenate(nlls)
        results[subset] = {'mean': float(nlls.mean()),
                           'stderror': float(nlls.std(ddof=1) / np.sqrt(len(nlls))),
                           'incomplete': incomplete}

    return results


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

    if not os.path.isdir(pjoin(experiment_path, "evaluation")):
        parser.error('Cannot find evaluations for experiment: {0}!'.format(experiment_path))

    results_file = pjoin(experiment_path, "results.json")

    if not os.path.isfile(results_file) or args.force:
        with Timer("Merging NLL evaluations"):
            results = merge_NLL_evaluations(evaluation_dir=pjoin(experiment_path, "evaluation"))
            smartutils.save_dict_to_json_file(results_file, {"NLL": results})

    else:
        print("Loading saved losses... (use --force to re-run evaluation)")
        results = smartutils.load_dict_from_json_file(results_file)["NLL"]

    nb_orderings = results['nb_orderings']
    for dataset in ['validset', 'testset']:
        print("NLL estimate on {} ({} orderings): {:.2f} ± {:.2f}".format(dataset, nb_orderings, results[dataset]['mean'], results[dataset]['stderror']))

        if results[dataset]['incomplete']:
            print("** Warning **: {} evaluation is incomplete. Missing some orderings or batches.".format(dataset))

if __name__ == '__main__':
    main()
