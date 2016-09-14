#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os
import sys

# Hack so you don't have to put the library containing this  script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import argparse
import csv
import re
from collections import OrderedDict
# from texttable import Texttable

from os.path import join as pjoin

from smartlearner.utils import load_dict_from_json_file


DESCRIPTION = 'Gather experiments results and save them in a CSV file.'


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('names', type=str, nargs='+', help='name/path of the experiments.')
    p.add_argument('--out', default="results.csv", help='save table in a CSV file. Default: results.csv')
    p.add_argument('-v', '--verbose', action="store_true", help='verbose mode')
    return p


class Experiment(object):
    def __init__(self, experiment_path):
        self.experiment_path = experiment_path
        self.name = os.path.basename(self.experiment_path)
        self.results_file = pjoin(self.experiment_path, "results.json")
        self.results_estimate_file = pjoin(self.experiment_path, "results_estimate.json")
        self.hyperparams_file = pjoin(self.experiment_path, "hyperparams.json")
        # self.model_hyperparams_file = pjoin(self.experiment_path, "DeepConvNADE", "hyperparams.json")
        self.status_file = pjoin(self.experiment_path, "training", "status.json")
        self.early_stopping_file = pjoin(self.experiment_path, "training", "tasks", "early_stopping.json")

        empty_results = {"trainset": {"mean": "", "stderror": ""},
                         "validset": {"mean": "", "stderror": ""},
                         "testset": {"mean": "", "stderror": ""}}
        self.results = empty_results
        if os.path.isfile(self.results_file):
            self.results = load_dict_from_json_file(self.results_file)["NLL"]

        self.results_estimate = empty_results
        if os.path.isfile(self.results_estimate_file):
            self.results_estimate = load_dict_from_json_file(self.results_estimate_file)["NLL_estimate"]

        self.hyperparams = load_dict_from_json_file(self.hyperparams_file)
        # self.model_hyperparams = load_dict_from_json_file(self.model_hyperparams_file)
        self.status = load_dict_from_json_file(self.status_file)

        self.early_stopping = empty_results
        if os.path.isfile(self.early_stopping_file):
            self.early_stopping = load_dict_from_json_file(self.early_stopping_file)


def list_of_dict_to_csv_file(csv_file, list_of_dicts):
    keys = list_of_dicts[0].keys()
    with open(csv_file, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_of_dicts)


def get_optimizer(e):
        if e.hyperparams.get("SGD") is not None:
            return "SGD"
        elif e.hyperparams.get("AdaGrad") is not None:
            return "AdaGrad"
        elif e.hyperparams.get("ADAGRAD") is not None:
            return "ADAGRAD"
        elif e.hyperparams.get("Adam") is not None:
            return "Adam"

        return ""


def extract_result_from_experiment(e):
    """e: `Experiment` object"""
    entry = OrderedDict()
    entry["Convnet Blueprint"] = e.hyperparams.get("convnet_blueprint", "")
    entry["Fullnet Blueprint"] = e.hyperparams.get("fullnet_blueprint", "")
    # entry["Convnet Blueprint"] = e.model_hyperparams.get("convnet_blueprint", "")
    # entry["Fullnet Blueprint"] = e.model_hyperparams.get("fullnet_blueprint", "")

    entry["Concatenate Mask"] = e.hyperparams.get("use_mask_as_input", "")
    entry["Activation Function"] = e.hyperparams.get("hidden_activation", "")
    entry["Initialization Seed"] = e.hyperparams.get("initialization_seed", "")
    entry["Weights Initialization"] = e.hyperparams.get("weights_initialization", "")
    entry["Look Ahead"] = e.hyperparams.get("lookahead", "")
    entry["Look Ahead eps"] = e.hyperparams.get("lookahead_eps", "")
    entry["Batch Size"] = e.hyperparams.get("batch_size", "")
    entry["Optimizer"] = get_optimizer(e)
    entry["Optimizer params"] = e.hyperparams.get(get_optimizer(e), "")
    entry["Best Epoch"] = e.early_stopping.get("best_epoch", "")
    entry["Max Epoch"] = e.status.get("current_epoch", "")
    entry["Training est. NLL"] = e.results_estimate["trainset"]["mean"]
    entry["Training est. NLL std"] = e.results_estimate["trainset"]["stderror"]
    entry["Validation est. NLL"] = e.results_estimate["validset"]["mean"]
    entry["Validation est. NLL std"] = e.results_estimate["validset"]["stderror"]
    entry["Testing est. NLL"] = e.results_estimate["testset"]["mean"]
    entry["Testing est. NLL std"] = e.results_estimate["testset"]["stderror"]
    entry["Validation NLL"] = e.results["validset"]["mean"]
    entry["Validation NLL std"] = e.results["validset"]["stderror"]
    entry["Testing NLL"] = e.results["testset"]["mean"]
    entry["Testing NLL std"] = e.results["testset"]["stderror"]
    entry["Nb. Orderings"] = e.results.get("nb_orderings", "")
    entry["Training Time"] = e.status.get("training_time", "")
    entry["Experiment"] = e.name
    return entry


def main():
    parser = buildArgsParser()
    args = parser.parse_args()


    # # names = []
    # results_files = []
    # results_estimate_files = []
    # hyperparams_files = []
    # status_files = []

    experiments_results = []

    for experiment_path in args.names:
        try:
            experiment = Experiment(experiment_path)
            experiments_results.append(extract_result_from_experiment(experiment))
        except IOError as e:
            if args.verbose:
                print(str(e))

            print("Skipping: '{}'".format(experiment_path))

    list_of_dict_to_csv_file(args.out, experiments_results)


    #     results_file = pjoin(exp_folder, "results.json")
    #     results_estimate_file = pjoin(exp_folder, "results_estimate.json")
    #     hyperparams_file = pjoin(exp_folder, "hyperparams.json")
    #     status_file = pjoin(exp_folder, "training", "status.json")

    #     if not os.path.isfile(results_file) and not os.path.isfile(results_estimate_file):
    #         print('Skip: {0} and {1} are not files!'.format(results_file, results_estimate_file))
    #         continue

    #     if not os.path.isfile(hyperparams_file):
    #         print('Skip: {0} is not a file!'.format(hyperparams_file))
    #         continue

    #     if not os.path.isfile(status_file):
    #         print('Skip: {0} is not a file!'.format(status_file))
    #         continue

    #     name = os.path.abspath(exp_folder)
    #     while 'hyperparams.json' in os.listdir(os.path.abspath(pjoin(name, os.path.pardir))):
    #         name = os.path.abspath(pjoin(name, os.path.pardir))

    #     name = os.path.basename(name)
    #     names.append(name)

    #     results_files.append(results_file)
    #     results_estimate_files.append(results_estimate_file)
    #     hyperparams_files.append(hyperparams_file)
    #     status_files.append(status_file)

    # if len([no for no in sort_by if no == 0]) > 0:
    #     parser.error('Column ID are starting at 1!')

    # # Retrieve headers from hyperparams
    # headers_hyperparams = set()
    # headers_results = set()
    # headers_status = set()

    # for hyperparams_file, status_file, results_file, results_estimate_file in zip(hyperparams_files, status_files, results_files, results_estimate_files):
    #     hyperparams = load_dict_from_json_file(hyperparams_file)
    #     results = {"NLL":{}, "NLL_Estimate":{}}
    #     try:
    #         results["NLL"] = load_dict_from_json_file(results_file)
    #     except:
    #         pass

    #     try:
    #         results["NLL_Estimate"] = load_dict_from_json_file(results_estimate_file)
    #     except:
    #         pass

    #     status = load_dict_from_json_file(status_file)
    #     headers_hyperparams |= set(hyperparams.keys())
    #     headers_results |= set(["est._trainset", "est._trainset_std", "est._validset", "est._validset_std", "est._testset", "est._testset_std"])
    #     headers_results |= set(["nb_orderings", "validset", "validset_std", "testset", "testset_std"])
    #     headers_status |= set(["best_epoch"])

    # headers_hyperparams = sorted(list(headers_hyperparams))
    # headers_status = sorted(list(headers_status))
    # headers_results = sorted(list(headers_results))
    # headers = headers_hyperparams + headers_status + ["name"] + headers_results

    # # Build results table
    # table = Texttable(max_width=0)
    # table.set_deco(Texttable.HEADER)
    # table.set_precision(8)
    # table.set_cols_dtype(['a'] * len(headers))
    # table.set_cols_align(['c'] * len(headers))

    # # Headers
    # table.header([str(i) + "\n" + h for i, h in enumerate(headers, start=1)])

    # if args.only_header:
    #     print(table.draw())
    #     return

    # # Results
    # for name, hyperparams_file, status_file, results_file in zip(names, hyperparams_files, status_files, results_files):
    #     hyperparams = load_dict_from_json_file(hyperparams_file)
    #     results = load_dict_from_json_file(results_file)
    #     status = load_dict_from_json_file(status_file)

    #     # Build results table row (hyperparams columns)
    #     row = []
    #     for h in headers_hyperparams:
    #         value = hyperparams.get(h, '')
    #         row.append(value)

    #     for h in headers_status:
    #         value = status.get(h, '')
    #         row.append(value)

    #     row.append(name)

    #     for h in headers_results:
    #         if h in ["trainset", "validset", "testset"]:
    #             value = results.get(h, '')[0]
    #         elif h in ["trainset_std", "validset_std", "testset_std"]:
    #             value = results.get(h[:-4], '')[1]
    #         else:
    #             value = results.get(h, '')
    #         row.append(value)

    #     table.add_row(row)

    # # Sort
    # for col in reversed(sort_by):
    #     table._rows = sorted(table._rows, key=sort_nicely(abs(col) - 1), reverse=col < 0)

    # if args.out is not None:
    #     import csv

    #     results = []
    #     results.append(headers)
    #     results.extend(table._rows)

    #     with open(args.out, 'wb') as csvfile:
    #         w = csv.writer(csvfile)
    #         w.writerows(results)

    # else:
    #     print(table.draw())


if __name__ == "__main__":
    main()
