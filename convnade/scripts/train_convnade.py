#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

from os.path import join as pjoin

import shutil
import argparse
import datetime
import theano.tensor as T

import pickle
import numpy as np

from smartlearner import Trainer
from smartlearner import tasks
from smartlearner import views
from smartlearner import stopping_criteria
import smartlearner.utils as smartutils

from convnade import utils
from convnade import datasets

from convnade.utils import Timer
from convnade.factories import WEIGHTS_INITIALIZERS, weigths_initializer_factory
from convnade.factories import ACTIVATION_FUNCTIONS
from convnade.factories import optimizer_factory

from convnade import DeepConvNADEBuilder, DeepConvNADEWithResidual, DeepConvNadeUsingLasagne, DeepConvNadeWithResidualUsingLasagne

from convnade.batch_schedulers import MiniBatchSchedulerWithAutoregressiveMask
from convnade.losses import BinaryCrossEntropyEstimateWithAutoRegressiveMask


# from smartpy.misc import utils
# from smartpy.misc.dataset import UnsupervisedDataset as Dataset

# from smartpy import optimizers

# from smartpy import update_rules
# from smartpy.optimizers import OPTIMIZERS

# from smartpy.trainers.trainer import Trainer
# from smartpy.trainers import tasks


# from smartpy.models.convolutional_deepnade import DeepConvNADEBuilder

# from smartpy.models.convolutional_deepnade import DeepNadeOrderingTask, DeepNadeTrivialOrderingsTask
# from smartpy.models.convolutional_deepnade import EvaluateDeepNadeNLL, EvaluateDeepNadeNLLEstimate
# from smartpy.models.convolutional_deepnade import EvaluateDeepNadeNLLEstimateOnTrivial

DATASETS = ["binarized_mnist", "caltech101_silhouettes28"]
MODELS = ['convnade', 'convnade-recipe', 'nade']


def generate_blueprints(seed, image_shape):
    rng = np.random.RandomState(seed)

    # Generate convoluational layers blueprint
    convnet_blueprint = []
    convnet_blueprint_inverse = []  # We want convnet to be symmetrical
    nb_layers = rng.randint(1, 5+1)
    layer_id_first_conv = -1
    for layer_id in range(nb_layers):
        if image_shape <= 2:
            # Too small
            continue

        if rng.rand() <= 0.8:
            # 70% of the time do a convolution
            nb_filters = rng.choice([16, 32, 64, 128, 256, 512])
            filter_shape = rng.randint(2, min(image_shape, 8+1))
            image_shape = image_shape-filter_shape+1

            filter_shape = (filter_shape, filter_shape)
            convnet_blueprint.append("{nb_filters}@{filter_shape}(valid)".format(nb_filters=nb_filters,
                                                                                 filter_shape="x".join(map(str, filter_shape))))
            convnet_blueprint_inverse.append("{nb_filters}@{filter_shape}(full)".format(nb_filters=nb_filters,
                                                                                        filter_shape="x".join(map(str, filter_shape))))
            if layer_id_first_conv == -1:
                layer_id_first_conv = layer_id
        else:
            # 30% of the time do a max pooling
            pooling_shape = rng.randint(2, 5+1)
            while not image_shape % pooling_shape == 0:
                pooling_shape = rng.randint(2, 5+1)

            image_shape = image_shape / pooling_shape
            #pooling_shape = 2  # For now, we limit ourselves to pooling of 2x2
            pooling_shape = (pooling_shape, pooling_shape)
            convnet_blueprint.append("max@{pooling_shape}".format(pooling_shape="x".join(map(str, pooling_shape))))
            convnet_blueprint_inverse.append("up@{pooling_shape}".format(pooling_shape="x".join(map(str, pooling_shape))))

    # Need to make sure there is only one channel in output
    infos = convnet_blueprint_inverse[layer_id_first_conv].split("@")[-1]
    convnet_blueprint_inverse[layer_id_first_conv] = "1@" + infos

    # Connect first part and second part of the convnet
    convnet_blueprint = "->".join(convnet_blueprint) + "->" + "->".join(convnet_blueprint_inverse[::-1])

    # Generate fully connected layers blueprint
    fullnet_blueprint = []
    nb_layers = rng.randint(1, 4+1)  # Deep NADE only used up to 4 hidden layers
    for layer_id in range(nb_layers):
        hidden_size = 500  # Deep NADE only used hidden layer of 500 units
        fullnet_blueprint.append("{hidden_size}".format(hidden_size=hidden_size))

    fullnet_blueprint.append("784")  # Output layer
    fullnet_blueprint = "->".join(fullnet_blueprint)

    return convnet_blueprint, fullnet_blueprint


# def buildArgsParser():
#     p.add_argument('--exact_inference', action='store_true', help='Compute the exact NLL on the validset and testset (slower)')
#     p.add_argument('--ensemble', type=int, help='Size of the ensemble. Default=1', default=1)
#     return p

def build_train_convnade_argparser(subparser):
    DESCRIPTION = "Train a Convolutional Deep NADE."

    p = subparser.add_parser("convnade", description=DESCRIPTION, help=DESCRIPTION)

    p.add_argument('dataset', type=str, choices=DATASETS, metavar="DATASET",
                   help='dataset to train on [{0}].'.format(', '.join(DATASETS))),

    # Model options (convnade)
    model = p.add_argument_group("ConvNADE arguments")

    model.add_argument('--convnet-blueprint', type=str, help='blueprint of the convolutional layers e.g. "64@3x3(valid)->32@7x7(full)".')
    model.add_argument('--fullnet-blueprint', type=str, help='blueprint of the fully connected layers e.g. "500->784".')
    model.add_argument('--blueprints-seed', type=int, help='seed used to generate random blueprints.')
    model.add_argument('--ordering-seed', type=int, help='seed used to generate new ordering. Default=1234', default=1234)
    model.add_argument('--use-mask-as-input', action='store_true',
                       help='if specified, concatenate the ordering mask $o_{<d}$ to the input. In the convolutional part this translates to adding a new channel.')
    # model.add_argument('--finetune_on_trivial_orderings', action='store_true', help='finetune model using the 8 trivial orderings.')
    model.add_argument('--with-residual', action='store_true',
                       help='if specified, train on residuals.')

    model.add_argument('--hidden-activation', type=str, choices=ACTIVATION_FUNCTIONS, default=ACTIVATION_FUNCTIONS[0],
                       help="Activation functions: {}".format(ACTIVATION_FUNCTIONS),)
    model.add_argument('--weights-initialization', type=str, default=WEIGHTS_INITIALIZERS[0], choices=WEIGHTS_INITIALIZERS,
                       help='which type of initialization to use when creating weights [{0}].'.format(", ".join(WEIGHTS_INITIALIZERS)))
    model.add_argument('--initialization-seed', type=int, default=1234,
                       help='seed used to generate random numbers. Default=1234')

    # General parameters (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')
    general.add_argument('--use-lasagne', action='store_true', help='if specified, the DeepConvNadeUsingLasagne model will be used.')


def buildArgsParser():
    DESCRIPTION = ("Script to train a Convolutional Deep NADE model on a dataset"
                   " (binarized MNIST or CalTech101 Silhouettes) using Theano.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    duration = p.add_argument_group("Training duration options")
    duration.add_argument('--max-epoch', type=int, metavar='N', help='if specified, train for a maximum of N epochs.')
    duration.add_argument('--lookahead', type=int, metavar='K', default=10,
                          help='use early stopping with a lookahead of K. Default: 10')
    duration.add_argument('--lookahead-eps', type=float, default=1e-3,
                          help='in early stopping, an improvement is whenever the objective improve of at least `eps`. Default: 1e-3',)

    # Training options
    training = p.add_argument_group("Training options")
    training.add_argument('--batch-size', type=int, metavar="M",
                          help='size of the batch to use when training the model. Default: 64.', default=64)
    training.add_argument('--batch-norm', action='store_true', help='if specified, batch normalization will be used.')

    # Optimizer options
    optimizer = p.add_argument_group("Optimizer (required)")
    optimizer = optimizer.add_mutually_exclusive_group(required=True)
    optimizer.add_argument('--SGD', metavar="LR", type=str, help='use SGD with constant learning rate for training.')
    optimizer.add_argument('--AdaGrad', metavar="LR [EPS=1e-6]", type=str, help='use AdaGrad for training.')
    optimizer.add_argument('--Adam', metavar="[LR=0.0001]", type=str, help='use Adam for training.')
    optimizer.add_argument('--RMSProp', metavar="LR", type=str, help='use RMSProp for training.')
    optimizer.add_argument('--Adadelta', action="store_true", help='use Adadelta for training.')

    # General options (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('--name', type=str,
                         help='name of the experiment. Default: name is generated from arguments.')
    #general.add_argument('--seed', type=int,
    #                     help='seed used to generate random numbers. Default=1234.', default=1234)
    general.add_argument('--keep', type=int, metavar="K",
                         help='if specified, keep a copy of the model each K epoch.')

    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')

    subparser = p.add_subparsers(title="Models", dest="model")
    subparser.required = True   # force 'required' testing
    build_train_convnade_argparser(subparser)

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    # Extract experiments hyperparameters
    hyperparams = dict(vars(args))

    # Remove hyperparams that should not be part of the hash
    del hyperparams['max_epoch']
    del hyperparams['keep']
    del hyperparams['force']
    del hyperparams['name']

    # Get/generate experiment name
    experiment_name = args.name
    if experiment_name is None:
        experiment_name = utils.generate_uid_from_string(repr(hyperparams))

    # Create experiment folder
    experiment_path = pjoin(".", "experiments", experiment_name)
    resuming = False
    if os.path.isdir(experiment_path) and not args.force:
        resuming = True
        print("### Resuming experiment ({0}). ###\n".format(experiment_name))
        # Check if provided hyperparams match those in the experiment folder
        hyperparams_loaded = utils.load_dict_from_json_file(pjoin(experiment_path, "hyperparams.json"))
        if hyperparams != hyperparams_loaded:
            print("{\n" + "\n".join(["{}: {}".format(k, hyperparams[k]) for k in sorted(hyperparams.keys())]) + "\n}")
            print("{\n" + "\n".join(["{}: {}".format(k, hyperparams_loaded[k]) for k in sorted(hyperparams_loaded.keys())]) + "\n}")
            print("The arguments provided are different than the one saved. Use --force if you are certain.\nQuitting.")
            sys.exit(1)
    else:
        if os.path.isdir(experiment_path):
            shutil.rmtree(experiment_path)

        os.makedirs(experiment_path)
        utils.save_dict_to_json_file(pjoin(experiment_path, "hyperparams.json"), hyperparams)

    with Timer("Loading dataset"):
        trainset, validset, testset = datasets.load(args.dataset)

        image_shape = (28, 28)
        nb_channels = 1 + (args.use_mask_as_input is True)

        batch_scheduler = MiniBatchSchedulerWithAutoregressiveMask(trainset, args.batch_size,
                                                                   use_mask_as_input=args.use_mask_as_input,
                                                                   seed=args.ordering_seed)
        print("{} updates per epoch.".format(len(batch_scheduler)))

    with Timer("Building model"):
        if args.use_lasagne:
            if args.with_residual:
                model = DeepConvNadeWithResidualUsingLasagne(image_shape=image_shape,
                                                             nb_channels=nb_channels,
                                                             convnet_blueprint=args.convnet_blueprint,
                                                             fullnet_blueprint=args.fullnet_blueprint,
                                                             hidden_activation=args.hidden_activation,
                                                             use_mask_as_input=args.use_mask_as_input)
            else:
                model = DeepConvNadeUsingLasagne(image_shape=image_shape,
                                                 nb_channels=nb_channels,
                                                 convnet_blueprint=args.convnet_blueprint,
                                                 fullnet_blueprint=args.fullnet_blueprint,
                                                 hidden_activation=args.hidden_activation,
                                                 use_mask_as_input=args.use_mask_as_input,
                                                 use_batch_norm=args.batch_norm)

        elif args.with_residual:
            model = DeepConvNADEWithResidual(image_shape=image_shape,
                                             nb_channels=nb_channels,
                                             convnet_blueprint=args.convnet_blueprint,
                                             fullnet_blueprint=args.fullnet_blueprint,
                                             hidden_activation=args.hidden_activation,
                                             use_mask_as_input=args.use_mask_as_input)

        else:
            builder = DeepConvNADEBuilder(image_shape=image_shape,
                                          nb_channels=nb_channels,
                                          hidden_activation=args.hidden_activation,
                                          use_mask_as_input=args.use_mask_as_input)

            if args.blueprints_seed is not None:
                convnet_blueprint, fullnet_blueprint = generate_blueprints(args.blueprint_seed, image_shape[0])
                builder.build_convnet_from_blueprint(convnet_blueprint)
                builder.build_fullnet_from_blueprint(fullnet_blueprint)
            else:
                if args.convnet_blueprint is not None:
                    builder.build_convnet_from_blueprint(args.convnet_blueprint)

                if args.fullnet_blueprint is not None:
                    builder.build_fullnet_from_blueprint(args.fullnet_blueprint)

            model = builder.build()
            # print(str(model.convnet))
            # print(str(model.fullnet))

        model.initialize(weigths_initializer_factory(args.weights_initialization,
                                                     seed=args.initialization_seed))
        print(str(model))

    with Timer("Building optimizer"):
        loss = BinaryCrossEntropyEstimateWithAutoRegressiveMask(model, trainset)
        optimizer = optimizer_factory(hyperparams, loss)

    with Timer("Building trainer"):
        trainer = Trainer(optimizer, batch_scheduler)

        if args.max_epoch is not None:
            trainer.append_task(stopping_criteria.MaxEpochStopping(args.max_epoch))

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

        # Log training error
        loss_monitor = views.MonitorVariable(loss.loss)
        avg_loss = tasks.AveragePerEpoch(loss_monitor)
        accum = tasks.Accumulator(loss_monitor)
        logger = tasks.Logger(loss_monitor, avg_loss)
        trainer.append_task(logger, avg_loss, accum)

        # Print average training loss.
        trainer.append_task(tasks.Print("Avg. training loss:     : {}", avg_loss))

        # Print NLL mean/stderror.
        model.deterministic = True  # For batch normalization, see https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/normalization.py#L198
        nll = views.LossView(loss=BinaryCrossEntropyEstimateWithAutoRegressiveMask(model, validset),
                             batch_scheduler=MiniBatchSchedulerWithAutoregressiveMask(validset, batch_size=0.1*len(validset),
                                                                                      use_mask_as_input=args.use_mask_as_input,
                                                                                      keep_mask=True,
                                                                                      seed=args.ordering_seed+1))
        # trainer.append_task(tasks.Print("Validset - NLL          : {0:.2f} ± {1:.2f}", nll.mean, nll.stderror, each_k_update=100))
        trainer.append_task(tasks.Print("Validset - NLL          : {0:.2f} ± {1:.2f}", nll.mean, nll.stderror))

        # direction_norm = views.MonitorVariable(T.sqrt(sum(map(lambda d: T.sqr(d).sum(), loss.gradients.values()))))
        # trainer.append_task(tasks.Print("||d|| : {0:.4f}", direction_norm, each_k_update=50))

        # Save training progression
        def save_model(*args):
            trainer.save(experiment_path)

        trainer.append_task(stopping_criteria.EarlyStopping(nll.mean, lookahead=args.lookahead, eps=args.lookahead_eps, callback=save_model))

        trainer.build_theano_graph()

    if resuming:
        with Timer("Loading"):
            trainer.load(experiment_path)

    with Timer("Training"):
        trainer.train()

    trainer.save(experiment_path)
    model.save(experiment_path)


    #     # Add a task that changes the ordering mask
    #     trainer.add_task(ordering_task)

    #     # Print time for one epoch
    #     trainer.add_task(tasks.PrintEpochDuration())
    #     trainer.add_task(tasks.AverageObjective(trainer))

    #     if args.subcommand == "finetune":
    #         nll_valid = EvaluateDeepNadeNLLEstimateOnTrivial(model, dataset.validset_shared, batch_size=args.batch_size)
    #     else:
    #         nll_valid = EvaluateDeepNadeNLLEstimate(model, dataset.validset_shared, ordering_task.ordering_mask, batch_size=args.batch_size)

    #     trainer.add_task(tasks.Print(nll_valid.mean, msg="Average NLL estimate on the validset: {0}"))

    #     # Add stopping criteria
    #     if args.max_epoch is not None:
    #         # Stop when max number of epochs is reached.
    #         print "Will train Convoluational Deep NADE for a total of {0} epochs.".format(args.max_epoch)
    #         trainer.add_stopping_criterion(tasks.MaxEpochStopping(args.max_epoch))

    #     # Do early stopping bywatching the average NLL on the validset.
    #     if args.lookahead is not None:
    #         print "Will train Convoluational Deep NADE using early stopping with a lookahead of {0} epochs.".format(args.lookahead)
    #         save_task = tasks.SaveTraining(trainer, savedir=data_dir)
    #         early_stopping = tasks.EarlyStopping(nll_valid.mean, args.lookahead, save_task, eps=args.lookahead_eps, skip_epoch0=True)
    #         trainer.add_stopping_criterion(early_stopping)
    #         trainer.add_task(early_stopping)

    #     # Add a task to save the whole training process
    #     if args.save_frequency < np.inf:
    #         save_task = tasks.SaveTraining(trainer, savedir=data_dir, each_epoch=args.save_frequency)
    #         trainer.add_task(save_task)

    #     if args.subcommand == "resume" or args.subcommand == "finetune":
    #         print "Loading existing trainer..."
    #         trainer.load(data_dir)

    #     if args.subcommand == "finetune":
    #         trainer.status.extra['best_epoch'] = trainer.status.current_epoch

    #     trainer._build_learn()

    # if not args.no_train:
    #     with Timer("Training"):
    #         trainer.run()
    #         trainer.status.save(savedir=data_dir)

    #         if not args.lookahead:
    #             trainer.save(savedir=data_dir)

    # with Timer("Reporting"):
    #     # Evaluate model on train, valid and test sets
    #     nll_train = EvaluateDeepNadeNLLEstimate(model, dataset.trainset_shared, ordering_task.ordering_mask, batch_size=args.batch_size)
    #     nll_valid = EvaluateDeepNadeNLLEstimate(model, dataset.validset_shared, ordering_task.ordering_mask, batch_size=args.batch_size)
    #     nll_test = EvaluateDeepNadeNLLEstimate(model, dataset.testset_shared, ordering_task.ordering_mask, batch_size=args.batch_size)

    #     if args.exact_inference:
    #         nll_valid = EvaluateDeepNadeNLL(model, dataset.validset_shared, batch_size=args.batch_size, nb_orderings=args.ensemble)
    #         nll_test = EvaluateDeepNadeNLL(model, dataset.testset_shared, batch_size=args.batch_size, nb_orderings=args.ensemble)

    #     print "Training NLL - Estimate:", nll_train.mean.view(trainer.status)
    #     print "Training NLL std:", nll_train.std.view(trainer.status)
    #     print "Validation NLL - Estimate:", nll_valid.mean.view(trainer.status)
    #     print "Validation NLL std:", nll_valid.std.view(trainer.status)
    #     print "Testing NLL - Estimate:", nll_test.mean.view(trainer.status)
    #     print "Testing NLL std:", nll_test.std.view(trainer.status)

    #     from collections import OrderedDict
    #     log_entry = OrderedDict()
    #     log_entry["Convnet Blueprint"] = args.convnet_blueprint
    #     log_entry["Fullnet Blueprint"] = args.fullnet_blueprint
    #     log_entry["Mask as channel"] = model.hyperparams["use_mask_as_input"]
    #     log_entry["Activation Function"] = args.hidden_activation
    #     log_entry["Initialization Seed"] = args.initialization_seed
    #     log_entry["Best Epoch"] = trainer.status.extra["best_epoch"] if args.lookahead else trainer.status.current_epoch
    #     log_entry["Max Epoch"] = trainer.stopping_criteria[0].nb_epochs_max if args.max_epoch else ''

    #     if args.max_epoch:
    #         log_entry["Look Ahead"] = trainer.stopping_criteria[1].lookahead if args.lookahead else ''
    #         log_entry["Look Ahead eps"] = trainer.stopping_criteria[1].eps if args.lookahead else ''
    #     else:
    #         log_entry["Look Ahead"] = trainer.stopping_criteria[0].lookahead if args.lookahead else ''
    #         log_entry["Look Ahead eps"] = trainer.stopping_criteria[0].eps if args.lookahead else ''

    #     log_entry["Batch Size"] = trainer.optimizer.batch_size
    #     log_entry["Update Rule"] = trainer.optimizer.update_rules[0].__class__.__name__
    #     update_rule = trainer.optimizer.update_rules[0]
    #     log_entry["Learning Rate"] = "; ".join(["{0}={1}".format(name, getattr(update_rule, name)) for name in update_rule.__hyperparams__.keys()])

    #     log_entry["Weights Initialization"] = args.weights_initialization
    #     log_entry["Training NLL - Estimate"] = nll_train.mean
    #     log_entry["Training NLL std"] = nll_train.std
    #     log_entry["Validation NLL - Estimate"] = nll_valid.mean
    #     log_entry["Validation NLL std"] = nll_valid.std
    #     log_entry["Testing NLL - Estimate"] = nll_test.mean
    #     log_entry["Testing NLL std"] = nll_test.std
    #     log_entry["Training Time"] = trainer.status.training_time
    #     log_entry["Experiment"] = os.path.abspath(data_dir)

    #     formatting = {}
    #     formatting["Training NLL - Estimate"] = "{:.6f}"
    #     formatting["Training NLL std"] = "{:.6f}"
    #     formatting["Validation NLL - Estimate"] = "{:.6f}"
    #     formatting["Validation NLL std"] = "{:.6f}"
    #     formatting["Testing NLL - Estimate"] = "{:.6f}"
    #     formatting["Testing NLL std"] = "{:.6f}"
    #     formatting["Training Time"] = "{:.4f}"

    #     logging_task = tasks.LogResultCSV("results_{}_{}.csv".format("ConvDeepNADE", dataset.name), log_entry, formatting)
    #     logging_task.execute(trainer.status)

    #     if args.gsheet is not None:
    #         gsheet_id, gsheet_email, gsheet_password = args.gsheet.split()
    #         logging_task = tasks.LogResultGSheet(gsheet_id, gsheet_email, gsheet_password, "ConvDeepNADE", log_entry, formatting)
    #         logging_task.execute(trainer.status)

if __name__ == '__main__':
    main()
