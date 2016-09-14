#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this  script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import theano
import theano.tensor as T
import numpy as np

import tempfile
from numpy.testing import assert_equal, assert_almost_equal, assert_array_equal, assert_array_almost_equal

import smartlearner.initializers as initer
from smartlearner import Trainer, Dataset, Model
from smartlearner import tasks
from smartlearner.status import Status
from smartlearner import views
from smartlearner import stopping_criteria

import smartlearner.initializers as initer
from smartlearner.utils import sharedX
from smartlearner.optimizers import SGD
from smartlearner.direction_modifiers import ConstantLearningRate
#from smartlearner.batch_schedulers import MiniBatchScheduler, FullBatchScheduler
#from smartlearner.losses.classification_losses import NegativeLogLikelihood as NLL
#from smartlearner.losses.classification_losses import ClassificationError

from convnade.utils import Timer, cartesian
from convnade.datasets import ReconstructionDataset, load_binarized_mnist

from convnade import DeepConvNADE, DeepConvNADEBuilder
from convnade import generate_blueprints
#from convnade.tasks import DeepNadeOrderingTask
from convnade.batch_schedulers import MiniBatchSchedulerWithAutoregressiveMask, BatchSchedulerWithAutoregressiveMasks
from convnade.losses import BinaryCrossEntropyEstimateWithAutoRegressiveMask
from convnade.losses import NllUsingBinaryCrossEntropyWithAutoRegressiveMask

np.set_printoptions(linewidth=220)


def test_simple_convnade():
    nb_kernels = 8
    kernel_shape = (2, 2)
    hidden_activation = "sigmoid"
    consider_mask_as_channel = True
    batch_size = 1024
    ordering_seed = 1234
    max_epoch = 3
    nb_orderings = 1

    print("Will train Convoluational Deep NADE for a total of {0} epochs.".format(max_epoch))

    with Timer("Loading/processing binarized MNIST"):
        trainset, validset, testset = load_binarized_mnist()

        # Extract the center patch (4x4 pixels) of each image.
        indices_to_keep = [348, 349, 350, 351, 376, 377, 378, 379, 404, 405, 406, 407, 432, 433, 434, 435]

        trainset = Dataset(trainset.inputs.get_value()[:, indices_to_keep], trainset.inputs.get_value()[:, indices_to_keep], name="trainset")
        validset = Dataset(validset.inputs.get_value()[:, indices_to_keep], validset.inputs.get_value()[:, indices_to_keep], name="validset")
        testset = Dataset(testset.inputs.get_value()[:, indices_to_keep], testset.inputs.get_value()[:, indices_to_keep], name="testset")

        image_shape = (4, 4)
        nb_channels = 1

    with Timer("Building model"):
        builder = DeepConvNADEBuilder(image_shape=image_shape,
                                      nb_channels=nb_channels,
                                      consider_mask_as_channel=True)

        convnet_blueprint = "64@2x2(valid) -> 1@2x2(full)"
        fullnet_blueprint = "5 -> 16"
        print("Convnet:", convnet_blueprint)
        print("Fullnet:", fullnet_blueprint)
        builder.build_convnet_from_blueprint(convnet_blueprint)
        builder.build_fullnet_from_blueprint(fullnet_blueprint)

        model = builder.build()
        model.initialize()  # By default, uniform initialization.

    with Timer("Building optimizer"):
        loss = BinaryCrossEntropyEstimateWithAutoRegressiveMask(model, trainset)

        optimizer = SGD(loss=loss)
        optimizer.append_direction_modifier(ConstantLearningRate(0.001))

    with Timer("Building trainer"):
        batch_scheduler = MiniBatchSchedulerWithAutoregressiveMask(trainset, batch_size)

        trainer = Trainer(optimizer, batch_scheduler)

        trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))

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
        nll = views.LossView(loss=BinaryCrossEntropyEstimateWithAutoRegressiveMask(model, validset),
                             batch_scheduler=MiniBatchSchedulerWithAutoregressiveMask(validset, batch_size=len(validset)))
        trainer.append_task(tasks.Print("Validset - NLL          : {0:.2f} ± {1:.2f}", nll.mean, nll.stderror))

        trainer.build_theano_graph()

    with Timer("Training"):
        trainer.train()

    with Timer("Checking the probs for all possible inputs sum to 1"):
        # rng = np.random.RandomState(ordering_seed)
        D = np.prod(image_shape)

        batch_scheduler = BatchSchedulerWithAutoregressiveMasks(validset,
                                                                batch_size=len(validset),
                                                                batch_id=0,
                                                                ordering_id=0,
                                                                concatenate_mask=model.nb_channels == 2,
                                                                seed=42)
        nll = views.LossView(loss=NllUsingBinaryCrossEntropyWithAutoRegressiveMask(model, validset, batch_scheduler.mod),
                             batch_scheduler=batch_scheduler)
        nlls_xod_given_xoltd = nll.losses.view(Status())
        nlls = np.sum(nlls_xod_given_xoltd.reshape(-1, len(validset)), axis=0)
        nll_validset = np.mean(nlls)
        print("Sum of NLL for validset:", nll_validset)

        inputs = cartesian([[0, 1]]*int(D), dtype=np.float32)
        dataset = ReconstructionDataset(inputs)
        batch_scheduler = BatchSchedulerWithAutoregressiveMasks(dataset,
                                                                batch_size=len(dataset),
                                                                batch_id=0,
                                                                ordering_id=0,
                                                                concatenate_mask=model.nb_channels == 2,
                                                                seed=42)
        nll = views.LossView(loss=NllUsingBinaryCrossEntropyWithAutoRegressiveMask(model, dataset, batch_scheduler.mod),
                             batch_scheduler=batch_scheduler)
        nlls_xod_given_xoltd = nll.losses.view(Status())
        nlls = np.sum(nlls_xod_given_xoltd.reshape(-1, len(dataset)), axis=0)
        p_x = np.exp(np.logaddexp.reduce(-nlls))
        print("Sum of p(x) for all x:", p_x)
        assert_almost_equal(p_x, 1., decimal=5)


        # symb_input = T.vector("input")
        # symb_input.tag.test_value = inputs[-len(inputs)//4]
        # symb_ordering = T.ivector("ordering")
        # symb_ordering.tag.test_value = ordering
        # nll_of_x_given_o = theano.function([symb_input, symb_ordering], model.nll_of_x_given_o(symb_input, symb_ordering), name="nll_of_x_given_o")
        # #theano.printing.pydotprint(nll_of_x_given_o, '{0}_nll_of_x_given_o_{1}'.format(model.__class__.__name__, theano.config.device), with_ids=True)

        # for i in range(nb_orderings):
        #     print ("Ordering:", ordering)
        #     ordering = np.arange(D, dtype=np.int32)
        #     rng.shuffle(ordering)

        #     nlls = []
        #     for no, input in enumerate(inputs):
        #         print("{}/{}".format(no, len(inputs)), end='\r')
        #         nlls.append(nll_of_x_given_o(input, ordering))

        #     print("{}/{} Done".format(len(inputs), len(inputs)))

        #     p_x = np.exp(np.logaddexp.reduce(-np.array(nlls)))
        #     print("Sum of p(x) for all x:", p_x)
        #     assert_almost_equal(p_x, 1., decimal=5)
