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
from convnade.datasets import load_binarized_mnist

from convnade import DeepConvNADE, DeepConvNADEBuilder
from convnade import generate_blueprints
#from convnade.tasks import DeepNadeOrderingTask
from convnade.batch_schedulers import MiniBatchSchedulerWithAutoregressiveMask
from convnade.losses import BinaryCrossEntropyEstimateWithAutoRegressiveMask

np.set_printoptions(linewidth=220)


def test_simple_convnade():
    nb_kernels = 8
    kernel_shape = (2, 2)
    hidden_activation = "sigmoid"
    use_mask_as_input = True
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
                                      use_mask_as_input=use_mask_as_input)

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
        rng = np.random.RandomState(ordering_seed)
        D = np.prod(image_shape)
        inputs = cartesian([[0, 1]]*int(D), dtype=np.float32)
        ordering = np.arange(D, dtype=np.int32)
        rng.shuffle(ordering)

        symb_input = T.vector("input")
        symb_input.tag.test_value = inputs[-len(inputs)//4]
        symb_ordering = T.ivector("ordering")
        symb_ordering.tag.test_value = ordering
        nll_of_x_given_o = theano.function([symb_input, symb_ordering], model.nll_of_x_given_o(symb_input, symb_ordering), name="nll_of_x_given_o")
        #theano.printing.pydotprint(nll_of_x_given_o, '{0}_nll_of_x_given_o_{1}'.format(model.__class__.__name__, theano.config.device), with_ids=True)

        for i in range(nb_orderings):
            print ("Ordering:", ordering)
            ordering = np.arange(D, dtype=np.int32)
            rng.shuffle(ordering)

            nlls = []
            for no, input in enumerate(inputs):
                print("{}/{}".format(no, len(inputs)), end='\r')
                nlls.append(nll_of_x_given_o(input, ordering))

            print("{}/{} Done".format(len(inputs), len(inputs)))

            p_x = np.exp(np.logaddexp.reduce(-np.array(nlls)))
            print("Sum of p(x) for all x:", p_x)
            assert_almost_equal(p_x, 1., decimal=5)


def test_convnade_with_max_pooling():
    nb_kernels = 8
    kernel_shape = (2, 2)
    hidden_activation = "sigmoid"
    use_mask_as_input = True
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
                                      use_mask_as_input=use_mask_as_input)

        convnet_blueprint = "64@3x3(valid) -> max@2x2 -> up@2x2 -> 1@3x3(full)"
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
        rng = np.random.RandomState(ordering_seed)
        D = np.prod(image_shape)
        inputs = cartesian([[0, 1]]*int(D), dtype=np.float32)
        ordering = np.arange(D, dtype=np.int32)
        rng.shuffle(ordering)

        symb_input = T.vector("input")
        symb_input.tag.test_value = inputs[-len(inputs)//4]
        symb_ordering = T.ivector("ordering")
        symb_ordering.tag.test_value = ordering
        nll_of_x_given_o = theano.function([symb_input, symb_ordering], model.nll_of_x_given_o(symb_input, symb_ordering), name="nll_of_x_given_o")
        #theano.printing.pydotprint(nll_of_x_given_o, '{0}_nll_of_x_given_o_{1}'.format(model.__class__.__name__, theano.config.device), with_ids=True)

        for i in range(nb_orderings):
            print ("Ordering:", ordering)
            ordering = np.arange(D, dtype=np.int32)
            rng.shuffle(ordering)

            nlls = []
            for no, input in enumerate(inputs):
                print("{}/{}".format(no, len(inputs)), end='\r')
                nlls.append(nll_of_x_given_o(input, ordering))

            print("{}/{} Done".format(len(inputs), len(inputs)))

            p_x = np.exp(np.logaddexp.reduce(-np.array(nlls)))
            print("Sum of p(x) for all x:", p_x)
            assert_almost_equal(p_x, 1., decimal=5)


def test_convnade_with_mask_as_input_channel():
    nb_kernels = 8
    kernel_shape = (2, 2)
    hidden_activation = "sigmoid"
    use_mask_as_input = True
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

        # We consider the mask as an input channel so we do the necessary modification to the datasets.
        nb_channels = 1 + (use_mask_as_input is True)
        batch_scheduler = MiniBatchSchedulerWithAutoregressiveMask(trainset, batch_size, use_mask_as_input=use_mask_as_input)

    with Timer("Building model"):
        builder = DeepConvNADEBuilder(image_shape=image_shape, nb_channels=nb_channels)

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
                             batch_scheduler=MiniBatchSchedulerWithAutoregressiveMask(validset, batch_size=len(validset), use_mask_as_input=use_mask_as_input))
        trainer.append_task(tasks.Print("Validset - NLL          : {0:.2f} ± {1:.2f}", nll.mean, nll.stderror))

        trainer.build_theano_graph()

    with Timer("Training"):
        trainer.train()

    with Timer("Checking the probs for all possible inputs sum to 1"):
        rng = np.random.RandomState(ordering_seed)
        D = np.prod(image_shape)
        inputs = cartesian([[0, 1]]*int(D), dtype=np.float32)
        ordering = np.arange(D, dtype=np.int32)
        rng.shuffle(ordering)

        d = rng.randint(D, size=(D, 1))
        masks_o_lt_d = np.arange(D) < d
        map(rng.shuffle, masks_o_lt_d)  # Inplace shuffling each row.

        symb_input = T.vector("input")
        symb_input.tag.test_value = inputs[-len(inputs)//4]
        symb_ordering = T.ivector("ordering")
        symb_ordering.tag.test_value = ordering
        nll_of_x_given_o = theano.function([symb_input, symb_ordering], model.nll_of_x_given_o(symb_input, symb_ordering), name="nll_of_x_given_o")
        #theano.printing.pydotprint(nll_of_x_given_o, '{0}_nll_of_x_given_o_{1}'.format(model.__class__.__name__, theano.config.device), with_ids=True)

        for i in range(nb_orderings):
            print ("Ordering:", ordering)
            ordering = np.arange(D, dtype=np.int32)
            rng.shuffle(ordering)

            nlls = []
            for no, input in enumerate(inputs):
                print("{}/{}".format(no, len(inputs)), end='\r')
                nlls.append(nll_of_x_given_o(input, ordering))

            print("{}/{} Done".format(len(inputs), len(inputs)))

            p_x = np.exp(np.logaddexp.reduce(-np.array(nlls)))
            print("Sum of p(x) for all x:", p_x)
            assert_almost_equal(p_x, 1., decimal=5)


def test_check_init():
    nb_kernels = 8
    kernel_shape = (2, 2)
    hidden_activation = "hinge"
    use_mask_as_input = True
    batch_size = 1024
    ordering_seed = 1234
    max_epoch = 5
    nb_orderings = 1

    with Timer("Loading/processing binarized MNIST"):
        trainset, validset, testset = load_binarized_mnist()

        # Extract the center patch (4x4 pixels) of each image.
        indices_to_keep = [348, 349, 350, 351, 376, 377, 378, 379, 404, 405, 406, 407, 432, 433, 434, 435]

        trainset = Dataset(trainset.inputs.get_value()[:, indices_to_keep], trainset.inputs.get_value()[:, indices_to_keep], name="trainset")
        validset = Dataset(validset.inputs.get_value()[:, indices_to_keep], validset.inputs.get_value()[:, indices_to_keep], name="validset")
        testset = Dataset(testset.inputs.get_value()[:, indices_to_keep], testset.inputs.get_value()[:, indices_to_keep], name="testset")

        image_shape = (4, 4)
        nb_channels = 1

    # Nested function to build a trainer.
    def _build_trainer(nb_epochs):
        print("Will train Convoluational Deep NADE for a total of {0} epochs.".format(nb_epochs))

        with Timer("Building model"):
            builder = DeepConvNADEBuilder(image_shape=image_shape,
                                          nb_channels=nb_channels,
                                          use_mask_as_input=use_mask_as_input)

            convnet_blueprint = "64@2x2(valid) -> 1@2x2(full)"
            fullnet_blueprint = "5 -> 16"
            print("Convnet:", convnet_blueprint)
            print("Fullnet:", fullnet_blueprint)
            builder.build_convnet_from_blueprint(convnet_blueprint)
            builder.build_fullnet_from_blueprint(fullnet_blueprint)

            model = builder.build()
            model.initialize(initer.UniformInitializer(random_seed=1234))

        with Timer("Building optimizer"):
            loss = BinaryCrossEntropyEstimateWithAutoRegressiveMask(model, trainset)

            optimizer = SGD(loss=loss)
            optimizer.append_direction_modifier(ConstantLearningRate(0.001))

        with Timer("Building trainer"):
            batch_scheduler = MiniBatchSchedulerWithAutoregressiveMask(trainset, batch_size)

            trainer = Trainer(optimizer, batch_scheduler)

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
                                 batch_scheduler=MiniBatchSchedulerWithAutoregressiveMask(validset, batch_size=len(validset),
                                                                                          keep_mask=True))
            trainer.append_task(tasks.Print("Validset - NLL          : {0:.2f} ± {1:.2f}", nll.mean, nll.stderror))

            trainer.append_task(stopping_criteria.MaxEpochStopping(nb_epochs))

            return trainer, nll

    trainer1, nll1 = _build_trainer(nb_epochs=5)
    with Timer("Compiling training graph"):
        trainer1.build_theano_graph()

    with Timer("Compiling training graph"):
        trainer2, nll2 = _build_trainer(nb_epochs=5)

    # Check the two models have been initializedd the same way.
    assert_equal(len(trainer1._optimizer.loss.model.parameters),
                 len(trainer2._optimizer.loss.model.parameters))
    for param1, param2 in zip(trainer1._optimizer.loss.model.parameters,
                              trainer2._optimizer.loss.model.parameters):
        assert_array_equal(param1.get_value(), param2.get_value(), err_msg=param1.name)

    with Timer("Training"):
        trainer1.train()
        trainer2.train()

    # Check the two models are the same after training for 5 epochs.
    assert_equal(len(trainer1._optimizer.loss.model.parameters),
                 len(trainer2._optimizer.loss.model.parameters))
    for param1, param2 in zip(trainer1._optimizer.loss.model.parameters,
                              trainer2._optimizer.loss.model.parameters):

        # I tested it, they are equal when using float64.
        assert_array_almost_equal(param1.get_value(), param2.get_value(), err_msg=param1.name)


def test_save_load_convnade():
    nb_kernels = 8
    kernel_shape = (2, 2)
    hidden_activation = "hinge"
    use_mask_as_input = True
    batch_size = 1024
    ordering_seed = 1234
    max_epoch = 5
    nb_orderings = 1

    with Timer("Loading/processing binarized MNIST"):
        trainset, validset, testset = load_binarized_mnist()

        # Extract the center patch (4x4 pixels) of each image.
        indices_to_keep = [348, 349, 350, 351, 376, 377, 378, 379, 404, 405, 406, 407, 432, 433, 434, 435]

        trainset = Dataset(trainset.inputs.get_value()[:, indices_to_keep], trainset.inputs.get_value()[:, indices_to_keep], name="trainset")
        validset = Dataset(validset.inputs.get_value()[:, indices_to_keep], validset.inputs.get_value()[:, indices_to_keep], name="validset")
        testset = Dataset(testset.inputs.get_value()[:, indices_to_keep], testset.inputs.get_value()[:, indices_to_keep], name="testset")

        image_shape = (4, 4)
        nb_channels = 1

    # Nested function to build a trainer.
    def _build_trainer(nb_epochs):
        print("Will train Convoluational Deep NADE for a total of {0} epochs.".format(nb_epochs))

        with Timer("Building model"):
            builder = DeepConvNADEBuilder(image_shape=image_shape,
                                          nb_channels=nb_channels,
                                          use_mask_as_input=use_mask_as_input)

            convnet_blueprint = "64@2x2(valid) -> 1@2x2(full)"
            fullnet_blueprint = "5 -> 16"
            print("Convnet:", convnet_blueprint)
            print("Fullnet:", fullnet_blueprint)
            builder.build_convnet_from_blueprint(convnet_blueprint)
            builder.build_fullnet_from_blueprint(fullnet_blueprint)

            model = builder.build()
            model.initialize(initer.UniformInitializer(random_seed=1234))

        with Timer("Building optimizer"):
            loss = BinaryCrossEntropyEstimateWithAutoRegressiveMask(model, trainset)

            optimizer = SGD(loss=loss)
            optimizer.append_direction_modifier(ConstantLearningRate(0.001))

        with Timer("Building trainer"):
            batch_scheduler = MiniBatchSchedulerWithAutoregressiveMask(trainset, batch_size)

            trainer = Trainer(optimizer, batch_scheduler)

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
                                 batch_scheduler=MiniBatchSchedulerWithAutoregressiveMask(validset, batch_size=len(validset),
                                                                                          keep_mask=True))
            trainer.append_task(tasks.Print("Validset - NLL          : {0:.2f} ± {1:.2f}", nll.mean, nll.stderror))

            trainer.append_task(stopping_criteria.MaxEpochStopping(nb_epochs))

            return trainer, nll, logger

    trainer1, nll1, logger1 = _build_trainer(nb_epochs=10)
    with Timer("Compiling training graph"):
        trainer1.build_theano_graph()

    with Timer("Training"):
        trainer1.train()

    trainer2a, nll2a, logger2a = _build_trainer(5)
    with Timer("Compiling training graph"):
        trainer2a.build_theano_graph()

    with Timer("Training"):
        trainer2a.train()

    # Save model halfway during training and resume it.
    with tempfile.TemporaryDirectory() as experiment_dir:
        with Timer("Saving"):
            # Save current state of the model (i.e. after 5 epochs).
            trainer2a.save(experiment_dir)

        with Timer("Loading"):
            # Load previous state from which training will resume.
            trainer2b, nll2b, logger2b = _build_trainer(10)
            trainer2b.load(experiment_dir)

            # Check we correctly reloaded the model.
            assert_equal(len(trainer2a._optimizer.loss.model.parameters),
                         len(trainer2b._optimizer.loss.model.parameters))
            for param1, param2 in zip(trainer2a._optimizer.loss.model.parameters,
                                      trainer2b._optimizer.loss.model.parameters):
                assert_array_equal(param1.get_value(), param2.get_value(), err_msg=param1.name)

    with Timer("Compiling training graph"):
        trainer2b.build_theano_graph()

    with Timer("Training"):
        trainer2b.train()

    # Check we correctly resumed training.
    assert_equal(len(trainer1._optimizer.loss.model.parameters),
                 len(trainer2b._optimizer.loss.model.parameters))
    for param1, param2 in zip(trainer1._optimizer.loss.model.parameters,
                              trainer2b._optimizer.loss.model.parameters):

        # I tested it, they are exactly equal when using float64.
        assert_array_almost_equal(param1.get_value(), param2.get_value(), err_msg=param1.name)

    # I tested it, they are exactly equal when using float64.
    assert_array_almost_equal(nll1.mean.view(trainer1.status), nll2b.mean.view(trainer2b.status))
    assert_array_almost_equal(nll1.stderror.view(trainer1.status), nll2b.stderror.view(trainer2b.status))

    # I tested it, they are exactly equal when using float64.
    assert_array_almost_equal(logger1.get_variable_history(0), logger2a.get_variable_history(0)+logger2b.get_variable_history(0))
    assert_array_almost_equal(logger1.get_variable_history(1), logger2a.get_variable_history(1)+logger2b.get_variable_history(1))


def test_new_fprop_matches_old_fprop():
    nb_kernels = 8
    kernel_shape = (2, 2)
    hidden_activation = "sigmoid"
    use_mask_as_input = True
    batch_size = 1024
    ordering_seed = 1234
    max_epoch = 10
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
        nb_channels = 1 + (use_mask_as_input is True)

    with Timer("Building model"):
        builder = DeepConvNADEBuilder(image_shape=image_shape,
                                      nb_channels=nb_channels,
                                      use_mask_as_input=use_mask_as_input)

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
        batch_scheduler = MiniBatchSchedulerWithAutoregressiveMask(trainset, batch_size,
                                                                   use_mask_as_input=use_mask_as_input)

        trainer = Trainer(optimizer, batch_scheduler)

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

        trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))

        trainer.build_theano_graph()

    with Timer("Training"):
        trainer.train()

    mask_o_lt_d = batch_scheduler._shared_batch_mask
    fprop_output, fprop_pre_output = model.fprop(trainset.inputs, mask_o_lt_d, return_output_preactivation=True)
    model_output = model.get_output(T.concatenate([trainset.inputs * mask_o_lt_d, mask_o_lt_d], axis=1))
    assert_array_equal(model_output.eval(), fprop_pre_output.eval())
    print(np.sum(abs(model_output.eval() - fprop_pre_output.eval())))


if __name__ == '__main__':
    # test_simple_convnade()
    # test_convnade_with_mask_as_input_channel()
    # test_convnade_with_max_pooling()
    test_save_load_convnade()
    test_check_init()
    test_new_fprop_matches_old_fprop()
