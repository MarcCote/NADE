from __future__ import division

from collections import OrderedDict

import pickle
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy as np
from os.path import join as pjoin

from smartlearner import Model
import smartlearner.initializers as initer
import smartlearner.utils as smartutils

from abc import ABCMeta, abstractmethod
from types import MethodType
from time import time

from convnade.factories import activation_function_factory


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


class LayerDecorator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def decorate(self, layer):
        raise NotImplementedError("Subclass of 'LayerDecorator' must implement 'decorate(layer)'.")


class MaxPoolDecorator(LayerDecorator):
    def __init__(self, pool_shape, ignore_border=True):
        self.pool_shape = pool_shape
        self.ignore_border = ignore_border

    def decorate(self, layer):
        self._decorate_fprop(layer)
        self._decorate_infer_shape(layer)
        self._decorate_to_text(layer)

    def _decorate_fprop(self, layer):
        layer_fprop = layer.fprop

        def decorated_fprop(instance, input, return_output_preactivation=False):
            if return_output_preactivation:
                output, pre_output = layer_fprop(input, return_output_preactivation)
                pooled_output = downsample.max_pool_2d(output, self.pool_shape, ignore_border=self.ignore_border)
                pooled_pre_output = downsample.max_pool_2d(pre_output, self.pool_shape, ignore_border=self.ignore_border)
                return pooled_output, pooled_pre_output

            output = layer_fprop(input, return_output_preactivation)
            pooled_output = downsample.max_pool_2d(output, self.pool_shape, ignore_border=self.ignore_border)
            return pooled_output

        layer.fprop = MethodType(decorated_fprop, layer)

    def _decorate_infer_shape(self, layer):
        layer_infer_shape = layer.infer_shape

        def decorated_infer_shape(instance, input_shape):
            input_shape = layer_infer_shape(input_shape)
            output_shape = np.array(input_shape[2:]) / np.array(self.pool_shape)
            if self.ignore_border:
                output_shape = np.floor(output_shape)
            else:
                output_shape = np.ceil(output_shape)

            output_shape = input_shape[:2] + tuple(output_shape.astype(int))
            return output_shape

        layer.infer_shape = MethodType(decorated_infer_shape, layer)

    def _decorate_to_text(self, layer):
        layer_to_text = layer.to_text

        def decorated_to_text(instance):
            text = layer_to_text()
            text += " -> max@{0}".format("x".join(map(str, self.pool_shape)))
            return text

        layer.to_text = MethodType(decorated_to_text, layer)


class UpSamplingDecorator(LayerDecorator):
    def __init__(self, up_shape):
        self.up_shape = up_shape

    def decorate(self, layer):
        self._decorate_fprop(layer)
        self._decorate_infer_shape(layer)
        self._decorate_str(layer)

    def _upsample_tensor(self, input):
        shp = input.shape
        upsampled_out = T.zeros((shp[0], shp[1], shp[2]*self.up_shape[0], shp[3]*self.up_shape[1]), dtype=input.dtype)
        upsampled_out = T.set_subtensor(upsampled_out[:, :, ::self.up_shape[0], ::self.up_shape[1]], input)
        return upsampled_out

    def _decorate_fprop(self, layer):
        layer_fprop = layer.fprop

        def decorated_fprop(instance, input, return_output_preactivation=False):
            if return_output_preactivation:
                output, pre_output = layer_fprop(input, return_output_preactivation)
                upsampled_output = self._upsample_tensor(output)
                upsampled_pre_output = self._upsample_tensor(pre_output)
                return upsampled_output, upsampled_pre_output

            output = layer_fprop(input, return_output_preactivation)
            upsampled_output = self._upsample_tensor(output)
            return upsampled_output

        layer.fprop = MethodType(decorated_fprop, layer)

    def _decorate_infer_shape(self, layer):
        layer_infer_shape = layer.infer_shape

        def decorated_infer_shape(instance, input_shape):
            input_shape = layer_infer_shape(input_shape)
            output_shape = np.array(input_shape[2:]) * np.array(self.up_shape)
            output_shape = input_shape[:2] + tuple(output_shape.astype(int))
            return output_shape

        layer.infer_shape = MethodType(decorated_infer_shape, layer)

    def _decorate_str(self, layer):
        layer_to_text = layer.to_text

        def decorated_to_text(instance):
            text = layer_to_text()
            text += " -> up@{0}".format("x".join(map(str, self.up_shape)))
            return text

        layer.to_text = MethodType(decorated_to_text, layer)


class Layer(object):
    def __init__(self, size, name=""):
        self.size = size
        self.name = name
        self.prev_layer = None
        self.next_layer = None

    @property
    def hyperparams(self):
        return {}

    @property
    def parameters(self):
        return []

    def allocate(self):
        pass

    def initialize(self, weight_initializer):
        pass

    def fprop(self, input, return_output_preactivation=False):
        if return_output_preactivation:
            return input, input

        return input

    def to_text(self):
        return self.name

    def __str__(self):
        return self.to_text()

    def infer_shape(self, input_shape):
        return input_shape

    def save(self, savedir):
        hyperparameters = {'version': 1,
                           'size': self.size,
                           'name': self.name}
        smartutils.save_dict_to_json_file(pjoin(savedir, "meta.json"), {"name": self.__class__.__name__})
        smartutils.save_dict_to_json_file(pjoin(savedir, "hyperparams.json"), hyperparameters)

    def load(self, loaddir):
        pass

    @classmethod
    def create(cls, loaddir):
        hyperparameters = smartutils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))
        return cls(size=hyperparameters["size"],
                   name=hyperparameters["name"])


class ConvolutionalLayer(Layer):
    def __init__(self, nb_filters, filter_shape, border_mode, activation="sigmoid", name=""):
        super(ConvolutionalLayer, self).__init__(size=nb_filters, name=name)
        self.nb_filters = nb_filters
        self.filter_shape = tuple(filter_shape)
        self.border_mode = border_mode
        self.activation = activation

        self.activation_fct = activation_function_factory(self.activation)

    def allocate(self):
        # Allocating memory for parameters
        nb_input_feature_maps = self.prev_layer.size
        W_shape = (self.nb_filters, nb_input_feature_maps) + self.filter_shape
        self.W = theano.shared(value=np.zeros(W_shape, dtype=theano.config.floatX), name=self.name+'W', borrow=True)
        self.b = theano.shared(value=np.zeros(self.nb_filters, dtype=theano.config.floatX), name=self.name+'b', borrow=True)
        print(W_shape, self.border_mode)

    def initialize(self, weights_initializer=initer.UniformInitializer(random_seed=1234)):
        weights_initializer(self.W)

    @property
    def hyperparams(self):
        return {'nb_filters': self.nb_filters,
                'filter_shape': self.filter_shape,
                'border_mode': self.border_mode,
                'activation': self.activation}

    @property
    def parameters(self):
        return [self.W, self.b]

    def fprop(self, input, return_output_preactivation=False):
        conv_out = conv.conv2d(input, filters=self.W, border_mode=self.border_mode)
        # TODO: Could be faster if pooling was done here instead
        pre_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        output = self.activation_fct(pre_output)

        if return_output_preactivation:
            return output, pre_output

        return output

    def to_text(self):
        return "{0}@{1}({2})".format(self.nb_filters, "x".join(map(str, self.filter_shape)), self.border_mode)

    def infer_shape(self, input_shape):
        assert len(input_shape) == 4
        if self.border_mode == "valid":
            output_shape = np.array(input_shape[2:]) - np.array(self.filter_shape) + 1
        else:
            output_shape = np.array(input_shape[2:]) + np.array(self.filter_shape) - 1

        output_shape = (input_shape[0], self.nb_filters) + tuple(output_shape.astype(int))
        return output_shape

    def save(self, savedir):
        hyperparameters = {'version': 1,
                           'nb_filters': self.nb_filters,
                           'filter_shape': self.filter_shape,
                           'border_mode': self.border_mode,
                           'activation': self.activation,
                           'name': self.name}
        smartutils.save_dict_to_json_file(pjoin(savedir, "meta.json"), {"name": self.__class__.__name__})
        smartutils.save_dict_to_json_file(pjoin(savedir, "hyperparams.json"), hyperparameters)

        params = {'W': self.W.get_value(),
                  'b': self.b.get_value()}
        np.savez(pjoin(savedir, "params.npz"), **params)

    def load(self, loaddir):
        parameters = np.load(pjoin(loaddir, "params.npz"))
        self.W.set_value(parameters['W'])
        self.b.set_value(parameters['b'])

    @classmethod
    def create(cls, loaddir):
        hyperparameters = smartutils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))
        return cls(nb_filters=hyperparameters["nb_filters"],
                   filter_shape=hyperparameters["filter_shape"],
                   border_mode=hyperparameters["border_mode"],
                   activation=hyperparameters["activation"])


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, hidden_size, activation="sigmoid", name=""):
        super(FullyConnectedLayer, self).__init__(size=hidden_size, name=name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.activation_fct = activation_function_factory(self.activation)

    def allocate(self):
        # Allocating memory for parameters
        W_shape = (self.input_size, self.hidden_size)
        self.W = theano.shared(value=np.zeros(W_shape, dtype=theano.config.floatX), name=self.name+'W', borrow=True)
        self.b = theano.shared(value=np.zeros(self.hidden_size, dtype=theano.config.floatX), name=self.name+'b', borrow=True)
        print(W_shape)

    def initialize(self, weights_initializer=initer.UniformInitializer(random_seed=1234)):
        weights_initializer(self.W)

    @property
    def hyperparams(self):
        return {'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'activation': self.activation}

    @property
    def parameters(self):
        return [self.W, self.b]

    def fprop(self, input, return_output_preactivation=False):
        pre_output = T.dot(input, self.W) + self.b
        output = self.activation_fct(pre_output)

        if return_output_preactivation:
            return output, pre_output

        return output

    def to_text(self):
        return "{0}".format(self.hidden_size)

    def infer_shape(self, input_shape):
        return (input_shape[0], self.hidden_size)

    def save(self, savedir):
        hyperparameters = {'version': 1,
                           'input_size': self.input_size,
                           'hidden_size': self.hidden_size,
                           'activation': self.activation,
                           'name': self.name}
        smartutils.save_dict_to_json_file(pjoin(savedir, "meta.json"), {"name": self.__class__.__name__})
        smartutils.save_dict_to_json_file(pjoin(savedir, "hyperparams.json"), hyperparameters)

        params = {'W': self.W.get_value(),
                  'b': self.b.get_value()}
        np.savez(pjoin(savedir, "params.npz"), **params)

    def load(self, loaddir):
        parameters = np.load(pjoin(loaddir, "params.npz"))
        self.W.set_value(parameters['W'])
        self.b.set_value(parameters['b'])

    @classmethod
    def create(cls, loaddir):
        hyperparameters = smartutils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))
        return cls(input_size=hyperparameters["input_size"],
                   hidden_size=hyperparameters["hidden_size"],
                   activation=hyperparameters["activation"])


class DeepModel(Model):
    def __init__(self, layers, name):
        self.layers = layers
        self.name = name

        # Rename shared variables.
        for i, layer in enumerate(self.layers):
            for param in layer.parameters:
                param.name = self.name + "layer{0}_{1}".format(i, param.name)

    def get_output(self, X):
        output = X
        for layer in self.layers:
            output, pre_activation = layer.fprop(output, return_output_preactivation=True)

        return pre_activation

    def fprop(self, input, return_output_preactivation=False):
        output = input
        for layer in self.layers:
            output, pre_output = layer.fprop(output, return_output_preactivation=True)

        if return_output_preactivation:
            return output, pre_output

        return output

    @property
    def hyperparams(self):
        hyperparams = OrderedDict()
        for i, layer in enumerate(self.layers):
            for k, v in layer.hyperparams.items():
                hyperparams[self.name + "layer{0}_{1}".format(i, k)] = v

        return hyperparams

    @property
    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.parameters

        return parameters

    def initialize(self, weights_initializer=initer.UniformInitializer(random_seed=1234)):
        for layer in self.layers:
            layer.initialize(weights_initializer)

    def __str__(self):
        return " -> ".join(map(str, self.layers))

    def infer_shape(self, input_shape):
        out_shape = input_shape
        for layer in self.layers:
            out_shape = layer.infer_shape(out_shape)

        return out_shape

    def save(self, savedir):
        hyperparameters = {'version': 1,
                           'name': self.name,
                           'nb_layers': len(self.layers)}
        smartutils.save_dict_to_json_file(pjoin(savedir, "meta.json"), {"name": self.__class__.__name__})
        smartutils.save_dict_to_json_file(pjoin(savedir, "hyperparams.json"), hyperparameters)

        for i, layer in enumerate(self.layers):
            savedir_layer = smartutils.create_folder(pjoin(savedir, 'layer_{}'.format(i)))
            layer.save(savedir_layer)

    def load(self, loaddir):
        for i, layer in enumerate(self.layers):
            loaddir_layer = pjoin(loaddir, 'layer_{}'.format(i))
            layer.load(loaddir_layer)

    @classmethod
    def create(cls, loaddir):
        hyperparameters = smartutils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))

        import os
        layers = []
        for i in range(hyperparameters['nb_layers']):
            loaddir_layer = pjoin(loaddir, 'layer_{}'.format(i))

            if not os.path.isfile(pjoin(loaddir_layer, "meta.json")):
                # TODO save `Layer` object too.
                # Assume it's the input layer with 2 channels
                nb_channels = 2
                layer = Layer(size=nb_channels, name=hyperparameters['name'] + "input")
            else:
                meta = smartutils.load_dict_from_json_file(pjoin(loaddir_layer, "meta.json"))
                if meta["name"] == "ConvolutionalLayer":
                    layer = ConvolutionalLayer.create(loaddir_layer)
                elif meta["name"] == "FullyConnectedLayer":
                    layer = FullyConnectedLayer.create(loaddir_layer)
                else:
                    raise NameError("Unknown layer: {}".format(meta["name"]))

            # Connect layers, if needed
            if i > 0:
                layers[-1].next_layer = layer
                layer.prev_layer = layers[-1]

            layers.append(layer)

        for layer in layers:
            layer.allocate()

        return cls(layers=layers, name=hyperparameters['name'])


class DeepConvNADE(Model):
    def __init__(self,
                 image_shape,
                 nb_channels,
                 convnet_layers,
                 fullnet_layers,
                 ordering_seed=1234,
                 use_mask_as_input=False,
                 hidden_activation="sigmoid"):

        self.has_convnet = len(convnet_layers) > 0
        self.has_fullnet = len(fullnet_layers) > 0

        self.convnet = DeepModel(convnet_layers, name="convnet_")
        self.fullnet = DeepModel(fullnet_layers, name="fullnet_")

        self.image_shape = tuple(image_shape)
        self.nb_channels = nb_channels
        self.ordering_seed = ordering_seed
        self.use_mask_as_input = use_mask_as_input
        self.hidden_activation = hidden_activation

        if self.has_convnet:
            # Make sure the convolutional network outputs 'np.prod(self.image_shape)' units.
            input_shape = (1, nb_channels) + self.image_shape
            out_shape = self.convnet.infer_shape(input_shape)
            if out_shape != (1, 1) + self.image_shape:
                raise ValueError("(Convnet) Output shape mismatched: {} != {}".format(out_shape, (1, 1) + self.image_shape))

        if self.fullnet:
            # Make sure the fully connected network outputs 'np.prod(self.image_shape)' units.
            input_shape = (1, int(np.prod(self.image_shape)))
            out_shape = self.fullnet.infer_shape(input_shape)
            if out_shape != (1, int(np.prod(self.image_shape))):
                raise ValueError("(Fullnet) Output shape mismatched: {} != {}".format(out_shape, (1, int(np.prod(self.image_shape)))))

    @property
    def hyperparams(self):
        #hyperparams = super(DeepConvNADE, self).hyperparams
        hyperparams = OrderedDict()
        hyperparams.update(self.convnet.hyperparams)
        hyperparams.update(self.fullnet.hyperparams)
        hyperparams['image_shape'] = self.image_shape
        hyperparams['nb_channels'] = self.nb_channels
        hyperparams['ordering_seed'] = self.ordering_seed
        hyperparams['use_mask_as_input'] = self.use_mask_as_input
        return hyperparams

    @property
    def parameters(self):
        return self.convnet.parameters + self.fullnet.parameters

    @property
    def updates(self):
        return {}  # No updates.

    def initialize(self, weights_initializer=initer.UniformInitializer(random_seed=1234)):
        self.convnet.initialize(weights_initializer)
        self.fullnet.initialize(weights_initializer)

    def get_output(self, X):
        convnet_output = 0
        if self.has_convnet:
            # Hack: input_masked is a 2D matrix instead of a 4D tensor, but we have all the information to fix that.
            input_4D = X.reshape((-1, self.nb_channels) + self.image_shape)
            convnet_output = self.convnet.get_output(input_4D)  # Returns the convnet's output preactivation.

            # This will generate a matrix of shape (batch_size, nb_kernels * kernel_height * kernel_width).
            convnet_output = convnet_output.flatten(2)

        fullnet_output = 0
        if self.has_fullnet:
            fullnet_output = self.fullnet.get_output(X)  # Returns the fullnet's output preactivation.

        output = convnet_output + fullnet_output
        return output

    def save(self, path):
        savedir = smartutils.create_folder(pjoin(path, type(self).__name__))

        hyperparameters = {'version': 2,
                           'image_shape': self.image_shape,
                           'nb_channels': self.nb_channels,
                           'ordering_seed': self.ordering_seed,
                           'use_mask_as_input': self.use_mask_as_input,
                           'hidden_activation': self.hidden_activation,
                           'has_convnet': self.has_convnet,
                           'has_fullnet': self.has_fullnet}
        smartutils.save_dict_to_json_file(pjoin(savedir, "meta.json"), {"name": self.__class__.__name__})
        smartutils.save_dict_to_json_file(pjoin(savedir, "hyperparams.json"), hyperparameters)

        # Save convnet part of the model.
        self.convnet.save(smartutils.create_folder(pjoin(savedir, 'convnet')))
        # self.convnet.save(savedir)

        # Save fullnet part of the model.
        self.fullnet.save(smartutils.create_folder(pjoin(savedir, 'fullnet')))
        # self.fullnet.save(savedir)

    def load(self, path):
        loaddir = pjoin(path, type(self).__name__)
        # self.convnet.load(loaddir)
        # self.fullnet.load(loaddir)
        self.convnet.load(pjoin(loaddir, 'convnet'))
        self.fullnet.load(pjoin(loaddir, 'fullnet'))

    @classmethod
    def create(cls, path):
        loaddir = pjoin(path, cls.__name__)
        hyperparameters = smartutils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))

        # Build convnet
        convnet = DeepModel([], name="convnet_")
        if hyperparameters['has_convnet']:
            convnet = DeepModel.create(pjoin(loaddir, 'convnet'))

        # Build fullnet
        fullnet = DeepModel([], name="fullnet_")
        if hyperparameters['has_fullnet']:
            fullnet = DeepModel.create(pjoin(loaddir, 'fullnet'))

        if hyperparameters["version"] == 1:
            use_mask_as_input = hyperparameters["consider_mask_as_channel"]
        else:
            use_mask_as_input = hyperparameters["use_mask_as_input"]

        return cls(image_shape=hyperparameters["image_shape"],
                   nb_channels=hyperparameters["nb_channels"],
                   convnet_layers=convnet.layers,
                   fullnet_layers=fullnet.layers,
                   ordering_seed=hyperparameters["ordering_seed"],
                   use_mask_as_input=use_mask_as_input)

    def fprop(self, input, mask_o_lt_d, return_output_preactivation=False):
        """ Returns the theano graph that computes the fprop given an `input` and an `ordering`.

        Parameters
        ----------
        input: 2D matrix
            Batch of images. The shape is (batch_size, nb_channels * images_height * images_width).

        mask_o_lt_d: 1D vector or 2D matrix
            Mask allowing only the $d-1$ first dimensions in the ordering i.e. $x_i : i \in o_{<d}$.
            If 1D vector, the same mask is applied to all images in the batch.
            If 2D matrix, each images in the batch will have a different mask meaning that
            mask_o_lt_d.shape[0] == input.shape[0].
        """
        pre_output_convnet = 0
        if self.has_convnet:
            input_masked = input * mask_o_lt_d

            nb_input_feature_maps = self.nb_channels
            if self.use_mask_as_input:
                # nb_input_feature_maps += 1
                nb_input_feature_maps = 2
                if mask_o_lt_d.ndim == 1:
                    # TODO: changed this hack
                    input_masked = T.concatenate([input_masked, T.ones_like(input_masked)*mask_o_lt_d], axis=1)
                else:
                    input_masked = T.concatenate([input_masked, mask_o_lt_d], axis=1)

            # Hack: input_masked is a 2D matrix instead of a 4D tensor, but we have all the information to fix that.
            input_masked = input_masked.reshape((-1, nb_input_feature_maps) + self.image_shape)

            # fprop through all layers
            #_, pre_output = super(DeepConvNADE, self).fprop(input_masked, return_output_preactivation=True)
            _, pre_output = self.convnet.fprop(input_masked, return_output_preactivation=True)

            # This will generate a matrix of shape (batch_size, nb_kernels * kernel_height * kernel_width).
            pre_output_convnet = pre_output.flatten(2)

        pre_output_fully = 0
        if self.has_fullnet:
            input_masked_fully_connected = input * mask_o_lt_d
            if self.use_mask_as_input:
                if mask_o_lt_d.ndim == 1:
                    input_masked_fully_connected = T.concatenate([input_masked_fully_connected, T.ones_like(input_masked_fully_connected)*mask_o_lt_d], axis=1)
                else:
                    input_masked_fully_connected = T.concatenate([input_masked_fully_connected, mask_o_lt_d], axis=1)

            _, pre_output_fully = self.fullnet.fprop(input_masked_fully_connected, return_output_preactivation=True)

            #fully_conn_hidden = T.nnet.sigmoid(T.dot(input_masked_fully_connected, self.W) + self.bhid)
            #pre_output_fully = T.dot(fully_conn_hidden, self.V)

        pre_output = pre_output_convnet + pre_output_fully
        output = T.nnet.sigmoid(pre_output)  # Force use of sigmoid for the output layer

        if return_output_preactivation:
            return output, pre_output

        return output

    # def get_cross_entropies(self, input, mask_o_lt_d):
    #     """ Returns the theano graph that computes the cross entropies for all input dimensions
    #     allowed by the mask `1-mask_o_lt_d` (i.e. the complementary mask).

    #     Parameters
    #     ----------
    #     input: 2D matrix
    #         Batch of images. The shape is (batch_size, nb_channels * images_height * images_width).

    #     mask_o_lt_d: 1D vector or 2D matrix
    #         Mask allowing only the $d-1$ first dimensions in the ordering i.e. $x_i : i \in o_{<d}$.
    #         If 1D vector, the same mask is applied to all images in the batch.
    #         If 2D matrix, each images in the batch will have a different mask meaning that
    #         mask_o_lt_d.shape[0] == input.shape[0].
    #     """
    #     _, pre_output = self.fprop(input, mask_o_lt_d, return_output_preactivation=True)
    #     cross_entropies = T.nnet.softplus(-input * pre_output + (1 - input) * pre_output)
    #     cross_entropies_masked = cross_entropies * (1-mask_o_lt_d)
    #     return cross_entropies_masked

    def get_binary_cross_entropies(self, X, Y):
        """ Returns the theano graph that computes the binary cross-entropies between the inputs $X$
        and their associated target $Y$.

        Notes
        -----
        One needs to sum over axis 1 to obtain the cross-entropy between each $x \in X$ and its
        associated $y \in Y$.
        """
        output = T.nnet.sigmoid(self.get_output(X))  # We assume `self.get_output` returns the preactivation.
        cross_entropies = T.nnet.binary_crossentropy(output, Y)
        return cross_entropies

    def lnp_x_o_d_given_x_o_lt_d(self, input, mask_o_d, mask_o_lt_d):
        """ Returns the theano graph that computes $ln p(x_{o_d}|x_{o_{<d}})$.

        Parameters
        ----------
        input: 2D matrix
            Batch of images. The shape is (batch_size, nb_channels * images_height * images_width).

        mask_o_d: 1D vector or 2D matrix
            Mask allowing only the $d$-th dimension in the ordering i.e. $x_{o_d}$.
            If 1D vector, the same mask is applied to all images in the batch.
            If 2D matrix, each images in the batch will have a different mask meaning that
            mask_o_d.shape[0] == input.shape[0].

        mask_o_lt_d: 1D vector or 2D matrix
            Mask allowing only the $d-1$ first dimensions in the ordering i.e. $x_i : i \in o_{<d}$.
            If 1D vector, the same mask is applied to all images in the batch.
            If 2D matrix, each images in the batch will have a different mask meaning that
            mask_o_lt_d.shape[0] == input.shape[0].
        """
        # Retrieves cross entropies for all possible $p(x_i|x_{o_{<d}})$ where $i \in o_{>=d}$.
        # cross_entropies = self.get_cross_entropies(input, mask_o_lt_d)
        X = input*mask_o_lt_d
        Y = input
        if self.nb_channels == 2:
            X = T.concatenate([X, mask_o_lt_d], axis=1)
        cross_entropies = self.get_binary_cross_entropies(X, Y)

        # We keep only the cross entropy corresponding to $p(x_{o_d}|x_{o_{<d}})$
        cross_entropies_masked = cross_entropies * mask_o_d
        ln_dth_conditional = -T.sum(cross_entropies_masked, axis=1)  # Keep only the d-th conditional
        return ln_dth_conditional

    def nll_of_x_given_o(self, input, ordering):
        """ Returns the theano graph that computes $-ln p(\bx|o)$.

        Parameters
        ----------
        input: 1D vector
            One image with shape (nb_channels * images_height * images_width).

        ordering: 1D vector of int
            List of pixel indices representing the input ordering.
        """

        D = int(np.prod(self.image_shape))
        mask_o_d = T.zeros((D, D), dtype=theano.config.floatX)
        mask_o_d = T.set_subtensor(mask_o_d[T.arange(D), ordering], 1.)

        mask_o_lt_d = T.cumsum(mask_o_d, axis=0)
        mask_o_lt_d = T.set_subtensor(mask_o_lt_d[1:], mask_o_lt_d[:-1])
        mask_o_lt_d = T.set_subtensor(mask_o_lt_d[0, :], 0.)

        input = T.tile(input[None, :], (D, 1))
        nll = -T.sum(self.lnp_x_o_d_given_x_o_lt_d(input, mask_o_d, mask_o_lt_d))
        return nll

    def __str__(self):
        text = ""

        if self.has_convnet:
            text += self.convnet.__str__() + " -> output\n"

        if self.has_fullnet:
            text += self.fullnet.__str__() + " -> output\n"

        return text[:-1]  # Do not return last \n

    def build_sampling_function(self, seed=None):
        from smartpy.misc.utils import Timer
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        rng = np.random.RandomState(seed)
        theano_rng = RandomStreams(rng.randint(2**30))

        # Build theano function
        # $X$: batch of inputs (flatten images)
        input = T.matrix('input')
        # $o_d$: index of d-th dimension in the ordering.
        mask_o_d = T.vector('mask_o_d')
        # $o_{<d}$: indices of the d-1 first dimensions in the ordering.
        mask_o_lt_d = T.vector('mask_o_lt_d')

        output = self.fprop(input, mask_o_lt_d)
        probs = T.sum(output*mask_o_d, axis=1)
        bits = theano_rng.binomial(p=probs, size=probs.shape, n=1, dtype=theano.config.floatX)
        sample_bit_plus = theano.function([input, mask_o_d, mask_o_lt_d], [bits, probs])

        def _sample(nb_samples, ordering_seed=1234):
            rng = np.random.RandomState(ordering_seed)
            D = int(np.prod(self.image_shape))
            ordering = np.arange(D)
            rng.shuffle(ordering)

            with Timer("Generating {} samples from ConvDeepNADE".format(nb_samples)):
                o_d = np.zeros((D, D), dtype=theano.config.floatX)
                o_d[np.arange(D), ordering] = 1

                o_lt_d = np.cumsum(o_d, axis=0)
                o_lt_d[1:] = o_lt_d[:-1]
                o_lt_d[0, :] = 0

                samples = np.zeros((nb_samples, D), dtype="float32")
                samples_probs = np.zeros((nb_samples, D), dtype="float32")
                for d, bit in enumerate(ordering):
                    if d % 100 == 0:
                        print(d)
                    bits, probs = sample_bit_plus(samples, o_d[d], o_lt_d[d])
                    samples[:, bit] = bits
                    samples_probs[:, bit] = probs

                return samples_probs

        return _sample


class DeepConvNADEBuilder(object):
    def __init__(self,
                 image_shape,
                 nb_channels,
                 ordering_seed=1234,
                 use_mask_as_input=False,
                 hidden_activation="sigmoid"):

        self.image_shape = image_shape
        self.nb_channels = nb_channels
        self.ordering_seed = ordering_seed
        self.use_mask_as_input = use_mask_as_input
        self.hidden_activation = hidden_activation

        self.convnet_layers = []
        self.fullnet_layers = []

    def stack(self, layer, layers):
        # Connect layers, if needed
        if len(layers) > 0:
            layers[-1].next_layer = layer
            layer.prev_layer = layers[-1]

        layers.append(layer)

    def build(self):
        for layer in self.convnet_layers:
            layer.allocate()

        for layer in self.fullnet_layers:
            layer.allocate()

        model = DeepConvNADE(image_shape=self.image_shape,
                             nb_channels=self.nb_channels,
                             convnet_layers=self.convnet_layers,
                             fullnet_layers=self.fullnet_layers,
                             ordering_seed=self.ordering_seed,
                             use_mask_as_input=self.use_mask_as_input,
                             hidden_activation=self.hidden_activation)

        return model

    def build_convnet_from_blueprint(self, blueprint):
        """
        Example:
        64@5x5(valid) -> max@2x2 -> 256@2x2(valid) -> 256@2x2(full) -> up@2x2 -> 64@5x5(full)
        """
        input_layer = Layer(size=self.nb_channels, name="convnet_input")
        self.stack(input_layer, self.convnet_layers)

        layers_blueprint = map(str.strip, blueprint.split("->"))

        for layer_blueprint in layers_blueprint:
            infos = layer_blueprint.lower().split("@")
            if infos[0] == "max":
                pool_shape = tuple(map(int, infos[1].split("x")))
                MaxPoolDecorator(pool_shape).decorate(self.convnet_layers[-1])
            elif infos[0] == "up":
                up_shape = tuple(map(int, infos[1].split("x")))
                UpSamplingDecorator(up_shape).decorate(self.convnet_layers[-1])
            else:
                nb_filters = int(infos[0])
                if "valid" in infos[1]:
                    border_mode = "valid"
                elif "full" in infos[1]:
                    border_mode = "full"
                else:
                    raise ValueError("Unknown border mode for '{}'".format(layer_blueprint))

                filter_shape = tuple(map(int, infos[1][:-len("(" + border_mode + ")")].split("x")))
                layer = ConvolutionalLayer(nb_filters=nb_filters,
                                           filter_shape=filter_shape,
                                           border_mode=border_mode,
                                           activation=self.hidden_activation)
                self.stack(layer, self.convnet_layers)

    def build_fullnet_from_blueprint(self, blueprint):
        """
        Example:
        500 -> 256 -> 300 -> 784
        """
        input_layer = Layer(size=int(np.prod(self.image_shape)) * self.nb_channels, name="fullnet_input")
        self.stack(input_layer, self.fullnet_layers)

        layers_blueprint = map(str.strip, blueprint.split("->"))

        for layer_blueprint in layers_blueprint:
            hidden_size = int(layer_blueprint)

            layer = FullyConnectedLayer(input_size=self.fullnet_layers[-1].size,
                                        hidden_size=hidden_size,
                                        activation=self.hidden_activation)
            self.stack(layer, self.fullnet_layers)
