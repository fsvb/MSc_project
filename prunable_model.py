from numbers import Number
from typing import Union, Callable

import tensorflow as tf
import numpy as np


class PrunableModel:
    def __init__(self, l2_reg=0.0):
        """ Creates a model builder that keeps track of weight matrices that can be pruned. All layers are initialized
            using Xavier init and Dropout of specified probability is applied to convolutional and fully-connected
            layers (only during training).
        """
        self.in_train_mode = tf.placeholder_with_default(False, shape=())
        self.l2_reg = l2_reg
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.update_ops = []

        # TODO: parameters below (and 'jacobians' created in 'build...') should be refactored into some WeightInfo class
        self.prunable_weights = []
        self.masks = []
        self.pruning_fracs = []
        self.regularizers = []
        self.reset_ops = []

    def make_placeholder(self, input_shape):
        """ Helper for placeholder tensor creation. """
        return tf.placeholder(tf.float32, input_shape)

    def _effective_retained_fraction(self, mask):
        return tf.reduce_sum(mask) / tf.to_float(tf.size(mask))

    # Dropout and Batch Normalisation are integrated with FC / Conv layers, because their behaviour depends on pruning
    def _dropout(self, input, dropout_prob, mask):
        frac = self._effective_retained_fraction(mask)
        keep_prob = tf.cond(self.in_train_mode,
                            lambda: 1.0 - dropout_prob * tf.sqrt(frac),
                            lambda: 1.0)
        return tf.nn.dropout(input, keep_prob)

    def _batch_norm(self, inputs, momentum=0.99, name=None):
        """
        Batch normalisation. Reduction is performed only for the last dimension, scaling with gamma and beta is not
        supported.
        """
        # TODO: reintroduce gamma and beta? (non-weight-decayable?)
        # Heavily inspired by Keras
        import tensorflow.keras.backend as K
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(len(input_shape)))[:-1]

        with tf.variable_scope("bn" if name is None else name + "/bn"):
            track_mean = tf.Variable(tf.zeros(input_shape[-1:]), name='track_mean', trainable=False)
            track_var = tf.Variable(tf.ones(input_shape[-1:]), name='track_var', trainable=False)

            # In inference mode BN just uses tracked mean and variance; in train, it uses batch statistics.
            # We manually track mean and variance using EWMA.

            bn_training, mean, variance = K.normalize_batch_in_training(inputs, None, None, reduction_axes)

            sample_size = tf.reduce_prod([tf.shape(inputs)[axis] for axis in reduction_axes])
            sample_size = tf.cast(sample_size, dtype=K.dtype(inputs))
            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - 1.001)

            self.update_ops.extend([K.moving_average_update(track_mean, mean, momentum),
                                    K.moving_average_update(track_var, variance, momentum)])
            self.reset_ops.append([tf.assign(track_mean, tf.zeros_like(track_mean)),
                                   tf.assign(track_var, tf.ones_like(track_var))])

            return tf.cond(self.in_train_mode,
                           lambda: bn_training,
                           lambda: K.batch_normalization(inputs, track_mean, track_var, None, None))

    def fully_connected(self, x, output_dim, name, bias=True,
                        activation=tf.nn.relu, dropout=0.5, batch_norm=False, pruning_frac=None):
        """ Creates a fully-connected layer, registering it's weight matrix with the
            pruning system and allowing for masking out certain elements of it.
            :param pruning_frac Hints to the pruner to use specified pruning fraction instead of global one.
        """
        input_dim = x.get_shape()[-1].value
        W_shape = (input_dim, output_dim)

        with tf.variable_scope(name):
            W = tf.Variable(self.initializer(W_shape), name="weight")
            mask = tf.placeholder_with_default(np.ones(W_shape, dtype=np.float32), shape=W_shape, name="mask")
            if self.l2_reg > 0.0:
                self.regularizers.append(tf.nn.l2_loss(W, name="weight_l2"))

            self.prunable_weights.append(W)
            self.masks.append(mask)
            self.pruning_fracs.append(pruning_frac)

            o = tf.matmul(x, tf.multiply(W, mask))
            if bias:
                b = tf.Variable(self.initializer([1, output_dim]), name="bias")
                o += b
            if batch_norm:
                o = self._batch_norm(o, name=name)
            if activation:
                o = activation(o)
            if dropout:
                o = self._dropout(o, dropout, mask)
            return o

    def conv2d(self, x, k_size, out_channels, name, padding="VALID", bias=True, strides=None,
               activation=tf.nn.relu, dropout=0.5, batch_norm=False, pruning_frac=None):
        """ Creates a convolutional layer, registering it's weight matrix with the
            pruning system and allowing for masking out certain elements of it.
            **Assumes that the input matrix has NHWC ordering.**
            :param pruning_frac Hints to the pruner to use specified pruning fraction instead of global one.
        """
        in_channels = x.get_shape()[-1].value
        k_h, k_w = k_size
        k_shape = (k_h, k_w, in_channels, out_channels)
        if strides is None:
            strides = [1, 1, 1, 1]

        with tf.variable_scope(name):
            W = tf.Variable(self.initializer(k_shape), name="weight")
            mask = tf.placeholder_with_default(np.ones(k_shape, dtype=np.float32), shape=k_shape, name="mask")
            if self.l2_reg > 0.0:
                self.regularizers.append(tf.nn.l2_loss(W, name="weight_l2"))

            self.prunable_weights.append(W)
            self.masks.append(mask)
            self.pruning_fracs.append(pruning_frac)

            o = tf.nn.conv2d(x, tf.multiply(W, mask), strides=strides, padding=padding)
            if bias:
                b = tf.Variable(self.initializer((out_channels,)), name="bias")
                o = tf.nn.bias_add(o, b)
            if batch_norm:
                o = self._batch_norm(o, name=name)
            if activation:
                o = activation(o)
            if dropout:
                o = self._dropout(o, dropout, mask)
            return o

    def max_pool(self, x, pool_size):
        """ Helper for max-pooling operation. """
        w_h, w_w = pool_size
        return tf.nn.max_pool(x, ksize=[1, w_h, w_w, 1], strides=[1, w_h, w_w, 1], padding="VALID")

    def avg_pool(self, x, pool_size):
        """ Helper for max-pooling operation. """
        w_h, w_w = pool_size
        return tf.nn.avg_pool(x, ksize=[1, w_h, w_w, 1], strides=[1, w_h, w_w, 1], padding="VALID")

    def flatten(self, x):
        """ Helper for flatting a tensor while preserving batch dimension. """
        dim = np.prod(x.get_shape().as_list()[1:])
        return tf.reshape(x, [-1, dim])

    def xentropy_loss(self, input, logits, labels):
        """ Finalises the model with cross-entropy loss. Returns tensor of predictions. """
        if hasattr(self, "loss"):
            raise ValueError("Only one loss at a time is supported.")
        self.input = input
        self.loss = \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        if self.regularizers:
            self.loss += self.l2_reg * tf.add_n(self.regularizers)
        self.predictions = tf.sigmoid(logits)
        self.labels = labels
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return self.predictions

    def build(self, optimizer="adam", learning_rate: Union[Number, Callable[[Number], Number]]=0.001,
              momentum: Union[Number, Callable[[Number], Number]]=None, **extra_optimizer_params):
        """
        Finalises the model for use in pruning.
        :param optimizer: optimizer name, either "adam" or "sgd"
        :param learning_rate: learning rate, either a number or a callback that returns LR given an epoch number
        :param momentum: mometum, either a number of a callback that returns momentum given an epoch number
        :param extra_optimizer_params: extra key-value pairs to be passed in to optimizer's constructor
        """
        if not hasattr(self, "loss"):
            raise ValueError("Model must have a loss set before being built.")

        self.jacobians = [tf.gradients(self.loss, w) for w in self.prunable_weights]

        def create_sgd(learning_rate, momentum, **extra_params):
            if momentum is not None:
                return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, **extra_params)
            else:
                return tf.train.GradientDescentOptimizer(learning_rate=learning_rate, **extra_params)

        def create_adam(learning_rate, momentum, **extra_params):
            if momentum is not None:
                return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=momentum, **extra_params)
            else:
                return tf.train.AdamOptimizer(learning_rate=learning_rate, **extra_params)

        def create_adamw(learning_rate, momentum, **extra_params):
            if "weight_decay" not in extra_params:
                raise ValueError("You must specify 'weight_decay' parameter if you use AdamW.")
            if momentum is not None:
                return tf.contrib.opt.AdamWOptimizer(learning_rate=learning_rate, beta1=momentum, **extra_params)
            else:
                return tf.contrib.opt.AdamWOptimizer(learning_rate=learning_rate, **extra_params)

        optimizers = {
            "sgd": create_sgd,
            "adam": create_adam,
            "adamw": create_adamw,
        }

        if optimizer not in optimizers.keys():
            raise ValueError("Unsupported optimizer {}, must be in {}".format(optimizer, optimizers.keys()))

        if not callable(learning_rate):
            self.lr_callback = lambda epoch: learning_rate
        else:
            self.lr_callback = learning_rate

        if not callable(momentum):
            self.momentum_callback = lambda epoch: momentum
        else:
            self.momentum_callback = momentum

        self.lr_tensor = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        # In Adam, beta variables have to be initialised in tf.global_variables_initializer,
        # so we provide a default value here.
        self.momentum_tensor = tf.placeholder_with_default(tf.constant(0.99, dtype=tf.float32),
                                                           shape=(), name="momentum") if momentum is not None else None

        opt = optimizers[optimizer](learning_rate=self.lr_tensor, momentum=self.momentum_tensor, **extra_optimizer_params)
        with tf.control_dependencies(self.update_ops):
            self.train_step = opt.minimize(self.loss)
