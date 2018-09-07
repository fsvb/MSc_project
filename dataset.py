import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, cifar10


class DataIterator:
    """ Provides iterator-like access to tensors of data, allowing to shuffle entries after every training epoch. """
    def __init__(self, name, batch_size, X, y, sess, augment=None):
        self.name = name

        if augment is None:
            augment = lambda img, lbl: (img, lbl)

        # NB. Seems like below will put all data onto a GPU..?, which is probably fine for small datasets
        # Do check that it doesn't affect serialisation negatively
        with tf.device("/cpu:0"):
            self.dataset = tf.data.Dataset.from_tensor_slices((X, y)) \
                .apply(tf.contrib.data.shuffle_and_repeat(5000, count=None)) \
                .apply(tf.contrib.data.map_and_batch(augment, batch_size,
                                                     num_parallel_calls=multiprocessing.cpu_count()))
        self.dataset = self.dataset.prefetch(1)
        self.num_samples = len(X)
        self.length = (self.num_samples + batch_size - 1) // batch_size  # num_samples / batch_size rounded up
        self.handle = sess.run(self.dataset.make_one_shot_iterator().string_handle())

    def __len__(self):
        return self.length


class Dataset:
    """ Loads and provides access to datasets. """
    def __init__(self, name, batch_size, val_fraction=0.1, sess=None):
        loading_fn, augment_fn = DATASETS[name]
        (X_train, y_train), (X_test, y_test) = loading_fn()

        # Randomly partition train into train & val datasets
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        validation_size = int(len(indices) * val_fraction)
        val_samples = indices[:validation_size]
        train_samples = indices[validation_size:]

        if sess is None:
            sess = tf.get_default_session()

        self.train = DataIterator("train", batch_size, X_train[train_samples], y_train[train_samples], sess,
                                  augment=augment_fn)
        self.val = DataIterator("val", batch_size, X_train[val_samples], y_train[val_samples], sess)
        self.test = DataIterator("test", batch_size, X_test, y_test, sess)

        self.handle = tf.placeholder(tf.string, shape=[], name="dataset_handle")
        self.X, self.y = tf.data.Iterator.from_string_handle(
            self.handle, self.train.dataset.output_types, self.train.dataset.output_shapes).get_next()

        self.img_shape = X_train.shape[1:]
        self.num_classes = y_train.shape[-1]

        print("Created train, val & test sets with {}, {} and {} samples respectively."
              .format(self.train.num_samples, self.val.num_samples, self.test.num_samples))

    @staticmethod
    def _load_mnist():
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        return (np.expand_dims(X_train.astype(np.float32), -1) / 255.0, to_categorical(y_train, num_classes=10)), \
               (np.expand_dims(X_test.astype(np.float32), -1) / 255.0, to_categorical(y_test, num_classes=10))

    @staticmethod
    def _load_cifar10():
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        m, v = np.array([0.4914, 0.48216, 0.44653]), np.array([0.24703, 0.24349, 0.26159])
        process = lambda t: ((t / 255.0 - m) / v).astype(np.float32)
        return (process(X_train), to_categorical(y_train, num_classes=10)), \
               (process(X_test), to_categorical(y_test, num_classes=10))

    @staticmethod
    def _mnist_augment(X, y):
        original_shape = X.get_shape()
        X = tf.pad(X, paddings=[[4, 4], [4, 4], [0, 0]])
        X = tf.random_crop(X, size=original_shape)
        return X, y

    @staticmethod
    def _cifar10_augment(X, y):
        original_shape = X.get_shape()
        X = tf.pad(X, paddings=[[4, 4], [4, 4], [0, 0]])
        X = tf.random_crop(X, size=original_shape)
        X = tf.image.random_flip_left_right(X)
        return X, y

DATASETS = {
    "cifar10": (Dataset._load_cifar10, Dataset._cifar10_augment),
    "mnist": (Dataset._load_mnist, Dataset._mnist_augment),
}
