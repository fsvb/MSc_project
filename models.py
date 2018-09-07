from dataset import Dataset
from prunable_model import PrunableModel
from misc import super_convergence
import tensorflow as tf


def lenet5_model(dataset: Dataset, dropout, l2_reg):
    # Works best on 28x28 MNIST, but could work on CIFAR-10 too
    # Taken from https://www.researchgate.net/figure/The-LeNet-5-Architecture-a-convolutional-neural-network_fig4_321586653
    pm = PrunableModel(l2_reg)
    inputs = dataset.X
    labels = dataset.y

    x = pm.conv2d(inputs, k_size=(5, 5), out_channels=6, dropout=None, name="conv_1", pruning_frac=0.4)
    x = pm.max_pool(x, pool_size=(2, 2))
    x = pm.conv2d(x, k_size=(5, 5), out_channels=16, dropout=None, name="conv_2", pruning_frac=0.6)
    x = pm.max_pool(x, pool_size=(2, 2))
    x = pm.flatten(x)
    x = pm.fully_connected(x, 120, dropout=dropout, name="fc1")
    x = pm.fully_connected(x, 84, dropout=dropout, name="fc2")
    x = pm.fully_connected(x, dataset.num_classes, dropout=dropout, activation=None, name="fc3")
    predictions = pm.xentropy_loss(input=inputs, logits=x, labels=labels)
    pm.build(optimizer="adam")
    return pm


def lenet_300_100_model(dataset: Dataset, dropout, l2_reg):
    pm = PrunableModel(l2_reg)
    inputs = dataset.X
    labels = dataset.y

    flat_img = pm.flatten(inputs)
    x = pm.fully_connected(flat_img, 300, dropout=dropout, name="fc1")
    x = pm.fully_connected(x, 100, dropout=dropout, name="fc2")
    x = pm.fully_connected(x, dataset.num_classes, activation=None, dropout=dropout, name="out")
    predictions = pm.xentropy_loss(input=inputs, logits=x, labels=labels)
    pm.build(optimizer="adam")
    return pm


def cifar10_convnet_model(dataset: Dataset, dropout, l2_reg):
    pm = PrunableModel(l2_reg)
    i = dataset.X
    l = dataset.y

    x = pm.conv2d(i, k_size=(3, 3), out_channels=32, padding="SAME",
                  dropout=dropout, name="conv1", pruning_frac=0.25)
    x = pm.conv2d(x, k_size=(3, 3), out_channels=32,
                  dropout=dropout, name="conv2", pruning_frac=0.30)
    x = pm.max_pool(x, pool_size=(2, 2))
    x = pm.conv2d(x, k_size=(3, 3), out_channels=64, padding="SAME",
                  dropout=dropout, name="conv3", pruning_frac=0.30)
    x = pm.conv2d(x, k_size=(3, 3), out_channels=64,
                  dropout=dropout, name="conv4", pruning_frac=0.30)
    x = pm.conv2d(x, k_size=(3, 3), out_channels=64, padding="SAME",
                  dropout=dropout, name="conv5", pruning_frac=0.50)
    x = pm.conv2d(x, k_size=(3, 3), out_channels=64,
                  dropout=dropout, name="conv6", pruning_frac=0.50)
    x = pm.max_pool(x, pool_size=(2, 2))
    x = pm.flatten(x)
    x = pm.fully_connected(x, 512, dropout=dropout, name="fc1")
    x = pm.fully_connected(x, dataset.num_classes, dropout=dropout, name="fc2", activation=None)
    pm.xentropy_loss(input=i, logits=x, labels=l)
    pm.build(optimizer="adam")

    return pm


def resnet_18(dataset: Dataset, dropout, l2_reg):
    pm = PrunableModel(l2_reg)

    def basic_block(i, in_channels, out_channels, name, stride):
        x = pm.conv2d(i, k_size=(3, 3), out_channels=out_channels, padding="SAME", strides=[1, stride, stride, 1],
                      dropout=None, bias=False, batch_norm=True, name=name + "_conv1", pruning_frac=0.5)
        x = pm.conv2d(x, k_size=(3, 3), out_channels=out_channels, padding="SAME", activation=None,
                      dropout=None, bias=False, batch_norm=True, name=name + "_conv2", pruning_frac=0.5)
        if in_channels != out_channels or stride != 1:
            i = pm.conv2d(i, k_size=(1, 1), out_channels=out_channels, padding="SAME", activation=None,
                          strides=[1, stride, stride, 1], dropout=None, bias=False, batch_norm=True, name=name + "_shortcut", pruning_frac=0.5)
        x = tf.nn.relu(x + i)
        return x

    i = dataset.X
    l = dataset.y

    x = pm.conv2d(i, k_size=(3, 3), out_channels=64, padding="VALID", pruning_frac=0.4,
                  dropout=None, bias=False, batch_norm=True, name="conv1")
    x = basic_block(x, in_channels=64, out_channels=64, name="bb11", stride=1)
    x = basic_block(x, in_channels=64, out_channels=64, name="bb12", stride=1)
    x = basic_block(x, in_channels=64, out_channels=128, name="bb21", stride=2)
    x = basic_block(x, in_channels=128, out_channels=128, name="bb22", stride=1)
    x = basic_block(x, in_channels=128, out_channels=256, name="bb31", stride=2)
    x = basic_block(x, in_channels=256, out_channels=256, name="bb32", stride=1)
    x = basic_block(x, in_channels=256, out_channels=512, name="bb41", stride=2)
    x = basic_block(x, in_channels=512, out_channels=512, name="bb42", stride=1)
    x = pm.avg_pool(x, pool_size=(4, 4))
    x = pm.flatten(x)
    x = pm.fully_connected(x, output_dim=dataset.num_classes, name="fc", activation=None, dropout=None)

    preds = pm.xentropy_loss(input=i, logits=x, labels=l)

    lr, mom = super_convergence(epochs_for_cycle=30)
    pm.build(optimizer="adamw", learning_rate=lr, momentum=mom, weight_decay=0.0001)
    return pm


MODELS = {
    "lenet5": lenet5_model,
    "lenet_300_100": lenet_300_100_model,
    "cifar10_convnet_model": cifar10_convnet_model,
    "resnet_18": resnet_18
}
