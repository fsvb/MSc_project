import argparse
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from models import MODELS
from dataset import DATASETS, Dataset
from pruner import Pruner


def run(model_name="lenet_300_100", dataset_name="mnist", dropout_prob=0.5, l2_reg=0.0, output_file="output.csv",
        patience=10, epochs=300, pruning_thresholds=None):
    if not pruning_thresholds:
        pruning_thresholds = [0.6, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98]

    print("Configured to run: model = {}, dataset = {}, dropout = {}, l2_reg = {}, output_file = {}, "
          "patience = {}, epochs = {}, thresholds = {}"
          .format(model_name, dataset_name, dropout_prob, l2_reg, output_file, patience, epochs, str(pruning_thresholds)))

    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    tf.reset_default_graph()

    criteria = OrderedDict({
        # "Random":    lambda w, dd: np.random.random(np.shape(w)),
        "Magnitude": lambda w, dd: np.absolute(w),
        "Stat":      lambda w, dd: np.multiply(np.absolute(w), np.sqrt(dd))
    })

    # pruning_thresholds = [0.40, 0.50, 0.65, 0.75, 0.80,
    #                       0.82, 0.84, 0.86, 0.88, 0.90,
    #                       0.91, 0.92, 0.93, 0.94, 0.95,
    #                       0.96, 0.97, 0.98, 0.99]

    with tf.Session() as sess:
        dataset = Dataset(dataset_name, batch_size=128)
        pm = MODELS[model_name](dataset, dropout_prob, l2_reg)
        sess.run(tf.global_variables_initializer())
        pruner = Pruner(pm, dataset, early_stopping_patience=patience, max_epochs=epochs)

        with open(output_file, "w") as o:
            print("Pruned (frac)," + ",".join("{c} (loss), {c} (acc)".format(c=c) for c in criteria.keys()), file=o)
            print("0.0," + ",".join("{},{}".format(pruner.full_loss, pruner.full_acc) for _ in criteria.keys()), file=o)

            for t in pruning_thresholds:
                scores = [pruner.prune(name, scorer, percentile=t) for name, scorer in criteria.items()]
                print(str(t) + "," + ",".join("{},{}".format(l, a) for l, a in scores), file=o)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lenet_300_100", choices=MODELS.keys())
    parser.add_argument("--dataset", default="mnist", choices=DATASETS.keys())
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--thresholds", type=str)
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset
    dropout_prob = args.dropout
    l2_reg = args.l2
    output_file = args.output
    patience = args.patience
    epochs = args.epochs
    thresholds = [float(x) for x in args.thresholds.split(",")] if args.thresholds else None

    run(model, dataset, dropout_prob, l2_reg, output_file, patience, epochs, pruning_thresholds=thresholds)


if __name__ == "__main__":
    main()
