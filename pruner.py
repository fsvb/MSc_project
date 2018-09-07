import tensorflow as tf
import numpy as np
from prunable_model import PrunableModel
from dataset import Dataset


class Pruner:
    def __init__(self, model: PrunableModel, dataset: Dataset, checkpoint_dir="./tmp", early_stopping_patience=10,
                 max_epochs=1000, sess=None):
        self.model = model
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir
        self.pretrain_checkpoint = checkpoint_dir + "/first_stage.ckpt"
        self.patience = early_stopping_patience
        self.max_epochs = max_epochs

        self.saver = tf.train.Saver()
        self.sess = sess if sess is not None else tf.get_default_session()
        if self.sess is None:
            raise ValueError("Make sure you're creating a Pruner within a `with session:` statement scope.")

        self.training_epochs = 0
        self._run_training()
        self.full_loss, self.full_acc = self._run_evaluation_on(self.dataset.test)
        self.saver.save(self.sess, self.pretrain_checkpoint)
        print("Model saved and ready for pruning experiments.")
        self.clean_state = True

    def _reset_state(self):
        if not self.clean_state:
            self.saver.restore(self.sess, self.pretrain_checkpoint)
            self.training_epochs = 0
            self.clean_state = True

    def _run_evaluation_on(self, data, extra_tensors=None):
        if extra_tensors is None:
            extra_tensors = {}

        losses = []
        accuracies = []

        for b in range(len(data)):
            l, a = self.sess.run([self.model.loss, self.model.accuracy],
                                 {self.dataset.handle: data.handle, **extra_tensors})
            losses.append(l)
            accuracies.append(a)

        avg_loss, avg_acc = np.mean(losses), np.mean(accuracies)
        print("Model evaluated on {}: loss = {}, accuracy = {}".format(data.name, avg_loss, avg_acc))
        return avg_loss, avg_acc

    def _run_training(self, extra_tensors=None):
        print("Training the model...")
        if extra_tensors is None:
            extra_tensors = {}

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        mdl = self.model

        epoch_goal = self.training_epochs + self.max_epochs
        while self.training_epochs < epoch_goal:
            losses = []
            lr = mdl.lr_callback(self.training_epochs)
            if mdl.momentum_tensor is not None:
                extra_tensors[mdl.momentum_tensor] = mdl.momentum_callback(self.training_epochs)
            for b in range(len(self.dataset.train)):
                _, l = self.sess.run([mdl.train_step, mdl.loss],
                                     {self.dataset.handle: self.dataset.train.handle,
                                      mdl.in_train_mode: True, mdl.lr_tensor: lr, **extra_tensors})
                losses.append(l)

            self.training_epochs += 1
            print("Epoch #{}: lr = {}, loss = {}".format(self.training_epochs, lr, np.mean(losses)))
            loss, _ = self._run_evaluation_on(self.dataset.val, extra_tensors=extra_tensors)

            if loss < best_val_loss:
                best_val_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if self.patience > 0 and epochs_without_improvement >= self.patience:
                    print("Training complete: early stopping after {} epoch(s).".format(self.training_epochs))
                    return

        print("Training complete.")

    def prune(self, method_name, salience_scorer, percentile):
        print("Running pruning with salience measure '{}', pruning ~{}% of weights."
              .format(method_name, percentile * 100))
        self._reset_state()

        computed_masks = {}
        # Prune every weight matrix in turn, preserving masks for any previously pruned weights
        for i in range(len(self.model.prunable_weights)):
            W = self.model.prunable_weights[i]
            J = self.model.jacobians[i]
            p = self.model.pruning_fracs[i]
            if p is None:
                p = percentile

            print("Pruning weight matrix {} at ~{}%".format(W.name, p * 100))

            # 1. Obtain current values for the weight matrix and approximate second-order derivatives
            W_value = self.sess.run(W)
            dd_value = np.zeros_like(W_value)

            print("Approximating second derivative by computing gradients...")
            train = self.dataset.train
            for b in range(len(train)):
                Js = self.sess.run(J, {self.dataset.handle: self.dataset.train.handle})
                dd_value += np.sum(np.square(Js), axis=0)  # Reduce along batch axis

            # 2. Compute salience measures using provided function
            print("Computing salience measures...")
            salience = salience_scorer(W_value, dd_value)
            k = int(p * salience.size)
            kth_smallest = np.partition(salience.flatten(), k)[k]

            # 3. Record pruned mask
            mask = (salience > kth_smallest).astype(np.float32)
            computed_masks[self.model.masks[i]] = mask

            print("Running pre-retrain evaluation...")
            self._run_evaluation_on(self.dataset.val, extra_tensors=computed_masks)

            # TODO: if pruned weights need to be saved, make sure we multiply them by a mask first.

            # 5. Retrain the model
            # (a) Reset any learned BN means and variances for this and subsequent layers
            reset_ops = [reset_op for layer in self.model.reset_ops[i:] for reset_op in layer]
            if reset_ops:
                self.sess.run(reset_ops)
            # (b) Run retrain.
            self._run_training(computed_masks)

        self.clean_state = False
        return self._run_evaluation_on(self.dataset.test, extra_tensors=computed_masks)
