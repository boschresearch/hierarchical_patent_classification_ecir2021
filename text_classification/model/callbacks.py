# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

from text_classification.utils.utils import precision_recall_f1


class Callback(tf.keras.callbacks.Callback):

    def __init__(self, data_loader, binarizer, experiment_dir, metrics, patience=5, base_epoch=0, best_f1_macro=0):
        self.data_loader = data_loader
        self.binarizer = binarizer
        self.experiment_dir = experiment_dir
        self.metrics = metrics
        self.patience = patience
        self.base_epoch = base_epoch

        self.best_f1_macro = best_f1_macro
        self.best_model = None
        self.best_epoch = None
        self.num_epoch_best_not_encountered = 0
        self.stopped_epoch = 0

    def on_train_end(self, numpy_logs):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

    def get_metric_dict(self, y_actual, y_pred, epoch, logs, split="train"):
        """
        Calculate the micro and macro average metrics.
        :param y_actual:
        :param y_pred:
        :param epoch:
        :param logs:
        :param split:
        :return:
        """

        precision_macro, recall_macro, f1_macro = precision_recall_f1(y_actual, y_pred, avg='macro')
        precision_micro, recall_micro, f1_micro = precision_recall_f1(y_actual, y_pred, avg='micro')
        f1_avg = (2 * f1_macro * f1_micro) / (f1_macro + f1_micro)
        metrics_dict = {
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'f1_avg': f1_avg,
            'split': split,
            'epoch': epoch,
            'loss': logs['loss'],
            'val_loss': logs['val_loss']
        }
        return metrics_dict

    def on_epoch_end(self, epoch, logs=None):

        epoch = self.base_epoch + epoch + 1

        # calculate the performance of the current model on train dataset.
        y_pred_train = self.model.predict(self.data_loader.X_train_bert_input)
        train_metrics_dict = self.get_metric_dict(
            self.binarizer.mlb_train.get_sklearn_mlb_from_pred(self.data_loader.y_train_bin),
            self.binarizer.mlb_train.get_sklearn_mlb_from_pred(y_pred_train),
            epoch,
            logs,
            "train"
        )
        os.makedirs(os.path.join(self.experiment_dir, "train"), exist_ok=True)
        pickle.dump(y_pred_train, open(os.path.join(self.experiment_dir, "train", "%s.pkl" % epoch), "wb"))
        with open(os.path.join(self.experiment_dir, "train", "%s.txt" % epoch), "w") as fileW:
            fileW.write("\n".join([
                ",".join(item[0]) + ":" + ",".join(item[1]) for item in zip(
                    self.binarizer.mlb_train.mlb.inverse_transform(
                        self.binarizer.mlb_train.get_sklearn_mlb_from_pred(self.data_loader.y_train_bin)),
                    self.binarizer.mlb_train.mlb.inverse_transform(
                        self.binarizer.mlb_train.get_sklearn_mlb_from_pred(y_pred_train))
                )
            ]))
        pickle.dump(y_pred_train, open(os.path.join(self.experiment_dir, "train", "%s.pkl" % epoch), "wb"))

        # calculate the performance of the current model on dev dataset.
        y_pred_dev = self.model.predict(self.data_loader.X_dev_bert_input)
        dev_metrics_dict = self.get_metric_dict(
            self.binarizer.mlb_train.get_sklearn_mlb_from_pred(self.data_loader.y_dev_bin),
            self.binarizer.mlb_train.get_sklearn_mlb_from_pred(y_pred_dev),
            epoch,
            logs,
            "dev"
        )
        os.makedirs(os.path.join(self.experiment_dir, "dev"), exist_ok=True)
        with open(os.path.join(self.experiment_dir, "dev", "%s.txt" % epoch), "w") as fileW:
            fileW.write("\n".join([
                ",".join(item[0]) + ":" + ",".join(item[1]) for item in zip(
                    self.binarizer.mlb_train.mlb.inverse_transform(
                        self.binarizer.mlb_train.get_sklearn_mlb_from_pred(self.data_loader.y_dev_bin)),
                    self.binarizer.mlb_train.mlb.inverse_transform(
                        self.binarizer.mlb_train.get_sklearn_mlb_from_pred(y_pred_dev))
                )
            ]))
        pickle.dump(y_pred_dev, open(os.path.join(self.experiment_dir, "dev", "%s.pkl" % epoch), "wb"))

        # check dev performance in current epoch is better than the last best performance.
        if dev_metrics_dict['f1_macro'] > self.best_f1_macro:
            self.best_model = self.model
            self.best_model.save_weights(os.path.join(self.experiment_dir, 'best-model.h5'))
            self.best_f1_macro = dev_metrics_dict['f1_macro']
            self.best_epoch = epoch
            self.num_epoch_best_not_encountered = 0
        else:
            # check if early stop needs to be triggered.
            if self.num_epoch_best_not_encountered == self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            else:
                self.num_epoch_best_not_encountered += 1

        # Update the metrics list with the train and dev metric values.
        self.metrics.append(train_metrics_dict)
        self.metrics.append(dev_metrics_dict)

        # save model as the updated one
        self.model.save_weights(os.path.join(self.experiment_dir, 'last-model.h5'))

        # save metrics
        df_metrics = pd.DataFrame(self.metrics)
        df_metrics.to_csv(os.path.join(self.experiment_dir, 'metrics.csv'), index=False)
        self.plot_metrics(df_metrics)

    def plot_metrics(self, df_metrics):
        """
        plot the dev and train curve for precision recall and f1 (micro/macro).
        :param df_metrics:
        :return:
        """
        f, axes = plt.subplots(1, 6, figsize=(20, 5))
        sns.lineplot(data=df_metrics, x='epoch', y='precision_macro', hue='split', ax=axes[0])
        axes[0].set_ylabel('precision_macro')
        axes[0].set_ylim([0, 1])
        sns.lineplot(data=df_metrics, x='epoch', y='recall_macro', hue='split', ax=axes[1])
        axes[1].set_ylabel('recall_macro')
        axes[1].set_ylim([0, 1])
        sns.lineplot(data=df_metrics, x='epoch', y='f1_macro', hue='split', ax=axes[2])
        axes[2].set_ylabel('f1_macro')
        axes[2].set_ylim([0, 1])
        sns.lineplot(data=df_metrics, x='epoch', y='precision_micro', hue='split', ax=axes[3])
        axes[3].set_ylabel('precision_micro')
        axes[3].set_ylim([0, 1])
        sns.lineplot(data=df_metrics, x='epoch', y='recall_micro', hue='split', ax=axes[4])
        axes[4].set_ylim([0, 1])
        axes[4].set_ylabel('recall_micro')
        sns.lineplot(data=df_metrics, x='epoch', y='f1_micro', hue='split', ax=axes[5])
        axes[5].set_ylim([0, 1])
        axes[5].set_ylabel('f1_micro')
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'performance_over_epochs.png'))
