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

import json
import os
import pickle

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import TFBertModel

from text_classification.model.callbacks import Callback
from text_classification.utils.utils import get_metrics


class ModelBase:

    def __init__(self, config, data_loader, binarizer):
        """
        Constructor.
        :param config: dict, condiguration of the experiment.
        :param data_loader: DataLoader class object
        :param binarizer: Binarizer class object.
        """
        self.experiment_dir = config["experiment_dir"]
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.embedding_dir = config["embedding_dir"]
        self.data_dir = config["data_dir"]
        self.dense_layer_size = config["dense_layer_size"]
        self.kernels = config["kernels"]
        self.filter_size = config["filter_size"]
        self.epoch = config["epoch"]
        self.batch_size = config["batch_size"]
        self.max_len = config["max_len"]
        self.learning_rate = config["learning_rate"]
        self.bert_trainable = config["bert_trainable"]
        self.dropout_rate = config["dropout_rate"]
        self.input_type = config["input_type"]
        open(os.path.join(self.experiment_dir, 'model_config.json'), 'w').write(json.dumps(config))
        self.binarizer = binarizer
        self.data_loader = data_loader
        self.callback = None

    def init_model(self):
        """
        Check if any past trained model is present in the experiment directory. If yes, load the model weights and other
        parameters.
        :return: self
        """
        if os.path.exists(os.path.join(self.experiment_dir, 'last-model.h5')):
            self.model.load_weights(os.path.join(self.experiment_dir, 'last-model.h5'))
            df_metrics_saved = pd.read_csv(os.path.join(self.experiment_dir, 'metrics.csv'))
            best_f1_macro = df_metrics_saved[df_metrics_saved["split"] == 'dev']["f1_macro"].max()
            metrics = list(df_metrics_saved.T.to_dict().values())
            base_epoch = df_metrics_saved.epoch.max()
        else:
            base_epoch = 0
            best_f1_macro = 0
            metrics = list()

        self.callback = Callback(self.data_loader,
                                 self.binarizer,
                                 self.experiment_dir,
                                 metrics,
                                 patience=5,
                                 base_epoch=base_epoch,
                                 best_f1_macro=best_f1_macro)
        return self

    def generate_document_representation(self):
        """
        Generate document representation. CLS, CNN.
        :return: input representation(CLS or CNN), [token_inputs, mask_inputs, seg_inputs]
        """
        token_inputs = tf.keras.layers.Input(self.max_len, dtype=tf.int32, name='input_word_ids')
        mask_inputs = tf.keras.layers.Input(self.max_len, dtype=tf.int32, name='input_masks')
        seg_inputs = tf.keras.layers.Input(self.max_len, dtype=tf.int32, name='input_segments')

        bert_layer = TFBertModel.from_pretrained(os.path.join(self.embedding_dir),
                                                 from_pt=True,
                                                 output_hidden_states=True,
                                                 name='bert')

        bert_layer.trainable = self.bert_trainable

        last_hidden_state, pooler_output, hidden_states = bert_layer([token_inputs, mask_inputs, seg_inputs])
        if self.input_type == "CNN":
            input_representation = self.__get_CNN_layer(hidden_states)
        elif self.input_type == "CLS":
            input_representation = pooler_output
        return input_representation, [token_inputs, mask_inputs, seg_inputs]

    def __get_CNN_layer(self, hidden_states):
        """
        Get the document reprensentation as CNN layer.
        :param hidden_states:
        :return: CNN layer output.
        """
        seq_output = tf.reduce_sum(tf.stack(hidden_states[-4:]), 0)
        conv_layers = []
        for index, kernel in enumerate(self.kernels):
            conv = tf.keras.layers.Conv1D(filters=self.filter_size,
                                          kernel_size=kernel,
                                          activation='relu')(seq_output)
            drop = tf.keras.layers.Dropout(self.dropout_rate)(conv)
            pool = tf.keras.layers.MaxPooling1D(pool_size=self.max_len - (index + 2) - 1)(drop)
            flat = tf.keras.layers.Flatten()(pool)
            conv_layers.append(flat)
        merged = tf.keras.layers.concatenate(conv_layers)
        return merged

    def define_core_architecture(self):
        """
        Define the core architecture of the model under consideration. flat, TMM,, THMM.
        :return:
        """
        raise NotImplementedError("this method needs to be implemented in child class")

    def compile_model(self):
        """
        Compile the model.
        :return:
        """
        doc_representation, bert_input = self.generate_document_representation()
        outputs = self.define_core_architecture(doc_representation)
        self.model = tf.keras.models.Model(bert_input, outputs)
        adam_optimizer = tfa.optimizers.AdamW(weight_decay=0, learning_rate=self.learning_rate)
        self.model.compile(optimizer=adam_optimizer, loss='binary_crossentropy')
        tf.keras.utils.plot_model(self.model, to_file=os.path.join(self.experiment_dir, "model.png"), show_shapes=True)
        return self

    def train(self):
        """
        Train the model.
        :return: None
        """
        self.model.fit(self.data_loader.X_train_bert_input,
                       self.data_loader.y_train_bin,
                       epochs=self.epoch,
                       batch_size=self.batch_size,
                       shuffle=True,
                       validation_data=(self.data_loader.X_dev_bert_input, self.data_loader.y_dev_bin),
                       callbacks=[self.callback])

    def test(self):
        """
        Load the model weights, predict and evaluate.
        :return: None
        """
        self.model.load_weights(os.path.join(self.experiment_dir, 'best-model.h5'))
        y_pred = self.model.predict(self.data_loader.X_test_bert_input)
        pickle.dump(y_pred, open(os.path.join(self.experiment_dir, "y_pred_test.pkl"), "wb"))
        y_pred = self.binarizer.mlb_train.get_sklearn_mlb_from_pred(y_pred)
        y_test_actual = self.binarizer.mlb_train.get_sklearn_mlb_from_pred(self.data_loader.y_test_bin)

        open(os.path.join(self.experiment_dir, "y_pred_test.txt"), "w").write(
            "\n".join([",".join(list(item[0])) + ":" + ",".join(list(item[1])) for item in zip(
                self.binarizer.mlb_train.mlb.inverse_transform(y_test_actual),
                self.binarizer.mlb_train.mlb.inverse_transform(y_pred))
                       ]))
        print(get_metrics(y_test_actual, y_pred))
