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

import tensorflow as tf
from text_classification.model.base import ModelBase


class TMM(ModelBase):

    def __init__(self, config, data_loader, binarizer, evaluator):
        """
        Constructor.
        :param config: dict, configuration to load the model.
        :param data_loader: DataLoader object.
        :param binarizer: Binarizer object.
        """
        super(TMM, self).__init__(config, data_loader, binarizer, evaluator)
        self.dense_layer_dict = dict()

    def define_core_architecture(self, doc_representation):
        """
        Define core model architecture with doc representation as input.
        :param doc_representation:
        :return:
        """
        outputs = list()
        for _class in self.binarizer.mlb_train.mlb.classes_:
            hidden_layer_1 = tf.keras.layers.Dense(self.dense_layer_size, activation='relu')(doc_representation)
            drop_hidden_layer_1 = tf.keras.layers.Dropout(self.dropout_rate)(hidden_layer_1)
            hidden_layer_2 = tf.keras.layers.Dense(self.dense_layer_size, activation='relu')(drop_hidden_layer_1)
            drop_hidden_layer_2 = tf.keras.layers.Dropout(self.dropout_rate)(hidden_layer_2)
            outputs.append(tf.keras.layers.Dense(2, activation='softmax')(drop_hidden_layer_2))
        return outputs
