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


class Flat(ModelBase):

    def __init__(self, config, data_loader, binarizer, evaluator):
        """
        :param config: contains the configuration to initailize a dataset and model.
        """
        super(Flat, self).__init__(config, data_loader, binarizer, evaluator)

    def define_core_architecture(self, doc_representation):
        """
        :return:
        """
        dense1 = tf.keras.layers.Dense(self.dense_layer_size, activation='relu')(doc_representation)
        drop_dense_1 = tf.keras.layers.Dropout(self.dropout_rate)(dense1)
        outputs = tf.keras.layers.Dense(len(self.binarizer.mlb_train.mlb.classes_), activation='sigmoid')(drop_dense_1)
        return outputs
