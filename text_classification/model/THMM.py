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


class THMM(ModelBase):

    def __init__(self, config, data_loader, binarizer, graph):
        super(THMM, self).__init__(config, data_loader, binarizer)
        self.dense_layer_dict = dict()
        self.graph = graph

    def create_dense(self, node, merged, parent_hidden_state):
        successors = [item for item in self.graph.successors(node)]
        if node != 'ROOT':
            if parent_hidden_state != None:
                pre_layer = tf.keras.layers.concatenate([merged, parent_hidden_state])
            else:
                pre_layer = merged
            hidden_layer_1 = tf.keras.layers.Dense(self.dense_layer_size, activation='relu')(pre_layer)
            drop_hidden_layer_1 = tf.keras.layers.Dropout(self.dropout_rate)(hidden_layer_1)
            hidden_layer_2 = tf.keras.layers.Dense(self.dense_layer_size, activation='relu')(drop_hidden_layer_1)
            drop_hidden_layer_2 = tf.keras.layers.Dropout(self.dropout_rate)(hidden_layer_2)
            output = tf.keras.layers.Dense(2, activation='softmax')(drop_hidden_layer_2)
            self.dense_layer_dict[node] = (output, drop_hidden_layer_2)
            if len(successors):
                for successor in successors:
                    self.create_dense(successor, merged, drop_hidden_layer_2)
            else:
                return

    def define_core_architecture(self, doc_representation):
        for node in self.graph.successors('ROOT'):
            self.create_dense(node, doc_representation, None)
        outputs = list()
        for class_ in self.binarizer.mlb_train.mlb.classes_:
            outputs.append(self.dense_layer_dict[class_][0])
        return outputs
