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

import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer


class BERTHandler:

    def __init__(self, embedding_dir):
        """
        Constructor.
        :param embedding_dir: Str, path to the embedding directory.
        """
        self.bert_tokenizer_transformer = BertTokenizer.from_pretrained(embedding_dir, do_lower_case=True)

    def create_bert_input(self, X, max_len):
        """
        Create input for BERT with provided text value in X.
        :param X: Different values of text.
        :param max_len: maimum length of text sequence.
        :return:
        """
        input_ids_all = []
        input_mask_all = []
        segment_ids_all = []

        for text in X:
            tokens = self.bert_tokenizer_transformer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

            segment_ids = [0] * len(tokens)
            input_ids = self.bert_tokenizer_transformer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            if len(input_ids) < max_len:
                # Zero-pad up to the sequence length.
                padding = [0] * (max_len - len(input_ids))
                padding_seg = [1] * (max_len - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding_seg
            else:
                input_ids = [input_ids[0]] + input_ids[1:max_len-1] + [input_ids[len(input_ids) - 1]]
                input_mask = input_mask[:max_len]
                segment_ids = segment_ids[:max_len]

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            segment_ids_all.append(segment_ids)

        return [tf.cast(input_ids_all, tf.int32), tf.cast(input_mask_all, tf.int32), tf.cast(segment_ids_all, tf.int32)]


class DataLoader:

    def __init__(self,
                 data_dir,
                 experiment_dir=None,
                 embedding_dir=None,
                 max_len=2,
                 ):
        """
        :param experiment_dir: str, directory to store information on current run.
        :param data_dir: str, where to load data from.
        :param embedding_dir: str, directory to look for embeddings.
        :param max_len: maximum length of input sequence.
        """

        self.mlb_train = None
        self.mlb_test = None
        self.mlb_dev = None

        self.experiment_dir = experiment_dir
        self.data_dir = data_dir
        self.max_len = max_len

        # raw text
        self.X_train = None
        self.X_test = None
        self.X_dev = None

        # bert inputs
        self.X_train_bert_input = None
        self.X_test_bert_input = None
        self.X_dev_bert_input = None

        # raw labels
        self.y_train_raw = None
        self.y_test_raw = None
        self.y_dev_raw = None

        # binarized labels
        self.y_train_bin = None
        self.y_test_bin = None
        self.y_dev_bin = None

        self.bert_handler = BERTHandler(embedding_dir)

    def load(self, expand_label=False):
        """
        load the data labels, check if label expansion required or not, flat vs global(hier).
        :param expand_label: boolean, flag for expansion of labels.
        :return: self
        """
        train = pd.read_csv(os.path.join(self.data_dir, "train.tsv"), sep="\t")
        test = pd.read_csv(os.path.join(self.data_dir, "test.tsv"), sep="\t")
        dev = pd.read_csv(os.path.join(self.data_dir, "dev.tsv"), sep="\t")

        self.X_train = [row["title"].lower() + " " + row["abstract"].lower() for _, row in train.iterrows()]
        self.X_dev = [row["title"].lower() + " " + row["abstract"].lower() for _, row in dev.iterrows()]
        self.X_test = [row["title"].lower() + " " + row["abstract"].lower() for _, row in test.iterrows()]
        self.y_train_raw = [eval(item) for item in train.labels.tolist()]
        self.y_test_raw = [eval(item) for item in test.labels.tolist()]
        self.y_dev_raw = [eval(item) for item in dev.labels.tolist()]

        if expand_label:
            self.y_train_raw = [list(set(sum([[j[0], j[:3], j] for j in tk], []))) for tk in self.y_train_raw]
            self.y_dev_raw = [list(set(sum([[j[0], j[:3], j] for j in tk], []))) for tk in self.y_dev_raw]
            self.y_test_raw = [list(set(sum([[j[0], j[:3], j] for j in tk], []))) for tk in self.y_test_raw]

        return self

    def initialize_bert_input(self):
        """
        Initialize BERT input.
        :return: self
        """
        if self.X_train:
            self.X_train_bert_input = self.bert_handler.create_bert_input(self.X_train, max_len=self.max_len)
        else:
            raise ValueError("X_train values not initialized.")

        if self.X_test:
            self.X_test_bert_input = self.bert_handler.create_bert_input(self.X_test, max_len=self.max_len)
        else:
            raise ValueError("X_test values not initialized.")

        if self.X_dev:
            self.X_dev_bert_input = self.bert_handler.create_bert_input(self.X_dev, max_len=self.max_len)
        else:
            raise ValueError("X_dev values not initialized.")

        return self

    def binarize(self, binarizer):
        """
        Convert the initialized labels into binary values.
        :param binarizer:
        :return: self
        """
        if self.y_train_raw:
            self.y_train_bin = binarizer.mlb_train.transform(self.y_train_raw)
        else:
            raise ValueError("y_train labels not intailized")

        if self.y_dev_raw:
            self.y_dev_bin = binarizer.mlb_train.transform(self.y_dev_raw)
        else:
            raise ValueError("y_dev labels not intailized")

        if self.y_test_raw:
            self.y_test_bin = binarizer.mlb_train.transform(self.y_test_raw)
        else:
            raise ValueError("y_test labels not intailized")

        return self
