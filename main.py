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

from text_classification.model.THMM import THMM
from text_classification.model.TMM import TMM
from text_classification.model.binarizer import BinarizerDataset, BinarizerTHMM_TMM, BinarizerFlat
from text_classification.model.flat import Flat
from text_classification.utils.common import DataLoader
from text_classification.utils.utils import create_hierarchical_tree

import argparse
import tensorflow as tf
import numpy as np


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Run the classification pipeline.')
    parser.add_argument('--config_file', type=str, help='path to configuration file.')
    parser.add_argument('--op', type=str, help='operation to execute, train|test')

    args = parser.parse_args()
    config = json.loads(open(args.config_file).read())

    os.makedirs(config["experiment_dir"], exist_ok=True)
    data_loader = DataLoader(config["data_dir"], config["experiment_dir"], config["embedding_dir"], config["max_len"])

    if config["model"] == "flat":
        data_loader.load().initialize_bert_input()
        binarizer = BinarizerDataset(data_loader, BinarizerFlat)
        data_loader.binarize(binarizer)
        model = Flat(config, data_loader, binarizer)
    elif config["model"] == "TMM" or config["model"] == "THMM":
        data_loader.load(expand_label=True).initialize_bert_input()
        binarizer = BinarizerDataset(data_loader, BinarizerTHMM_TMM)
        data_loader.binarize(binarizer)
        if config["model"] == "TMM":
            model = TMM(config, data_loader, binarizer)
        elif config["model"] == "THMM":
            child_labels = [[label for label in labels if len(label) == 4] for labels in data_loader.y_train_raw]
            graph = create_hierarchical_tree(child_labels)
            model = THMM(config, data_loader, binarizer, graph=graph)
    else:
        raise ValueError("The model %s not supported." % config["model"])

    binarizer.save(os.path.join(config["experiment_dir"], "binarizer.pkl"))
    model.compile_model().init_model()

    if args.op == "train":
        model.train()
    elif args.op == "test":
        model.test()
    else:
        raise ValueError("The operation type %s is not supported." % args["op"])


if __name__ == '__main__':
    main()
