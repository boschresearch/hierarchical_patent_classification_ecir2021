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

import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = 'ROOT'


class BinarizerBase:
    """
    Base class for binarizer.
    """

    def __init__(self):
        self.mlb = None

    def fit(self, labels):
        """
        fit the binarizer, to be implemented in the child.
        :param labels:
        :return:
        """
        raise NotImplementedError("fit method needs to be implemented in child class.")

    def transform(self, y):
        """
        transform the list of labels into binary values, to be implemented in the child.
        :param y: list, String
        :return: numpy array.
        """
        raise NotImplementedError("transform method needs to be implemented in child class.")

    def fit_transform(self, y):
        """
        fit the binarizer and transform the list of labels into binary values, to be implemented in the child..
        :param y: list, String
        :return: numpy array.
        """
        self.fit(y)
        return self.transform(y)

    def get_sklearn_mlb_from_pred(self, y, pred_prob=0.5, prob_value=False):
        """
        Get the binary values for prediction probability from the binarizer.
        :param y: list, String
        :return: numpy array.
        """
        raise NotImplementedError("this method needs to be implemented in child class.")


class BinarizerFlat(BinarizerBase):
    """
    Binarizer for flat classifier. It wraps the sklearn binarizer.
    """

    def __init__(self):
        super(BinarizerFlat, self).__init__()

    def fit(self, y):
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(y)

    def transform(self, y):
        return self.mlb.transform(y)

    def inverse_transform(self, binary, pred_prob=0.5):
        binary = binary > pred_prob
        return self.mlb.inverse_transform(binary)

    def get_sklearn_mlb_from_pred(self, y, pred_prob=0.5, prob_value=False):
        if prob_value:
            return y
        else:
            return y > pred_prob


class BinarizerTHMM_TMM(BinarizerBase):
    """
    Binarizer for multi-task hierarchical classifier. It wraps the sklearn binarizer.
    """

    def __init__(self):
        super(BinarizerTHMM_TMM, self).__init__()

    def fit(self, y):
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(y)

    def transform(self, y):
        if not self.mlb:
            raise Exception("binarizer not trained")
        y = self.mlb.transform(y)
        y_binary_list = [(self.mlb.classes_[index], list()) for index in range(y.shape[1])]
        for item in list(y):
            for index, label_bin in enumerate(list(item)):
                if label_bin == 1:
                    y_binary_list[index][1].append([1, 0])
                else:
                    y_binary_list[index][1].append([0, 1])
        y_binary_list = [np.asarray(item[1]) for item in y_binary_list]
        return y_binary_list

    def get_sklearn_mlb_from_pred(self, y, pred_prob=0.5, prob_value=False):
        predictions = list()
        y = [list(item) for item in y]
        for index in range(len(y[0])):
            instance_list = list()
            for class_index in range(len(self.mlb.classes_)):
                if prob_value:
                    instance_list.append(y[class_index][index][0])
                else:
                    if y[class_index][index][0] > pred_prob:
                        instance_list.append(self.mlb.classes_[class_index])
            predictions.append(instance_list)
        if prob_value:
            return np.asarray([np.asarray(row) for row in predictions])
        else:
            return self.mlb.transform(predictions)

    def inverse_transform(self, y, pred_prob=0.5):
        predictions = list()
        y = [list(item) for item in y]
        for index in range(len(y[0])):
            instance_list = list()
            for class_index in range(len(self.mlb.classes_)):
                if y[class_index][index][0] > pred_prob:
                    instance_list.append(self.mlb.classes_[class_index])
            predictions.append(instance_list)
        return predictions


class BinarizerDataset:
    """
    A binarizer for each train, dev and test.
    """

    def __init__(self, data_loader, class_object):
        self.mlb_train = class_object()
        self.mlb_train.fit(data_loader.y_train_raw)

        self.mlb_dev = class_object()
        self.mlb_dev.fit(data_loader.y_dev_raw)

        self.mlb_test = class_object()
        self.mlb_test.fit(data_loader.y_test_raw)

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))
