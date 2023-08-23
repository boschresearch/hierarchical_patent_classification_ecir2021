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

from networkx import DiGraph
from sklearn.metrics import precision_score, recall_score, f1_score


def precision_recall_f1(y_actual, y_pred, avg):
    return precision_score(y_actual, y_pred, average=avg), \
           recall_score(y_actual, y_pred, average=avg), \
           f1_score(y_actual, y_pred, average=avg)


ROOT = 'ROOT'


def extend_hierarchy(hierarchy, y_labs):
    for samples_t in y_labs:
        if not isinstance(samples_t, list):
            samples = [samples_t]
        else:
            samples = samples_t
        for lab in samples:
            par_1 = lab[0]
            par_2 = lab[:3]
            child = lab[:]

            if par_1 not in hierarchy[ROOT]:
                hierarchy[ROOT].append(par_1)
            if par_1 not in hierarchy:
                hierarchy[par_1] = [par_2]
            else:
                if par_2 not in hierarchy[par_1]:
                    hierarchy[par_1].append(par_2)
            if par_2 not in hierarchy:
                hierarchy[par_2] = [child]
            else:
                if child not in hierarchy[par_2]:
                    hierarchy[par_2].append(child)
    return hierarchy


def build_hierarchy(issues):
    hierarchy = {ROOT: []}
    for i in issues:
        par_1 = i[0]
        par_2 = i[:3]
        child = i[:]

        if par_1 not in hierarchy[ROOT]:
            hierarchy[ROOT].append(par_1)
        if par_1 not in hierarchy:
            hierarchy[par_1] = [par_2]
        else:
            if par_2 not in hierarchy[par_1]:
                hierarchy[par_1].append(par_2)
        if par_2 not in hierarchy:
            hierarchy[par_2] = [child]
        else:
            hierarchy[par_2].append(child)
    return hierarchy


def create_hierarchical_tree(labels):
    """
    create the graph g, for the labels.
    :param labels:
    :return:
    """
    hierarchy_f = build_hierarchy([tj for tk in labels for tj in tk])
    class_hierarchy = extend_hierarchy(hierarchy_f, labels)
    g = DiGraph(class_hierarchy)
    return g


import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score


class Evaluator:

    def __init__(self, mlb):
        self.mlb = mlb

    def calc_metrics(self, true, pred) -> dict:
        return {
            "precision_macro": round(precision_score(true, pred, average="macro"), 3),
            "recall_macro": round(recall_score(true, pred, average="macro"), 3),
            "f1_macro": round(f1_score(true, pred, average="macro"), 3),
            "precision_micro": round(precision_score(true, pred, average="micro"), 3),
            "recall_micro": round(recall_score(true, pred, average="micro"), 3),
            "f1_micro": round(f1_score(true, pred, average="micro"), 3),
        }


    def expand_labels(self, 
                    predicted_labels_list) -> list:
        
        pred_labels_list_exp = list()
        for y in predicted_labels_list:
            exp_labels = set()

            for label in y:
                exp_labels.add(label)
                if len(label) == 3:
                    exp_labels.add(label[:1])
                elif len(label) == 4:
                    exp_labels.add(label[:1])
                    exp_labels.add(label[:3])

            pred_labels_list_exp.append(list(exp_labels))
        return pred_labels_list_exp


    def get_metrics(self,
                actual : np.ndarray, 
                pred: np.ndarray):

        pred_labels = self.mlb.inverse_transform(pred)
        
        pred_labels = self.expand_labels(pred_labels)
        
        pred_labels = self.mlb.transform(pred_labels)

        metrics = self.calc_metrics(actual, pred_labels)

        return metrics, pred

