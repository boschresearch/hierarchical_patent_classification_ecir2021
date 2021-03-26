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


def get_metrics(y_actual, y_pred):
    """
    Create the metrics with precision, recall and f1-score (micro, macro)
    :param y_actual: numpy array
    :param y_pred: numpy array
    :return: dict
    """
    precision_macro, recall_macro, f1_macro = precision_recall_f1(y_actual, y_pred, avg='macro')
    precision_micro, recall_micro, f1_micro = precision_recall_f1(y_actual, y_pred, avg='micro')
    f1_avg = (2 * f1_macro * f1_micro) / (f1_macro + f1_micro)
    return {
        'precision_micro': round(precision_micro, 4),
        'recall_micro': round(recall_micro, 4),
        'f1_micro': round(f1_micro, 4),
        'precision_macro': round(precision_macro, 4),
        'recall_macro': round(recall_macro, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_avg': round(f1_avg, 4)
    }

