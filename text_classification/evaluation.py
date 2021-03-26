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
from sklearn.metrics import average_precision_score

from text_classification.utils.common import DataLoader
from text_classification.utils.utils import precision_recall_f1

exps = json.load(open("/home/ujp5kor/bosch/code/ecir-2021-code/config/evaluation_config.json"))

metrics = list()

for exp in exps:
    config = json.loads(open(os.path.join(exp["dir"], "model_config.json")).read())

    data_loader = DataLoader(config["data_dir"], config["experiment_dir"], config["embedding_dir"], config["max_len"])
    data_loader.load(expand_label=True)

    binarizer = pickle.load(open(os.path.join(config["experiment_dir"], "binarizer.pkl"), "rb"))
    y_test_actual = data_loader.y_test_raw
    y_pred = pickle.load(open(os.path.join(config["experiment_dir"], "y_pred_test.pkl"), "rb"))
    y_pred_values = binarizer.mlb_train.inverse_transform(y_pred)
    hierarchical_binarizer = pickle.load(open(exp["binarizer_hierarchical"], "rb"))

    if exp["model_type"] == "flat":
        y_test_actual = [list(set(sum([[j[0], j[:3], j] for j in tk], []))) for tk in y_test_actual]
        y_test_actual = hierarchical_binarizer.mlb_train.mlb.transform(y_test_actual)

        y_pred_exp = [list(set(sum([[j[0], j[:3], j] for j in tk], []))) for tk in y_pred_values]
        y_pred_values = hierarchical_binarizer.mlb_train.mlb.transform(y_pred_exp)

        y_pred_probs = binarizer.mlb_train.get_sklearn_mlb_from_pred(y_pred, prob_value=True)

        predictions = list()
        for probs in y_pred_probs:
            pred_dict = dict()
            for label, prob in zip(list(binarizer.mlb_train.mlb.classes_), probs):
                pred_dict[label] = prob
                if label[:3] not in pred_dict:
                    pred_dict[label[:3]] = list()
                pred_dict[label[:3]].append(prob)

                if label[:1] not in pred_dict:
                    pred_dict[label[:1]] = list()
                pred_dict[label[:1]].append(prob)

            for label in pred_dict:
                if len(label) == 3 or len(label) == 1:
                    pred_dict[label] = max(pred_dict[label])

            predictions.append(pred_dict)
        df_predictions = pd.DataFrame(predictions)
        df_predictions = df_predictions[list(hierarchical_binarizer.mlb_train.mlb.classes_)]
        flat_labels = df_predictions.columns.tolist()
        y_pred_probs = df_predictions.to_numpy()
    elif exp["model_type"] == "hier":
        y_test_actual = binarizer.mlb_train.mlb.transform(y_test_actual)
        y_pred_probs = binarizer.mlb_train.get_sklearn_mlb_from_pred(y_pred, prob_value=True)
        y_pred_values = binarizer.mlb_train.get_sklearn_mlb_from_pred(y_pred)
    else:
        raise ValueError("Unknown model type %s " % config["model_type"])

    precision_micro, recall_micro, f1_micro = precision_recall_f1(y_test_actual, y_pred_values, avg='micro')

    precision_macro, recall_macro, f1_macro = precision_recall_f1(
        hierarchical_binarizer.mlb_test.mlb.transform(hierarchical_binarizer.mlb_train.mlb.inverse_transform(y_test_actual)),
        hierarchical_binarizer.mlb_test.mlb.transform(hierarchical_binarizer.mlb_train.mlb.inverse_transform(y_pred_values)),
        avg='macro'
    )

    f1_avg = (2 * f1_macro * f1_micro) / (f1_macro + f1_micro)
    metric_dict = dict()
    metric_dict["precision_micro"] = round(precision_micro, 4)
    metric_dict["recall_micro"] = round(recall_micro, 4)
    metric_dict["f1_micro"] = round(f1_micro, 4)
    metric_dict["precision_macro"] = round(precision_macro, 4)
    metric_dict["recall_macro"] = round(recall_macro, 4)
    metric_dict["f1_macro"] = round(f1_macro, 4)
    metric_dict["f1_avg"] = round(f1_avg, 4)
    metric_dict["exp"] = exp["exp_label"]
    metric_dict["AUC"] = average_precision_score(y_test_actual, y_pred_probs, average="micro")
    metrics.append(metric_dict)

df = pd.DataFrame(metrics)
df = df[['exp', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro',
         'f1_avg', 'AUC']]
