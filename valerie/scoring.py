"""Official and modified scoring functions."""
import copy
import json
import collections

import numpy as np


def validate_predictions_phase2(predictions):
    """validates if the predictions is 'valid' or not"""
    # check predictions is dict
    assert isinstance(predictions, dict)

    for claim_id, data in predictions.items():
        # this test shouldn't fail if loaded from json but why not
        # assert isinstance(claim_id, str)

        # if loaded from json, claim_id will be a str, so check if the str is a valid int
        if isinstance(claim_id, str):
            int(claim_id)
        # otherwise, claim_id must be an int
        else:
            assert isinstance(claim_id, int), type(claim_id)

        # make sure only 3 items exist
        assert len(data) == 3, len(data)

        # make sure the 3 items are the desired keys
        for field in data:
            assert field in ["related_articles", "explanation", "label"], field

        # check validity of related articles
        assert isinstance(data["related_articles"], dict), type(
            data["related_articles"]
        )
        for k, v in data["related_articles"].items():
            assert isinstance(k, str) or isinstance(k, int), type(k)
            assert int(k) in [1, 2], str(k)
            assert isinstance(v, str), type(v)

        # check validity of explanation
        assert isinstance(data["explanation"], str), type(data["explanation"])
        assert len(data["explanation"]) < 1000, len(data["explanation"])

        # check validity of label
        assert isinstance(data["label"], int), type(data["label"])
        assert data["label"] in {0, 1, 2}, data["label"]


def compute_score_phase2(labels, predictions):
    """
    predictions and labels are a dict
    check for duplicate relevant articles
    """
    n_class = collections.Counter([labels[o]["label"] for o in labels])
    scores = {"0": [], "1": [], "2": []}
    preds = []
    explanations = []
    if len(predictions) != len(labels):
        raise ValueError("prediction missing for some claims")
    # loop over predictions as the normalizing factor is n_class (# labels predicted)
    for claim_id in predictions:
        if len(predictions[claim_id]["explanation"]) > 1000:
            return {
                "score": 0.0,
                "explanation": "'N/S'",
                "error": "'MaxCharacterLimitError'",
                "predictions": "'N/A'",
            }
        pred = predictions[claim_id]["label"]
        preds.append(str(pred))
        label = labels[claim_id]["label"]
        if pred != label:
            scores[str(label)].append(0)
            continue
        rel_articles = list(predictions[claim_id]["related_articles"].values())
        if len(rel_articles) > 2:
            return {
                "score": 0.0,
                "explanation": "'N/S'",
                "error": "'MaxRelatedArticlesLimitError'",
                "predictions": "'N/A'",
            }
        # remove any duplicate url links
        rel_articles = set(rel_articles)
        gt_rel_articles = list(labels[claim_id]["related_articles"].values())
        scores[str(label)].append(
            sum([int(a in gt_rel_articles) for a in rel_articles])
        )
        explanations.append(predictions[claim_id]["explanation"].replace("'", ""))

    for l in scores:
        if not scores[l]:  # if scores[l] is [], np.mean returns a NaN
            scores[l] = 0.0
        else:
            scores[l] = sum(scores[l]) / n_class[int(l)]

    return {
        "score": np.mean(list(scores.values())),
        "error": "'None'",
        "explanation": "'{}'".format("|".join(explanations)),
        "predictions": "'[{}]'".format(",".join(preds)),
    }


def compute_detailed_score_phase2(labels, predictions):
    """
    predictions and labels are a dict
    check for duplicate relevant articles
    """
    from sklearn.metrics import classification_report

    n_class = collections.Counter([labels[o]["label"] for o in labels])
    perfect_clf_scores = {"0": [], "1": [], "2": []}
    scores = {"0": [], "1": [], "2": []}
    preds = []
    explanations = []
    if len(predictions) != len(labels):
        raise ValueError("prediction missing for some claims")
    # loop over predictions as the normalizing factor is n_class (# labels predicted)
    for claim_id in predictions:
        if len(predictions[claim_id]["explanation"]) > 1000:
            return {
                "score": 0.0,
                "explanation": "'N/S'",
                "error": "'MaxCharacterLimitError'",
                "predictions": "'N/A'",
            }
        pred = predictions[claim_id]["label"]
        preds.append(str(pred))
        label = labels[claim_id]["label"]

        rel_articles = list(predictions[claim_id]["related_articles"].values())
        if len(rel_articles) > 2:
            return {
                "score": 0.0,
                "explanation": "'N/S'",
                "error": "'MaxRelatedArticlesLimitError'",
                "predictions": "'N/A'",
            }
        # remove any duplicate url links
        rel_articles = set(rel_articles)
        gt_rel_articles = list(labels[claim_id]["related_articles"].values())

        _score = sum([int(a in gt_rel_articles) for a in rel_articles])
        perfect_clf_scores[str(label)].append(_score)
        if pred != label:
            scores[str(label)].append(0)
        else:
            scores[str(label)].append(_score)
        explanations.append(predictions[claim_id]["explanation"].replace("'", ""))

    for l in scores:
        if not scores[l]:
            scores[l] = 0.0
        else:
            scores[l] = sum(scores[l]) / n_class[int(l)]

    for l in perfect_clf_scores:
        if not perfect_clf_scores[l]:
            perfect_clf_scores[l] = 0.0
        else:
            perfect_clf_scores[l] = sum(perfect_clf_scores[l]) / n_class[int(l)]

    _labs = [labels[k]["label"] for k in labels]
    _prds = [predictions[k]["label"] for k in labels]
    report = classification_report(_labs, _prds)

    output = {
        "score": np.mean(list(scores.values())),
        "perfect_clf_score": np.mean(list(perfect_clf_scores.values())),
        "scores:": {
            "score_0": scores["0"],
            "score_1": scores["1"],
            "score_2": scores["2"],
        },
        "perfect_clf_scores": {
            "score_0": perfect_clf_scores["0"],
            "score_1": perfect_clf_scores["1"],
            "score_2": perfect_clf_scores["2"],
        },
    }

    # test official scoring function output is same as this one
    official_output = compute_score_phase2(labels, predictions)
    assert abs(output["score"] - official_output["score"]) <= 1e-9 * max(
        abs(output["score"]), abs(official_output["score"])
    )

    return report, output, official_output
