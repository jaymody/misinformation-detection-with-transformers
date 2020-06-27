"""Official and modified scoring functions."""
import json
import copy

import numpy as np


def validate_predictions_phase2(predictions):
    """validates if the predictions dictionary is 'valid' or not"""
    req_fields = ["related_articles", "explanation", "label"]
    try:
        for claim_id, objects in predictions.items():
            for f in req_fields:
                if f not in objects.keys():
                    return {}, False
        return predictions, True
    except Exception as e:
        return {}, False


def compute_score_phase2(labels, predictions):
    """Adapted for new submission format."""
    _predictions = copy.deepcopy(predictions)
    for pred in _predictions.values():
        pred["related_articles"] = list(pred["related_articles"].values())
    return _compute_score_phase2(labels, _predictions)


def _compute_score_phase2(labels, predictions):
    """
    predictions and labels are a dict
    check for duplicate relevant articles
    """
    scores = {"0": [], "1": [], "2": []}
    preds = []
    explanations = []
    if len(predictions) != len(labels):
        print("prediction missing for some claims")
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
            scores[str(pred)].append(0)
            continue
        rel_articles = predictions[claim_id]["related_articles"]
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
        scores[str(pred)].append(sum([int(a in gt_rel_articles) for a in rel_articles]))
        explanations.append(predictions[claim_id]["explanation"])

    for l in scores:
        if not scores[l]:
            scores[l] = 0.0
        else:
            scores[l] = np.mean(scores[l])

    return {
        "score": np.mean(list(scores.values())),
        "error": "'None'",
        "explanation": "'{}'".format("|".join(explanations)),
        "predictions": "'[{}]'".format(",".join(preds)),
    }


def compute_detailed_score_phase2(labels, predictions):
    """Adapted for new submission format."""
    _predictions = copy.deepcopy(predictions)
    for pred in _predictions.values():
        pred["related_articles"] = list(pred["related_articles"].values())
    return _compute_detailed_score_phase2(labels, _predictions)


def _compute_detailed_score_phase2(labels, predictions):
    from sklearn.metrics import classification_report

    perfect_scores = {"0": [], "1": [], "2": []}
    scores = {"0": [], "1": [], "2": []}
    preds = []
    if len(predictions) != len(labels):
        print("prediction missing for some claims")
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

        rel_articles = predictions[claim_id]["related_articles"]
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

        perfect_scores[str(pred)].append(
            sum([int(a in gt_rel_articles) for a in rel_articles])
        )
        if pred != label:
            scores[str(pred)].append(0)
            continue
        else:
            scores[str(pred)].append(
                sum([int(a in gt_rel_articles) for a in rel_articles])
            )

    for l in scores:
        if not scores[l]:
            scores[l] = 0.0
        else:
            scores[l] = np.mean(scores[l])

    for l in perfect_scores:
        if not perfect_scores[l]:
            perfect_scores[l] = 0.0
        else:
            perfect_scores[l] = np.mean(perfect_scores[l])

    _labs, _prds = zip(*[(labels[k]["label"], predictions[k]["label"]) for k in labels])
    report = classification_report(_labs, _prds)

    output = {
        "score": np.mean(list(scores.values())),
        "perfect_score": np.mean(list(perfect_scores.values())),
        "scores:": {
            "score_0": scores["0"],
            "score_1": scores["1"],
            "score_2": scores["2"],
        },
        "perfect_scores": {
            "score_0": perfect_scores["0"],
            "score_1": perfect_scores["1"],
            "score_2": perfect_scores["2"],
        },
    }
    official_output = _compute_score_phase2(labels, predictions)
    assert abs(output["score"] - official_output["score"]) <= 1e-9 * max(
        abs(output["score"]), abs(official_output["score"])
    )

    return report, output, official_output
