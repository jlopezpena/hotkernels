"""Scores a classifier's predictions with scikit-learn's metrics.

What follows every trained model: ranking metrics and a calibration curve for
a binary classifier, and a confusion matrix and per-class report for a
multi-class one whose prediction is the argmax of its probabilities.
"""
from __future__ import annotations

import numpy as np

TITLE = "Scoring a classifier: ROC, precision-recall, calibration, confusion matrix"
SOURCE = "scikit-learn (sklearn.metrics, sklearn.calibration), called as a library"
LICENCE = None
REQUIRES = ("sklearn",)

# Predictions scored in the binary problem.
BINARY = 1_000_000
# Predictions and classes in the multi-class problem.
MULTI, CLASSES = 300_000, 10
# Bins of the calibration curve.
BINS = 10
# Probability above which the binary classifier predicts the positive class.
THRESHOLD = 0.5
# Seed of the labels and scores.
SEED = 0
SIZE = f"{BINARY:,} binary predictions, {MULTI:,} predictions over {CLASSES} classes"


def make_data() -> tuple:
    """Build labels and the scores of a classifier that is right more often than not.

    Returns:
        (binary labels as int64, positive-class probabilities as float64,
        multi-class labels as int64, class probabilities as float64 rows).
    """
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, 2, BINARY)
    score = 1 / (1 + np.exp(-(rng.standard_normal(BINARY) + 1.5 * (y - 0.5))))
    labels = rng.integers(0, CLASSES, MULTI)
    logits = rng.standard_normal((MULTI, CLASSES))
    logits[np.arange(MULTI), labels] += 1.5
    proba = np.exp(logits)
    proba /= proba.sum(axis=1, keepdims=True)
    return y, score, labels, proba


def run(y: np.ndarray, score: np.ndarray, labels: np.ndarray, proba: np.ndarray) -> tuple:
    """Score both classifiers.

    Args:
        y: Binary labels.
        score: Predicted probability of the positive class.
        labels: Multi-class labels.
        proba: Predicted probability of each class, one example per row.

    Returns:
        Every metric computed, as scalars and arrays.
    """
    from sklearn import metrics
    from sklearn.calibration import calibration_curve

    predicted = (score > THRESHOLD).astype(int)
    fpr, tpr, _ = metrics.roc_curve(y, score)
    precision, recall, _ = metrics.precision_recall_curve(y, score)
    fraction, mean_predicted = calibration_curve(y, score, n_bins=BINS)
    binary = (
        metrics.roc_auc_score(y, score), metrics.average_precision_score(y, score), fpr, tpr, precision, recall,
        fraction, mean_predicted, metrics.brier_score_loss(y, score), metrics.log_loss(y, score),
        metrics.accuracy_score(y, predicted), metrics.f1_score(y, predicted), metrics.confusion_matrix(y, predicted),
    )
    chosen = np.argmax(proba, axis=1)
    multi = (
        metrics.confusion_matrix(labels, chosen), metrics.accuracy_score(labels, chosen),
        metrics.classification_report(labels, chosen), metrics.log_loss(labels, proba),
        metrics.top_k_accuracy_score(labels, proba, k=3),
    )
    return (*binary, *multi)
