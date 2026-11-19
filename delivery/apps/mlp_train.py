"""Trains a two-layer perceptron in NumPy by mini-batch gradient descent.

Written for this study, the way such a loop is usually written by hand: a
forward pass, a softmax loss, a backward pass, and the accuracy on held-out
data after each epoch. It knows nothing of hotkernels.
"""
from __future__ import annotations

import numpy as np

TITLE = "Training a two-layer perceptron by mini-batch gradient descent"
SOURCE = "written for this study"
LICENCE = None
REQUIRES = ()

# Training examples and held-out examples, at either size.
TRAIN, HELD_OUT = 12_000, 2_000
# Classes.
CLASSES = 10
# Examples per gradient step, and passes over the training set.
BATCH, EPOCHS = 128, 2
# Step size, and the weight of the L2 penalty.
LEARNING_RATE, PENALTY = 0.1, 1e-4
# Seed of the data and of the initial weights.
SEED = 0
# Inputs per example and hidden units, by size. The large network takes a
# 28 x 28 image and its weight matrix has 200,704 elements; the small one takes
# a 14 x 14 image and has 12,544.
NETWORKS = {"small": (196, 64), "large": (784, 256)}
# One line per size saying how large its data is.
SIZES = {
    size: f"{TRAIN:,} examples of {inputs} float64 inputs, {hidden} hidden units, batches of {BATCH}, {EPOCHS} epochs"
    for size, (inputs, hidden) in NETWORKS.items()
}


def make_data(size: str) -> tuple:
    """Build a classification problem a small network can learn.

    Args:
        size: One of SIZES.

    Returns:
        (training inputs, training labels, held-out inputs, held-out labels,
        hidden units). The labels come from a random linear teacher, so
        accuracy rises during training.
    """
    inputs, hidden = NETWORKS[size]
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((TRAIN + HELD_OUT, inputs))
    teacher = rng.standard_normal((inputs, CLASSES))
    y = np.argmax(x @ teacher, axis=1)
    return x[:TRAIN], y[:TRAIN], x[TRAIN:], y[TRAIN:], hidden


def forward(params: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the network on a batch.

    Args:
        params: The weights and biases.
        x: Inputs, one example per row.

    Returns:
        (hidden activations, class scores).
    """
    hidden = np.maximum(0, x @ params["w1"] + params["b1"])
    return hidden, hidden @ params["w2"] + params["b2"]


def loss_and_gradients(params: dict, x: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """Compute the softmax loss of a batch and its gradient.

    Args:
        params: The weights and biases.
        x: Inputs, one example per row.
        y: The class of each example.

    Returns:
        (mean loss with the L2 penalty, gradient of each parameter).
    """
    n = x.shape[0]
    hidden, scores = forward(params, x)
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), y]).mean()
    loss += 0.5 * PENALTY * (np.sum(params["w1"] ** 2) + np.sum(params["w2"] ** 2))

    dscores = probs
    dscores[np.arange(n), y] -= 1
    dscores /= n
    dhidden = dscores @ params["w2"].T
    dhidden[hidden <= 0] = 0
    return float(loss), {
        "w2": hidden.T @ dscores + PENALTY * params["w2"],
        "b2": dscores.sum(axis=0),
        "w1": x.T @ dhidden + PENALTY * params["w1"],
        "b1": dhidden.sum(axis=0),
    }


def accuracy(params: dict, x: np.ndarray, y: np.ndarray) -> float:
    """Return the share of examples the network classifies correctly."""
    return float(np.mean(np.argmax(forward(params, x)[1], axis=1) == y))


def run(x: np.ndarray, y: np.ndarray, x_held: np.ndarray, y_held: np.ndarray, hidden: int) -> tuple:
    """Train the network and measure it.

    Args:
        x: Training inputs.
        y: Training labels.
        x_held: Held-out inputs.
        y_held: Held-out labels.
        hidden: Hidden units.

    Returns:
        (the four parameters, the loss of every step, the held-out accuracy
        after every epoch).
    """
    inputs = x.shape[1]
    rng = np.random.default_rng(SEED + 1)
    params = {
        "w1": rng.standard_normal((inputs, hidden)) * np.sqrt(2.0 / inputs),
        "b1": np.zeros(hidden),
        "w2": rng.standard_normal((hidden, CLASSES)) * np.sqrt(2.0 / hidden),
        "b2": np.zeros(CLASSES),
    }
    losses, accuracies = [], []
    for _ in range(EPOCHS):
        order = rng.permutation(x.shape[0])
        for start in range(0, x.shape[0], BATCH):
            batch = order[start:start + BATCH]
            loss, gradients = loss_and_gradients(params, x[batch], y[batch])
            for name in params:
                params[name] -= LEARNING_RATE * gradients[name]
            losses.append(loss)
        accuracies.append(accuracy(params, x_held, y_held))
    return (*(params[name] for name in sorted(params)), np.array(losses), np.array(accuracies))
