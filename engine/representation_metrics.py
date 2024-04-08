import numpy as np
import torch

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import calinski_harabasz_score


def get_encoder_accuracy(calibration_tasks, evaluation_tasks, encoder):
    correctness = []
    probas = []
    labels = []
    for task in calibration_tasks:
        (X1, y1), (X2, y2), label = task
        encoding_1 = encoder(X1, y1)
        encoding_2 = encoder(X2, y2)
        probas.append(
            torch.exp(
                -torch.sqrt(((encoding_1 - encoding_2) ** 2).sum())
            ).item()
        )
        labels.append(label)
    calib = IsotonicRegression().fit(np.array(probas).reshape(-1, 1), labels)

    for task in evaluation_tasks:
        (X1, y1), (X2, y2), label = task
        encoding_1 = encoder(X1, y1)
        encoding_2 = encoder(X2, y2)
        prediction = int(
            calib.predict(
                [
                    torch.exp(
                        -torch.sqrt(((encoding_1 - encoding_2) ** 2).sum())
                    ).item()
                ]
            )
            >= 0.5
        )
        correctness.append(prediction == label)
    return np.mean(correctness)


def get_encoder_ch_index(tasks, labels, encoder):
    encodings = [encoder(X, y) for (X, y) in tasks]
    encodings = torch.stack(encodings).numpy()
    return calinski_harabasz_score(encodings, labels)


def get_metrics_summary(metrics):
    return {
        "mean": np.mean(metrics),
        "std": np.std(metrics),
        "min": np.min(metrics),
        "max": np.max(metrics),
    }
