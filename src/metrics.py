"""Metryki oceny segmentacji naczyń (klasa pozytywna = naczynie).

Wszystkie metryki liczone są na pikselach wewnątrz FOV (poza FOV nie ma danych
eksperckich, a tło i tak dominowałoby wynik). Maski normalizujemy do {0,1}.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import confusion_matrix


@dataclass
class Metrics:
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    sensitivity: float  # czułość (recall klasy naczynie)
    specificity: float  # swoistość
    arithmetic_mean: float  # średnia arytmetyczna czułości i swoistości
    geometric_mean: float  # średnia geometryczna czułości i swoistości

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(pred: np.ndarray, truth: np.ndarray, fov: np.ndarray | None = None) -> Metrics:
    """Porównuje maskę predykcji z maską ekspercką i zwraca komplet metryk.

    pred, truth, fov: tablice 2D; dowolne kodowanie (0/1, bool, 0/255) — binaryzujemy.
    Jeśli podano `fov`, ocena obejmuje wyłącznie piksele wewnątrz FOV.
    """
    pred_b = np.asarray(pred) > 0
    truth_b = np.asarray(truth) > 0
    if fov is not None:
        sel = np.asarray(fov) > 0
        pred_b = pred_b[sel]
        truth_b = truth_b[sel]
    else:
        pred_b = pred_b.ravel()
        truth_b = truth_b.ravel()

    tn, fp, fn, tp = confusion_matrix(
        truth_b, pred_b, labels=[False, True]
    ).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    arithmetic = (sensitivity + specificity) / 2
    geometric = float(geometric_mean_score(truth_b, pred_b, labels=[False, True]))

    return Metrics(
        tn=tn, fp=fp, fn=fn, tp=tp,
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        arithmetic_mean=arithmetic,
        geometric_mean=geometric,
    )
