"""Wizualizacje: nakładka maski na obraz, mapa błędów, panel porównawczy."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.6) -> np.ndarray:
    """Zamalowuje piksele maski (naczynia) wyróżniającym się kolorem na obrazie."""
    out = rgb.astype(np.float64).copy()
    m = np.asarray(mask) > 0
    col = np.array(color, dtype=np.float64)
    out[m] = (1 - alpha) * out[m] + alpha * col
    return out.clip(0, 255).astype(np.uint8)


def error_map(pred: np.ndarray, truth: np.ndarray, fov: np.ndarray | None = None) -> np.ndarray:
    """Koloruje typy odpowiedzi: TP=zielony, FP=czerwony, FN=niebieski, TN=czarny."""
    p = np.asarray(pred) > 0
    t = np.asarray(truth) > 0
    h, w = p.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[p & t] = (0, 200, 0)        # TP
    out[p & ~t] = (220, 0, 0)       # FP
    out[~p & t] = (0, 80, 255)      # FN
    if fov is not None:
        out[~(np.asarray(fov) > 0)] = (40, 40, 40)
    return out


def comparison_panel(rgb, truth, pred, fov=None, title="", pred_label="Predykcja"):
    """Panel 1x4: obraz | maska ekspercka | predykcja | mapa błędów."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    axes[0].imshow(rgb)
    axes[0].set_title("Obraz (RGB)")
    axes[1].imshow(np.asarray(truth) > 0, cmap="gray")
    axes[1].set_title("Maska ekspercka (GT)")
    axes[2].imshow(np.asarray(pred) > 0, cmap="gray")
    axes[2].set_title(pred_label)
    axes[3].imshow(error_map(pred, truth, fov))
    axes[3].set_title("Mapa błędów\nTP=zielony FP=czerwony FN=niebieski")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
