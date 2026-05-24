"""Pipeline przetwarzania obrazu dla wymagań na 3.0 (bez uczenia maszynowego).

Etapy (zgodnie z treścią zadania):
  1. wstępne przetworzenie  — kanał zielony + CLAHE (wyrównanie kontrastu),
  2. właściwe przetworzenie  — filtr Frangi'ego (wykrywanie struktur naczyniowych),
  3. końcowe przetworzenie   — maskowanie FOV, progowanie, usuwanie drobnych artefaktów.

Wynik to binarna maska naczyń. Stanowi ona baseline (punkt odniesienia) dla
metod uczenia maszynowego z wymagań na 4.0/5.0.
"""

from __future__ import annotations

import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.filters import apply_hysteresis_threshold, frangi
from skimage.morphology import remove_small_objects
from skimage.util import img_as_float

from .data import green_channel


def preprocess_green(rgb: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """Kanał zielony → float [0,1] → CLAHE (adaptacyjne wyrównanie histogramu).

    Na kanale zielonym naczynia są ciemniejsze od tła; CLAHE lokalnie podbija
    kontrast, dzięki czemu cienkie naczynia stają się bardziej widoczne.
    """
    green = img_as_float(green_channel(rgb))
    return equalize_adapthist(green, clip_limit=clip_limit)


def vesselness(green: np.ndarray, sigmas=range(1, 8, 1)) -> np.ndarray:
    """Odpowiedź filtra Frangi'ego znormalizowana do [0,1].

    black_ridges=True — naczynia to ciemne "grzbiety" na kanale zielonym.
    Zakres `sigmas` odpowiada spodziewanym szerokościom naczyń (w pikselach).
    """
    resp = frangi(green, sigmas=sigmas, black_ridges=True)
    peak = resp.max()
    return resp / peak if peak > 0 else resp


def segment_vessels(
    rgb: np.ndarray,
    fov: np.ndarray | None = None,
    sigmas=range(1, 8, 1),
    low: float = 0.004,
    high: float = 0.02,
    min_size: int = 200,
    clip_limit: float = 0.01,
):
    """Pełny pipeline 3.0. Zwraca (maska_binarna, odpowiedz_frangi).

    Progowanie histerezowe (dwa progi): piksel powyżej `high` to pewne naczynie
    (ziarno), a piksel powyżej `low` zaliczamy do naczyń tylko gdy łączy się z
    ziarnem. Potem usuwamy izolowane składowe ≤ `min_size` px. Oba kroki obniżają
    liczbę fałszywych trafień (FP).

    low, high — progi histerezy na znormalizowanej odpowiedzi Frangi'ego,
    min_size  — maksymalny rozmiar usuwanej izolowanej składowej.
    """
    green = preprocess_green(rgb, clip_limit=clip_limit)
    resp = vesselness(green, sigmas=sigmas)
    if fov is not None:
        resp = resp * (np.asarray(fov) > 0)
    binary = apply_hysteresis_threshold(resp, low, high)
    if fov is not None:
        binary &= np.asarray(fov) > 0
    if min_size > 0:
        binary = remove_small_objects(binary, max_size=min_size)
    return binary, resp
