"""Ekstrakcja cech dla klasyfikatora (wymagania na 4.0).

Dla każdego piksela budujemy wektor cech opisujący jego **otoczenie** (patch).
Zamiast pętli po wycinkach liczymy cechy jako **mapy na całym obrazie** (filtry),
a potem dla danego piksela odczytujemy wartości ze wszystkich map — to ten sam
pomysł (cechy z otoczenia), ale wektoryzowany, więc szybki i policzalny dla
każdego piksela obrazu testowego.

Zestaw cech (z kanału zielonego po poprawie kontrastu):
  - jasność piksela,
  - rozmycia Gaussa w kilku skalach (kontekst o różnym zasięgu),
  - lokalna średnia i odchylenie std. (= 2. moment centralny) w oknach 5/9/15,
  - lokalne min/max (okno 9),
  - moduł gradientu (Sobel),
  - różnice Gaussów (DoG) — wzmacniają struktury o danej grubości,
  - odpowiedź filtra Frangi'ego (cecha „naczyniowatości" z poziomu 3.0).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, sobel

from .processing import preprocess_green, vesselness


def feature_maps(rgb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Zwraca (stos_cech (H,W,F), nazwy_cech)."""
    g = preprocess_green(rgb)  # kanał zielony + CLAHE, float [0,1]
    feats, names = [], []

    def add(name, m):
        feats.append(m.astype(np.float32)); names.append(name)

    add("green", g)
    for s in (1, 2, 4, 8):
        add(f"gauss{s}", gaussian(g, sigma=s))
    for w in (5, 9, 15):
        mean = ndi.uniform_filter(g, w)
        var = np.clip(ndi.uniform_filter(g * g, w) - mean * mean, 0, None)
        add(f"mean{w}", mean)
        add(f"std{w}", np.sqrt(var))
    add("min9", ndi.minimum_filter(g, 9))
    add("max9", ndi.maximum_filter(g, 9))
    add("sobel", sobel(g))
    add("dog13", gaussian(g, 1) - gaussian(g, 3))
    add("dog26", gaussian(g, 2) - gaussian(g, 6))
    add("frangi", vesselness(g))

    return np.stack(feats, axis=-1), names


def build_dataset(
    images, fovs, labels, n_per_class_per_image: int = 8000, seed: int = 0
):
    """Buduje zrównoważony zbiór (X, y) z listy obrazów.

    Dla każdego obrazu losujemy po `n_per_class_per_image` pikseli-naczyń i pikseli-tła
    (z wnętrza FOV) — to undersampling wyrównujący silnie niezrównoważone klasy w
    zbiorze UCZĄCYM (ocena na obrazach testowych zostaje na pełnym, realnym rozkładzie).
    """
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    names = None
    for rgb, fov, gt in zip(images, fovs, labels):
        maps, names = feature_maps(rgb)
        fov_b = np.asarray(fov) > 0
        vessel = fov_b & (np.asarray(gt) > 0)
        backgr = fov_b & ~(np.asarray(gt) > 0)
        for cls_mask in (vessel, backgr):
            ys, xs = np.where(cls_mask)
            if len(ys) == 0:
                continue
            k = min(n_per_class_per_image, len(ys))
            idx = rng.choice(len(ys), size=k, replace=False)
            X_parts.append(maps[ys[idx], xs[idx]])
            y_parts.append((np.asarray(gt) > 0)[ys[idx], xs[idx]])
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts).astype(int)
    return X, y, names


def predict_mask(clf, rgb: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Predykcja klasyfikatora dla wszystkich pikseli w FOV → maska binarna (H,W)."""
    maps, _ = feature_maps(rgb)
    h, w, f = maps.shape
    fov_flat = (np.asarray(fov) > 0).reshape(-1)
    pred = np.zeros(h * w, dtype=bool)
    pred[fov_flat] = clf.predict(maps.reshape(-1, f)[fov_flat]).astype(bool)
    return pred.reshape(h, w)
