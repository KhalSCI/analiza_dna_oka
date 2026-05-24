"""Wczytywanie danych STARE oraz wyznaczanie maski pola widzenia (FOV).

STARE (zbiór główny): 20 obrazów RGB 700x605 w `data/stare/images/imXXXX.ppm`,
dwa niezależne zestawy etykiet eksperckich (binarne 0/255):
  - `data/stare/labels-ah/imXXXX.ah.ppm`  (ekspert A. Hoover)
  - `data/stare/labels-vk/imXXXX.vk.ppm`  (ekspert V. Kouznetsova)

STARE nie dostarcza gotowej maski FOV (w odróżnieniu od HRF), więc wyznaczamy ją
z obrazu: dno oka to jasne koło na niemal czarnym tle.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.morphology import disk, erosion
from skimage.transform import rescale

REPO_ROOT = Path(__file__).resolve().parent.parent
STARE_DIR = REPO_ROOT / "data" / "stare"
IMAGES_DIR = STARE_DIR / "images"
LABELS = {"ah": STARE_DIR / "labels-ah", "vk": STARE_DIR / "labels-vk"}

HRF_DIR = REPO_ROOT / "data" / "hrf"
HRF_IMAGES = HRF_DIR / "images"
HRF_MANUAL = HRF_DIR / "manual1"
HRF_MASK = HRF_DIR / "mask"


def list_stare_ids() -> list[str]:
    """Posortowana lista identyfikatorów obrazów, np. ['im0001', 'im0002', ...]."""
    ids = [p.stem for p in IMAGES_DIR.glob("*.ppm")]
    return sorted(ids, key=lambda s: int(re.sub(r"\D", "", s) or 0))


def load_image(image_id: str) -> np.ndarray:
    """Wczytuje obraz RGB jako uint8 o kształcie (H, W, 3)."""
    img = imread(IMAGES_DIR / f"{image_id}.ppm")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3]


def load_label(image_id: str, expert: str = "ah") -> np.ndarray:
    """Wczytuje maskę ekspercką i normalizuje do binarnej {0, 1} (bool).

    Etykiety są kodowane 0/255; klasa pozytywna (naczynie) = 1.
    """
    if expert not in LABELS:
        raise ValueError(f"expert must be one of {list(LABELS)}, got {expert!r}")
    mask = imread(LABELS[expert] / f"{image_id}.{expert}.ppm")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 127


def green_channel(rgb: np.ndarray) -> np.ndarray:
    """Kanał zielony (G) — niesie najwięcej informacji o naczyniach."""
    return rgb[..., 1]


# --- HRF (zbiór pomocniczy) ---------------------------------------------------
# 45 obrazów RGB 3504x2336 w trzech kategoriach: _h (zdrowe), _dr (cukrzyca),
# _g (jaskra). HRF dostarcza GOTOWE maski: naczyń (manual1/) i pola widzenia (mask/).
# Obrazy są ~5x większe od STARE, więc domyślnie skalujemy je w dół (scale<1),
# dzięki czemu te same parametry filtra (sigmas) działają i przetwarzanie jest szybkie.

def list_hrf_ids() -> list[str]:
    """Identyfikatory HRF, np. ['01_dr', '01_g', '01_h', '02_dr', ...]."""
    ids = {p.stem for p in HRF_IMAGES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg"}}
    return sorted(ids, key=lambda s: (int(s.split("_")[0]), s.split("_")[1]))


def _hrf_image_path(image_id: str) -> Path:
    # rozszerzenie bywa .jpg lub .JPG zależnie od kategorii — szukamy po nazwie
    hits = sorted(HRF_IMAGES.glob(f"{image_id}.*"))
    if not hits:
        raise FileNotFoundError(f"Brak obrazu HRF dla {image_id!r}")
    return hits[0]


def load_hrf_image(image_id: str, scale: float = 0.2) -> np.ndarray:
    """Obraz RGB uint8; domyślnie pomniejszony do ~1/5 (≈ rozmiar STARE)."""
    img = imread(_hrf_image_path(image_id))[..., :3]
    if scale != 1.0:
        img = rescale(img, scale, channel_axis=-1, anti_aliasing=True,
                      preserve_range=True).astype(np.uint8)
    return img


def load_hrf_label(image_id: str, scale: float = 0.2) -> np.ndarray:
    """Maska ekspercka naczyń, binarna {0,1} (bool), w tej samej skali co obraz."""
    # maski HRF to TIFF-y z kompresją LZW/PackBits — czytamy je przez Pillow
    mask = np.asarray(Image.open(HRF_MANUAL / f"{image_id}.tif"))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 127
    if scale != 1.0:
        mask = rescale(mask.astype(float), scale, anti_aliasing=False, order=0) > 0.5
    return mask


def load_hrf_fov(image_id: str, scale: float = 0.2, erode_px: int = 5) -> np.ndarray:
    """Gotowa maska pola widzenia (FOV) z HRF, pomniejszona i lekko zerodowana.

    Erozja odsuwa granicę FOV od krawędzi koła — ostra krawędź dno/tło jest dla
    filtra Frangi'ego nie do odróżnienia od naczynia (ta sama uwaga co dla STARE).
    """
    fov = np.asarray(Image.open(HRF_MASK / f"{image_id}_mask.tif"))
    if fov.ndim == 3:
        fov = fov[..., 0]
    fov = fov > 127
    if scale != 1.0:
        fov = rescale(fov.astype(float), scale, anti_aliasing=False, order=0) > 0.5
    if erode_px > 0:
        fov = erosion(fov, disk(erode_px))
    return fov


def extract_fov_mask(
    rgb: np.ndarray, threshold: int = 30, erode_px: int = 5
) -> np.ndarray:
    """Wyznacza maskę pola widzenia (FOV) — jasne koło dna oka na czarnym tle.

    Progujemy jasność, zamykamy dziury, a następnie erodujemy o `erode_px`, aby
    odsunąć granicę FOV od krawędzi koła. To kluczowe: filtr Frangi'ego reaguje na
    ostre przejście tło→dno jak na naczynie ("ramka"), więc kilka pikseli przy
    brzegu trzeba wykluczyć.
    """
    luminance = rgb.astype(np.float64).mean(axis=-1)
    fov = luminance > threshold
    fov = ndi.binary_fill_holes(fov)
    if erode_px > 0:
        fov = erosion(fov, disk(erode_px))
    return fov
