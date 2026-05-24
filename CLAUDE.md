# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University project (IwM, "Projekt 2") for **retinal blood-vessel segmentation** in color fundus
photographs: per-pixel binary classification (vessel vs. background). The README and PDFs
(`DnoOka-wymagania.pdf` = requirements, `DnoOka.pdf` = implementation hints) are in Polish; the
project deliverables (notebooks, reports) are expected in Polish too.

**Current state: the repo contains only data + docs. No source code exists yet.** The directories
`notebooks/`, `src/`, and `outputs/` are described in the README as the intended layout but have not
been created. When you start implementing, create them.

## Three grading tiers (intended architecture)

The work is structured as three escalating approaches, each in its own notebook under `notebooks/`.
Higher tiers build on lower ones (3.0 is the baseline for 4.0; 4.0 is the reference point for 5.0):

| Grade | Notebook (planned)      | Method |
|-------|-------------------------|--------|
| 3.0   | `01_przetwarzanie_3`    | Pure image processing: pre-process → vesselness filter (e.g. Frangi) → post-process → threshold to binary mask |
| 4.0   | `02_klasyczne_ml_4`     | Classical ML: extract patches (e.g. 5×5 px), compute features (variance, central moments, Hu moments), train a scikit-learn classifier (kNN / decision tree / random forest / SVM) |
| 5.0   | `03_siec_5`             | Deep network (e.g. U-Net) trained on patches or full images (PyTorch / Keras / TF) |

Shared, reusable code (data loading, metrics, visualization) goes in `src/` and is imported by all
three notebooks. Generated masks and plots go in `outputs/`.

### Mandatory requirements for every tier

These apply to all three notebooks and are graded:
- Visualize the result as a binary mask **and** as an overlay on the original image.
- Evaluate on **≥ 5 images**, comparing against the expert ground-truth mask.
- Per-image metrics: **confusion matrix, accuracy, sensitivity (recall), specificity**, plus
  imbalanced-data measures (arithmetic & geometric mean of sensitivity and specificity).
- **Positive class = vessel, negative class = background.**

## Domain constraints (easy to get wrong — read before coding)

These come from `DnoOka.pdf` and are the cross-cutting gotchas that aren't obvious from any single file:
- **Masks are encoded as 0 and 255**, not 0/1. Normalize to 0/1 before computing any metric.
- The **green (G) channel** carries the most vessel information — prefer it over RGB/grayscale for
  the classical pipeline and feature extraction.
- The **FOV (field-of-view) edge** creates a bright ring that the Frangi filter mistakes for a
  vessel. Mask out the FOV before/after filtering (HRF ships explicit FOV masks in `data/hrf/mask/`).
- The dataset is **heavily class-imbalanced** (few vessel pixels) — this is why the balanced
  mean metrics are required, and why raw accuracy alone is misleading.

## Data

Two datasets. **STARE is the primary set** — base solutions and model training on it (it has two
independent expert label sets, `ah` and `vk`, which can be compared or averaged). HRF is auxiliary.

```
data/
├── stare/              # PRIMARY — 20 images, RGB 605×700
│   ├── images/         #   im0001.ppm ...
│   ├── labels-ah/      #   im0001.ah.ppm ...   expert A. Hoover (binary 0/255)
│   └── labels-vk/      #   im0001.vk.ppm ...   expert V. Kouznetsova (binary 0/255)
└── hrf/                # AUXILIARY — 45 images (15 each: _h healthy, _dr diabetic, _g glaucoma), RGB 2336×3504
    ├── images/         #   01_h.jpg ...
    ├── manual1/        #   01_h.tif ...        expert vessel mask (binary 0/255)
    └── mask/           #   01_h_mask.tif ...   FOV mask
```

`data/` is currently extracted and present. To reconstruct it from the source archives in
`archives/` (STARE `.ppm.gz` files must be gunzipped after extraction):

```bash
mkdir -p data/stare/{images,labels-ah,labels-vk}
tar -xf archives/stare-images.tar -C data/stare/images   && gunzip -f data/stare/images/*.gz
tar -xf archives/labels-ah.tar    -C data/stare/labels-ah && gunzip -f data/stare/labels-ah/*.gz
tar -xf archives/labels-vk.tar    -C data/stare/labels-vk && gunzip -f data/stare/labels-vk/*.gz
mkdir -p data/hrf && unzip -q archives/all.zip -d data/hrf
```

## Conventions

- `.gitignore` policy: **commit notebooks with outputs cleared** (`.ipynb_checkpoints/` is ignored).
  Trained model artifacts are deliberately not committed (`*.ckpt`, `*.pth`, `*.h5`, `*.keras`,
  `*.joblib`, `*.pkl` are all ignored) — regenerate them from the notebooks.
- No dependency manifest (`requirements.txt` / `pyproject.toml`) exists yet; create one when you add code.
