# Analiza dna oka — segmentacja naczyń krwionośnych

Projekt 2 z przedmiotu IwM. Dla obrazu dna siatkówki oka (color fundus photography)
automatycznie wykrywamy naczynia krwionośne — **binarna klasyfikacja każdego piksela**
(naczynie vs tło).

Treść zadania: [`DnoOka-wymagania.pdf`](DnoOka-wymagania.pdf), wskazówki implementacyjne:
[`DnoOka.pdf`](DnoOka.pdf).

## Poziomy wymagań

Każdy poziom realizujemy w osobnym notebooku w `notebooks/`. Poziomy wyższe zawierają
niższe (3 jest baseline'em dla 4, a 4 punktem odniesienia dla 5).

| Ocena | Metoda | Opis |
|-------|--------|------|
| **3.0** | Przetwarzanie obrazu | Pre-processing → filtr naczyniowy (np. Frangi) → post-processing → progowanie do maski binarnej |
| **4.0** | Klasyczne ML | Wycinki (np. 5×5 px), ekstrakcja cech (wariancja, momenty centralne, momenty Hu), klasyfikator scikit-learn (kNN / drzewo / las / SVM) |
| **5.0** | Głęboka sieć | Sieć neuronowa (np. U-Net), uczona na wycinkach lub całych obrazach (PyTorch / Keras / TF) |

Wspólne dla wszystkich poziomów (**wymagania obowiązkowe**):
- wizualizacja wyniku jako maski binarnej + nałożenie na obraz,
- test na min. 5 obrazach, porównanie z maską ekspercką (ground truth),
- metryki per obraz: macierz pomyłek, **accuracy, sensitivity (czułość), specificity (swoistość)**,
  oraz miary dla danych niezrównoważonych (średnia arytmetyczna/geometryczna czułości i swoistości).
  Klasa pozytywna = naczynie, negatywna = tło.

## Dane

Używamy dwóch baz. **Zbiorem głównym jest STARE** — na nim opieramy rozwiązania i
uczenie modeli (są dwa niezależne zestawy etykiet eksperckich). HRF jest pomocniczy.

```
data/
├── stare/                  # GŁÓWNY zbiór — STARE database (20 obrazów)
│   ├── images/             #   im0001.ppm ...        obraz RGB 605×700
│   ├── labels-ah/          #   im0001.ah.ppm ...     etykiety eksperta A. Hoover (binarne 0/255)
│   └── labels-vk/          #   im0001.vk.ppm ...     etykiety eksperta V. Kouznetsova (binarne 0/255)
└── hrf/                    # POMOCNICZY — HRF database (45 obrazów: 15× _h zdrowe, _dr cukrzyca, _g jaskra)
    ├── images/             #   01_h.jpg ...          obraz RGB 2336×3504
    ├── manual1/            #   01_h.tif ...          maska ekspercka naczyń (binarna 0/255)
    └── mask/               #   01_h_mask.tif ...     maska pola widzenia (FOV)
```

Istotne uwagi techniczne (z `DnoOka.pdf`):
- maski są kodowane wartościami **0 i 255** — przy liczeniu metryk normalizować do 0/1,
- najwięcej informacji o naczyniach niesie **kanał zielony** (G) obrazu,
- przy filtrze Frangi'ego uważać na „ramkę" wokół obrazu (krawędź FOV) — maskować FOV,
- STARE: 20 obrazów, każdy z dwoma niezależnymi zestawami etykiet (ah, vk) — można je
  porównać lub uśrednić.

### Źródło i odtworzenie danych

Oryginalne archiwa trzymamy w `archives/`: `stare-images.tar`, `labels-ah.tar`,
`labels-vk.tar` (STARE) oraz `all.zip` (HRF). Katalog `data/` odtwarzamy z nich
poleceniami poniżej (pliki `.ppm.gz` w STARE wymagają dekompresji):

```bash
# STARE — obrazy i dwa zestawy etykiet (pliki .ppm.gz wymagają dekompresji)
mkdir -p data/stare/{images,labels-ah,labels-vk}
tar -xf archives/stare-images.tar -C data/stare/images && gunzip -f data/stare/images/*.gz
tar -xf archives/labels-ah.tar   -C data/stare/labels-ah && gunzip -f data/stare/labels-ah/*.gz
tar -xf archives/labels-vk.tar   -C data/stare/labels-vk && gunzip -f data/stare/labels-vk/*.gz

# HRF
mkdir -p data/hrf && unzip -q archives/all.zip -d data/hrf
```

Bazy źródłowe: STARE — http://cecas.clemson.edu/~ahoover/stare/probing/ ·
HRF — https://www5.cs.fau.de/research/data/fundus-images/

## Struktura projektu

```
.
├── data/          # rozpakowane dane (STARE + HRF) — patrz wyżej
├── archives/      # oryginalne archiwa źródłowe
├── notebooks/     # 01_przetwarzanie_3, 02_klasyczne_ml_4, 03_siec_5
├── src/           # współdzielony kod (wczytywanie danych, metryki, wizualizacje)
└── outputs/       # wygenerowane maski, wykresy
```
