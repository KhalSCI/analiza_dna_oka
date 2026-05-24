# Jak rozumieć metryki oceny segmentacji naczyń

Ten dokument tłumaczy miary, którymi oceniamy jakość wykrywania naczyń, na
konkretnych liczbach z naszego projektu. Wszystkie metryki liczymy **tylko
wewnątrz pola widzenia (FOV)** — poza nim nie ma danych eksperckich, a czarne tło
sztucznie zawyżałoby wyniki.

## Punkt wyjścia: co jest klasą pozytywną

Dla **każdego piksela** algorytm podejmuje decyzję binarną:

- **klasa pozytywna (1) = naczynie krwionośne**,
- **klasa negatywna (0) = tło** (nie-naczynie).

Kluczowa cecha problemu: **klasy są silnie niezrównoważone**. Naczynia to tylko
~8–13% pikseli w FOV, reszta to tło. To niezrównoważenie sprawia, że niektóre
metryki (np. sama trafność) potrafią wprowadzać w błąd — o tym niżej.

## Macierz pomyłek (confusion matrix)

Porównujemy maskę algorytmu z maską ekspercką (ground truth) piksel po pikselu.
Każdy piksel trafia do jednej z czterech kategorii:

|                          | ekspert: naczynie | ekspert: tło |
|--------------------------|-------------------|--------------|
| **algorytm: naczynie**   | **TP** (trafienie)| **FP** (fałszywy alarm) |
| **algorytm: tło**        | **FN** (przeoczenie) | **TN** (poprawne odrzucenie) |

W naszym zadaniu:

- **TP** (*true positive*) — piksel naczynia poprawnie wykryty jako naczynie. *Na mapie błędów: zielony.*
- **FP** (*false positive*) — piksel tła błędnie uznany za naczynie. To są właśnie te
  „naczynka, których nie ma w masce eksperckiej". *Na mapie błędów: czerwony.*
- **FN** (*false negative*) — prawdziwe naczynie, którego algorytm nie wykrył (przeoczenie). *Na mapie błędów: niebieski.*
- **TN** (*true negative*) — piksel tła poprawnie zostawiony jako tło. *Na mapie błędów: czarny.*

**Przykład — obraz `im0001`** (z `outputs/01_przetwarzanie_3/metryki.csv`):

| TP | FP | FN | TN |
|----|----|----|----|
| 27 794 | 47 356 | 5 294 | 222 525 |

Razem 302 969 pikseli w FOV; z tego naczyń (TP+FN) = 33 088 (≈ 10,9%).

## Trafność (accuracy)

> **accuracy = (TP + TN) / (TP + TN + FP + FN)** — odsetek wszystkich poprawnie sklasyfikowanych pikseli.

Dla `im0001`: (27 794 + 222 525) / 302 969 = **0,826**.

⚠️ **Dlaczego sama trafność myli przy niezrównoważeniu.** Wyobraźmy sobie trywialny
„klasyfikator", który **każdy piksel uznaje za tło**. Na `im0001` osiągnąłby:

- accuracy = 269 881 / 302 969 = **0,891** — *wyższą* niż nasz algorytm!
- ...ale wykryłby **zero** naczyń (czułość = 0).

To pokazuje sedno problemu: można mieć ~89% trafności, nie wykrywając ani jednego
naczynia. Dlatego trafność raportujemy, ale **nie opieramy na niej oceny** — patrzymy
na czułość, swoistość i ich średnie.

## Czułość (sensitivity / recall / TPR)

> **czułość = TP / (TP + FN)** — jaki odsetek *prawdziwych naczyń* udało się wykryć.

Patrzy tylko na piksele, które naprawdę są naczyniami. Odpowiada na pytanie:
„ile naczyń złapaliśmy, a ile przeoczyliśmy?".

Dla `im0001`: 27 794 / (27 794 + 5 294) = **0,840** → wykryto 84% naczyń.

- **Wysoka czułość** = mało przeoczeń (FN), wykrywamy nawet cienkie naczynia.
- **Niska czułość** = gubimy naczynia (dużo niebieskiego na mapie błędów) — tak jest
  na `im0004` (czułość 0,51), obrazie o niskim kontraście.

## Swoistość (specificity / TNR)

> **swoistość = TN / (TN + FP)** — jaki odsetek *prawdziwego tła* poprawnie zostawiliśmy jako tło.

Patrzy tylko na piksele tła. Odpowiada na pytanie: „jak bardzo algorytm się myli,
zaznaczając tło jako naczynie?".

Dla `im0001`: 222 525 / (222 525 + 47 356) = **0,825** → 82,5% tła poprawnie odrzucone,
czyli 17,5% tła to fałszywe alarmy (FP).

- **Wysoka swoistość** = mało fałszywych trafień (mało czerwonego na mapie błędów) —
  „czysta" maska, blisko eksperckiej.
- **Niska swoistość** = nadwykrywanie, dużo śmieci/szumu uznanego za naczynia.

👉 **To jest miara, którą poprawiliśmy** wprowadzając progowanie histerezowe i
usuwanie izolowanych składowych — mniej FP oznacza wyższą swoistość i wizualnie
czystszą maskę.

### Czułość vs swoistość — kompromis

Te dwie miary „ciągną w przeciwne strony". Obniżając próg detekcji, łapiemy więcej
naczyń (↑ czułość), ale i więcej szumu (↓ swoistość). Podnosząc próg — odwrotnie.
Dobry algorytm utrzymuje **obie** wysoko jednocześnie, dlatego potrzebujemy miary
łączącej je w jedną liczbę.

## Miary dla danych niezrównoważonych: średnia arytmetyczna i geometryczna

Ponieważ trafność zawodzi przy niezrównoważeniu, łączymy czułość i swoistość:

> **średnia arytmetyczna = (czułość + swoistość) / 2**
> (zwana też *balanced accuracy* — zrównoważona trafność)

> **średnia geometryczna = √(czułość × swoistość)**

Dla `im0001`: śr. arytm. = (0,840 + 0,825)/2 = **0,832**; śr. geom. = √(0,840 × 0,825) = **0,832**.

**Różnica między nimi (ważna intuicja):** średnia geometryczna **mocniej karze
nierównowagę** między czułością a swoistością. Przykład:

| czułość | swoistość | śr. arytm. | śr. geom. |
|---------|-----------|-----------|-----------|
| 0,90    | 0,90      | 0,90      | 0,90      |
| 0,98    | 0,50      | 0,74      | 0,70      |
| 1,00    | 0,00      | 0,50      | **0,00**  |

Gdy jedna z miar spada do zera (np. „wszystko jest tłem" → czułość 0), średnia
geometryczna spada do **0**, bezlitośnie demaskując bezużyteczny klasyfikator —
podczas gdy trafność dawała mylące 0,89. Dlatego **średnia geometryczna to nasza
główna miara jakości** zrównoważonej.

## Jak czytać nasze wyniki

Średnie na 6 obrazach testowych (pełna tabela: `outputs/01_przetwarzanie_3/metryki.csv`):

| miara          | wartość | interpretacja |
|----------------|---------|---------------|
| trafność       | ~0,87   | mało znacząca przy niezrównoważeniu — patrz wyżej |
| czułość        | ~0,78   | wykrywamy ~78% pikseli naczyń |
| swoistość      | ~0,88   | poprawnie odrzucamy ~88% tła |
| śr. arytm.     | ~0,83   | zrównoważona trafność |
| **śr. geom.**  | ~0,82   | **główna miara** — dobry, zrównoważony baseline bez uczenia |

**Jak interpretować pojedynczy obraz:**
- `im0044` (śr. geom. 0,89) — najlepszy: wyraźne, grube naczynia, czyste tło.
- `im0002` (śr. geom. 0,79) — patologia (jasne wysięki) tworzy zwarty obszar FP →
  obniżona swoistość. Klasyczny „trudny przypadek".
- `im0004` (śr. geom. 0,71) — niski kontrast → dużo przeoczeń (FN) → niska czułość.

Porażki są **celowo** pokazywane w raporcie (wymóg zadania) — pokazują granice metod
przetwarzania obrazu bez uczenia maszynowego, które staną się punktem odniesienia
(baseline) dla klasyfikatora (4.0) i sieci U-Net (5.0).

## Skąd to w kodzie

- liczenie metryk: [`src/metrics.py`](../src/metrics.py) — `evaluate()` zwraca komplet
  miar; korzysta z `sklearn.metrics.confusion_matrix` oraz
  `imblearn.metrics.geometric_mean_score`.
- mapa błędów (kolory TP/FP/FN): [`src/viz.py`](../src/viz.py) — `error_map()`.
- maski normalizujemy z kodowania 0/255 do {0,1} przy wczytywaniu
  ([`src/data.py`](../src/data.py)).
