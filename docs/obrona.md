# Ściąga do obrony projektu (IwM — segmentacja naczyń dna oka)

Cel projektu w jednym zdaniu: **dla każdego piksela zdjęcia dna oka decydujemy „naczynie czy tło"** (klasyfikacja binarna), trzema metodami o rosnącej złożoności — przetwarzanie obrazu (3.0), klasyczne ML (4.0), sieć U-Net (5.0). Klasa pozytywna = naczynie.

Ten dokument: co dzieje się w każdym notebooku po kolei, jaki jest wynik, **i o co może zapytać prowadzący + jak odpowiedzieć**. Pełne uzasadnienia metody w [`proces.md`](proces.md), metryki w [`metryki.md`](metryki.md), gotowy raport w [`raport.md`](raport.md).

Dane: **HRF** (45 zdjęć 3504×2336, 3 kategorie po 15: `_h` zdrowe, `_dr` cukrzyca, `_g` jaskra). Skalujemy ×5 w dół (`scale=0.2` → 467×701). 6 obrazów testowych (`01_h, 02_h, 01_dr, 02_dr, 01_g, 02_g`) jest **wspólnych dla wszystkich trzech metod** i wykluczonych z uczenia → uczciwe porównanie. Poziom 3.0 dodatkowo na STARE.

---

## Notebook 01 — poziom 3.0: przetwarzanie obrazu (bez uczenia)

Pliki: `01_przetwarzanie_3.ipynb` (STARE), `01_przetwarzanie_3_hrf.ipynb` (HRF). Kod: [`src/processing.py`](../src/processing.py).

### Co się dzieje po kolei
1. **Wczytanie + kanał zielony.** Z RGB bierzemy tylko kanał G — tam naczynia mają najlepszy kontrast.
2. **Poprawa kontrastu (CLAHE).** Lokalne wyrównanie histogramu — wyciąga cienkie naczynia z nierówno oświetlonego tła.
3. **Filtr Frangiego.** Dla każdego piksela liczy „jak bardzo wygląda na rurkę/naczynie" (0–1). `sigmas=1..7` = różne grubości naczyń, `black_ridges=True` = szukamy ciemnych rurek.
4. **Maska FOV.** Liczymy/wczytujemy okrąg dna oka i pracujemy tylko w nim; krawędź lekko erodujemy (ostra granica koła udawałaby naczynie).
5. **Dwa progi (histereza).** Piksel mocny = pewne naczynie; piksel słaby liczy się **tylko gdy dotyka mocnego**. Odcina samotny szum, zachowując cienkie naczynia wyrastające z grubych.
6. **Usunięcie izolowanych kawałków** (`remove_small_objects`). Naczynia to jedno spójne drzewo — samotne wysepki to FP.
7. **Metryki** na 6 obrazach + panele wizualizacji (obraz | maska eksperta | nasza maska | mapa błędów).

### Wynik
HRF: średnia geometryczna **~0,90** (czułość 0,87 / swoistość 0,93). STARE trudniejsze: **~0,82**. Główne naczynia łapie dobrze; **zostają fałszywe alarmy na zmianach chorobowych** (np. wysięki przy cukrzycy) — to motywacja dla 4.0 i 5.0. To jest **baseline**.

### Możliwe pytania prowadzącego
- **Czemu kanał zielony, a nie szarość/RGB?** Czerwony prześwietlony, niebieski ciemny i zaszumiony; uśrednianie do szarości rozcieńcza dobry kanał G słabszymi. G ma najlepszy kontrast naczyń za darmo.
- **Czemu CLAHE, a nie globalne wyrównanie histogramu?** Oświetlenie jest nierówne (jasny środek, ciemne brzegi). Globalne ustawia jeden kontrast dla całości; CLAHE poprawia lokalnie w każdym kawałku.
- **Czemu Frangi, a nie Sobel/Canny?** Detektory krawędzi reagują na *każdą* granicę jasności (brzeg oka, tarcza nerwu, patologie). Frangi reaguje na **kształt rurki** — jest selektywny dla naczyń.
- **Co to FOV i czemu erozja?** Pole widzenia = okrągły obszar oka. Ostra granica czarne-tło/jasne-oko wygląda dla Frangiego jak gruba rurka → „ramka". Erozja odsuwa się od krawędzi. Metryki też liczymy tylko w FOV (poza nim nie ma danych eksperta, a czarne tło zawyżałoby wynik).
- **Czemu dwa progi, a nie jeden?** Jeden próg to zły kompromis: niski łapie szum (FP↑), wysoki gubi cienkie naczynia (FN↑). Histereza daje czułość niskiego progu prawie bez jego śmieci.
- **Skąd wartości progów (0,005 / 0,02), min_size, sigmas?** Dobrane eksperymentalnie pod najwyższą średnią geometryczną — nie są to magiczne stałe.
- **Pułapka: kodowanie masek.** Maski eksperckie to 0/**255**, nie 0/1 — normalizujemy do {0,1} przy wczytywaniu (`mask > 127`), inaczej metryki byłyby błędne.

---

## Notebook 02 — poziom 4.0: klasyczne uczenie maszynowe

Plik: `02_klasyczne_ml_4_hrf.ipynb`. Cechy: [`src/features.py`](../src/features.py).

### Co się dzieje po kolei
1. **Cechy piksela.** Dla każdego piksela liczymy **17 cech opisujących jego otoczenie** (pełna lista niżej w sekcji „Cechy"). Etykieta = wartość maski eksperta w tym pikselu.
2. **Zbiór uczący zrównoważony (undersampling).** Z 7 obrazów uczących losujemy po **8000 naczyń + 8000 tła na obraz** → ~112 tys. przykładów 50:50. Bez tego klasyfikator „poszedłby na łatwiznę" i przewidywał zawsze tło.
3. **Uczenie dwóch klasyfikatorów:** KNN (k=11, ze `StandardScaler`) i Random Forest (120 drzew).
4. **Ocena hold-out.** Predykcja na **innych** 6 obrazach (tych co w 3.0), porównanie 3.0 vs KNN vs RF.
5. **Wizualizacja RF** + tabele metryk per obraz.

### Wynik
**Random Forest wygrywa**: śr. geometryczna **~0,912** (czułość 0,90 / swoistość 0,92) — bije 3.0 głównie wyższą czułością (łapie więcej naczyń, nie dodając tła). KNN ~ poziom 3.0 i wolny w predykcji. **Wybór: Random Forest.**

### Możliwe pytania prowadzącego
- **Co to jest „cecha z otoczenia / patch"?** Decyzja o pikselu zależy od jego sąsiedztwa, nie tylko od niego samego. Zamiast wycinać okno 5×5 w pętli, liczymy te same statystyki jako **mapy filtrów na całym obrazie** i odczytujemy wektor dla każdego piksela — to ten sam pomysł, ale wektorowo (szybko). (Patrz: pytanie o wycinki niżej.)
- **Czemu undersampling tylko na zbiorze uczącym?** Żeby model uczył się obu klas po równo. Zbiór **testowy zostaje na realnym, niezrównoważonym rozkładzie** — inaczej metryki byłyby zawyżone/nierealne.
- **Czemu RF, a nie KNN?** RF łączy wiele cech i sam uczy się ich progów, jest odporny na przeuczenie i szybki w predykcji. KNN musi liczyć odległość do wszystkich przykładów uczących → wolny, i wyszedł słabiej.
- **Czemu RF jest lepszy od 3.0?** 3.0 to jeden ręczny próg na jednej cesze (Frangi). RF używa **17 cech naraz** i uczy się ich kombinacji z danych.
- **Czy to uczciwy test?** Tak — obrazy testowe to inne zdjęcia niż uczące (hold-out), więc to oszacowanie działania na nowych danych.
- **Pułapka: czemu nie liczycie accuracy jako głównej miary?** Przy ~10% naczyń „wszystko tło" daje ~90% accuracy przy zerowej czułości. Dlatego główna miara to średnia geometryczna czułości i swoistości.

---

## Notebook 03 — poziom 5.0: sieć U-Net

Plik: `03_siec_5_hrf.ipynb`. Model: [`src/unet.py`](../src/unet.py).

### Co się dzieje po kolei
1. **Dane.** Sieć dostaje **cały obraz RGB** i ma odtworzyć maskę naczyń (nie wycinki). Obrazy dopełniamy do 480×704.
2. **Uczenie U-Net.** 39 obrazów uczących, augmentacja (odbicia), strata **BCE + Dice liczona tylko w FOV**, Adam (lr=1e-3), 70 epok (~5 min na Apple MPS).
3. **Dobór progu decyzyjnego.** Sieć zwraca prawdopodobieństwo 0–1. Sprawdzamy progi 0,10–0,50 i wybieramy ten z najwyższą średnią geometryczną → **0,10**.
4. **Porównanie 3.0 vs RF (4.0) vs U-Net (5.0)** na tych samych 6 obrazach hold-out.
5. **Wizualizacja U-Net** + metryki per obraz.

### Wynik
**U-Net najlepszy ze wszystkich**: śr. geometryczna **~0,925**, trafność **0,950**, swoistość **0,956** (czułość 0,896). Wygrywa na **każdym z 6 obrazów**, zwłaszcza na trudnych cukrzycowych — najlepiej oddziela naczynia od zmian chorobowych. Koszt: dłuższe uczenie, potrzeba GPU/MPS i wielu obrazów.

### Możliwe pytania prowadzącego
- **Co to U-Net i czemu działa?** Enkoder–dekoder ze skip-połączeniami. Enkoder zmniejsza rozdzielczość i wychwytuje *kontekst* (co to za struktura), dekoder odtwarza rozdzielczość, a skip-połączenia wnoszą *szczegóły* (dokładne położenie cienkich naczyń). Daje maskę piksel-w-piksel.
- **Czym to się różni od 4.0?** W 4.0 cechy projektujemy ręcznie (17 map). U-Net **uczy się cech sam** z całych obrazów, wraz z kontekstem przestrzennym.
- **Czemu próg 0,10, a nie 0,5?** Przy 0,5 sieć ma wysoką swoistość, ale za niską czułość (gubi cienkie naczynia). Obniżenie progu łapie ich więcej; 0,10 maksymalizuje średnią geometryczną.
- **Czemu strata BCE + Dice, i czemu tylko w FOV?** BCE uczy klasyfikacji per piksel, Dice dba o pokrycie cienkich struktur przy niezrównoważeniu klas. FOV — poza okręgiem nie ma danych eksperta, więc strata go ignoruje.
- **Czemu dopełnianie do 480×704?** U-Net 4× zmniejsza rozdzielczość (4 poolingi) → bok musi być wielokrotnością 16. 467×701 dopełniamy odbiciem do najbliższej wielokrotności.
- **Czemu augmentacja tylko odbicia?** Naczynia są symetryczne na odbicia (nie ma „góry/dołu"), więc to bezpieczne sztuczne powiększenie zbioru. Obroty/skalowanie zmieniłyby skalę naczyń.
- **Czemu na całych obrazach, nie na wycinkach?** U-Net to sieć w pełni konwolucyjna — przetwarza cały obraz naraz i widzi szeroki kontekst; wycinki byłyby potrzebne tylko dla sieci klasyfikującej środkowy piksel.

---

## Cechy do uczenia (4.0) — pełna lista i skąd pochodzą

**To kluczowa sekcja na obronę 4.0.** Wszystkie cechy liczymy z **kanału zielonego po CLAHE** (czyli z tego samego, co w 3.0). Dla każdego piksela budujemy wektor **17 liczb** — wartości odczytane z 17 „map cech". Kod: `feature_maps()` w [`src/features.py`](../src/features.py).

| # | cecha | z czego powstaje | co opisuje (po co) |
|---|-------|------------------|--------------------|
| 1 | `green` | kanał zielony + CLAHE | sama jasność piksela |
| 2–5 | `gauss1,2,4,8` | rozmycie Gaussa o σ=1,2,4,8 | kontekst o rosnącym zasięgu (od drobnego do dużego) |
| 6,8,10 | `mean5,9,15` | lokalna średnia (okno 5/9/15) | typowa jasność otoczenia |
| 7,9,11 | `std5,9,15` | lokalne odchylenie std. = √wariancji (okno 5/9/15) | „zmienność" otoczenia = **2. moment centralny**; naczynie wprowadza kontrast |
| 12 | `min9` | filtr minimum (okno 9) | najciemniejszy punkt otoczenia (naczynia są ciemne) |
| 13 | `max9` | filtr maksimum (okno 9) | najjaśniejszy punkt otoczenia |
| 14 | `sobel` | moduł gradientu Sobela | siła krawędzi (granice naczyń) |
| 15 | `dog13` | różnica Gaussów (σ=1 − σ=3) | wzmacnia cienkie struktury |
| 16 | `dog26` | różnica Gaussów (σ=2 − σ=6) | wzmacnia grubsze struktury |
| 17 | `frangi` | filtr Frangiego | gotowa „naczyniowatość" z poziomu 3.0 |

**Etykieta (y):** wartość maski eksperta dokładnie w tym pikselu (naczynie=1 / tło=0).

### Pytania prowadzącego o cechy
- **Czy to są „momenty centralne" z treści zadania?** Tak — wariancja/odchylenie std. (`std5/9/15`) to **2. moment centralny** w oknie. Treść dopuszcza „np. wariancja, momenty centralne, momenty Hu" — wybraliśmy wariancję w kilku skalach + zestaw filtrów odpowiednich dla cienkich struktur. Cechy są zaprojektowane, nie obowiązkowo wszystkie z listy.
- **Treść mówi o wycinkach 5×5 — gdzie one są?** Liczymy cechy z okien (5, 9, 15 px) jako mapy na całym obrazie zamiast wycinać patche w pętli. **Wynik jest ten sam** (cecha z otoczenia środkowego piksela), ale wektorowo — szybko i policzalne dla każdego piksela obrazu testowego.
- **Czemu różne skale (σ, okna)?** Naczynia mają różną grubość — drobne cechy łapią cienkie, większe okna łapią grube. Zestaw skal pokrywa cały zakres.
- **Czy cechy są skalowane?** Dla KNN tak (`StandardScaler` — wrażliwy na skalę odległości); RF skalowania nie wymaga (drzewa tną po progach).

### A sieć (5.0)?
U-Net **nie używa tych ręcznych cech** — uczy się własnych filtrów z surowego obrazu RGB podczas treningu. To główna różnica koncepcyjna: 4.0 = cechy projektowane ręcznie, 5.0 = cechy uczone automatycznie.

---

## Raport (`docs/raport.md`)

Raport pokrywa wszystkie punkty z wymagań: skład grupy, język/biblioteki, opis metod (3.0/4.0/5.0), wizualizacje (sukces + porażka), analizę i porównanie metod. Trzeba tylko **uzupełnić imiona w sekcji 1**.

### Najważniejsza tabela (średnie na 6 obrazach hold-out)

| metoda | trafność | czułość | swoistość | śr. geom. |
|--------|:---:|:---:|:---:|:---:|
| 3.0 — Frangi        | 0,925 | 0,870 | 0,931 | 0,900 |
| 4.0 — Random Forest | 0,921 | 0,901 | 0,923 | 0,912 |
| **5.0 — U-Net**     | **0,950** | 0,896 | **0,956** | **0,925** |

### Wnioski do obrony (jednym tchem)
- **Jakość rośnie z poziomem:** 0,900 → 0,912 → 0,925. U-Net wygrywa na każdym obrazie.
- **3.0 → 4.0:** RF podnosi głównie **czułość** (więcej cech zamiast jednego progu).
- **4.0 → 5.0:** U-Net poprawia **swoistość i trafność** — najczystsze maski, zwł. na cukrzycy (`01_dr`: 0,893 → 0,920), gdzie 3.0 mylił patologię z naczyniami.
- **Czemu średnia geometryczna, nie accuracy:** klasy niezrównoważone (~10% naczyń) → accuracy myli.

### Pytania przekrojowe (mogą paść przy każdym poziomie)
- **Macierz pomyłek?** TP=trafione naczynie, FP=fałszywy alarm (tło uznane za naczynie), FN=przeoczone naczynie, TN=poprawne tło. Na mapie błędów: zielony/czerwony/niebieski/czarny.
- **Czułość vs swoistość?** Czułość = TP/(TP+FN) = ile naczyń złapaliśmy. Swoistość = TN/(TN+FP) = ile tła poprawnie odrzuciliśmy. Ciągną w przeciwne strony — stąd średnia geometryczna jako jedna liczba.
- **Czemu średnia geometryczna mocniej karze niż arytmetyczna?** Gdy jedna miara → 0, śr. geom. → 0 (demaskuje „wszystko tło"); śr. arytm. dałaby mylące 0,5.
- **Ten sam zbiór na wszystkich poziomach?** Tak — HRF, te same 6 obrazów testowych, zawsze poza zbiorem uczącym. Spełnia wymóg porównywalności.
- **Czemu 3.0 też na STARE?** Pomocniczo, żeby pokazać metodę na drugim zbiorze (STARE ma 2 zestawy eksperckie — użyliśmy `ah`). Spójne porównanie 3 metod robimy na HRF.
- **Czemu scale=0.2?** Przyspiesza obliczenia i ujednolica skalę naczyń, dzięki czemu te same `sigmas` Frangiego działają.
