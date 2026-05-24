# Jak działa nasz algorytm wykrywania naczyń (wersja na 3.0)

Ten dokument opisuje **krok po kroku** cały proces, którego używamy, i — co
najważniejsze — **dlaczego robimy to tak, a nie inaczej**. Cel: dla każdego piksela
zdjęcia dna oka odpowiedzieć „naczynie czy tło", używając **tylko przetwarzania
obrazu** (bez uczenia maszynowego).

Kod: [`src/processing.py`](../src/processing.py). Metryki oceny: [`docs/metryki.md`](metryki.md).

Cały potok w jednym zdaniu:

> **kanał zielony → poprawa kontrastu → filtr Frangiego → maska FOV → dwa progi (histereza) → usunięcie izolowanych kawałków**

---

## Krok 1. Wybór kanału zielonego

**Co robimy:** ze zdjęcia RGB bierzemy tylko **kanał zielony** (G).

**Dlaczego:** zdjęcie ma 3 kanały (czerwony, zielony, niebieski). Naczynia są na nich
widoczne różnie dobrze:
- **czerwony** — prześwietlony, prawie biały, naczynia giną,
- **niebieski** — ciemny i zaszumiony, mało informacji,
- **zielony** — naczynia (ciemne) najbardziej odcinają się od jasnego tła.

To dobrze widać na obrazku w sekcji 1 notebooka.

**Dlaczego nie inaczej:** moglibyśmy zamienić obraz na skalę szarości (średnia z 3
kanałów), ale uśrednianie „rozcieńcza" dobry kanał zielony słabszymi. Sam zielony
daje lepszy kontrast za darmo.

---

## Krok 2. Poprawa kontrastu (CLAHE)

**Co robimy:** lokalnie podbijamy kontrast kanału zielonego
(`skimage.exposure.equalize_adapthist`).

**Dlaczego:** zdjęcia dna oka są nierównomiernie oświetlone — środek bywa jasny,
brzegi ciemne. Najcieńsze naczynia w ciemniejszych miejscach prawie znikają. Poprawa
kontrastu „wyciąga" je z tła, zanim zacznie działać filtr.

**Dlaczego lokalnie, a nie globalnie:** zwykłe (globalne) wyrównanie histogramu
ustawia jeden kontrast dla całego obrazu — i tak nie poradzi sobie z tym, że różne
fragmenty mają różną jasność. CLAHE dzieli obraz na małe kawałki i poprawia kontrast
w każdym osobno, więc działa równie dobrze w jasnym środku i ciemnym brzegu.

---

## Krok 3. Filtr Frangiego

**Co robimy:** uruchamiamy filtr Frangiego (`skimage.filters.frangi`). Zwraca on dla
każdego piksela liczbę: „jak bardzo wygląda on na fragment naczynia" (0 = wcale,
1 = bardzo). Wynik normalizujemy do zakresu 0–1.

**Dlaczego Frangi:** naczynia to **cienkie, wydłużone, podłużne rurki**. Filtr
Frangiego jest zaprojektowany dokładnie do wykrywania takich kształtów — wzmacnia
linie/rurki, a wygasza płaskie obszary i pojedyncze plamy. To standardowe narzędzie
do naczyń (i wskazane wprost w treści zadania).

**Dwa ważne ustawienia:**
- `black_ridges=True` — szukamy **ciemnych** rurek (na kanale zielonym naczynia są
  ciemniejsze od tła).
- `sigmas = 1..7` — filtr sprawdza naczynia o **różnej grubości**: cienkie i grube.
  Każda wartość „sigma" to inna spodziewana szerokość naczynia (w pikselach).

**Dlaczego nie zwykłe wykrywanie krawędzi (Sobel/Canny):** detektory krawędzi
reagują na **każdą** granicę jasności — także na obwódkę oka, brzeg tarczy nerwu czy
krawędzie zmian chorobowych. Frangi reaguje na **kształt rurki**, więc jest dużo
bardziej selektywny dla naczyń.

---

## Krok 4. Maska obszaru oka (FOV)

**Co robimy:** wyznaczamy maskę **pola widzenia (FOV)** — okrągły obszar dna oka na
czarnym tle — i liczymy/zostawiamy naczynia **tylko w jego wnętrzu**. Dodatkowo
„chowamy się" kilka pikseli od krawędzi koła (erozja).

**Dlaczego:** STARE nie dołącza gotowej maski FOV, więc robimy ją sami: tło jest
prawie czarne, więc progujemy jasność i wypełniamy dziury. Odsunięcie od krawędzi
jest kluczowe — **ostra granica czarne-tło/jasne-oko wygląda dla filtra Frangiego jak
gruba rurka** (naczynie). Bez tego dostalibyśmy jasną „ramkę" wzdłuż całego brzegu.

Liczenie metryk również ograniczamy do FOV — poza nim nie ma danych eksperta, a
ogromne czarne tło sztucznie zawyżałoby wyniki.

---

## Krok 5. Dwa progi (histereza) — zamiana wyniku filtra na maskę 0/1

To jest etap, który **najmocniej ogranicza fałszywe alarmy (FP)**.

**Problem z jednym progiem:** wynik Frangiego to liczby 0–1. Trzeba wybrać próg,
powyżej którego mówimy „naczynie". Ale:
- **niski próg** → łapiemy cienkie naczynia, ale też mnóstwo szumu (dużo FP),
- **wysoki próg** → czysto, ale gubimy cienkie naczynia (dużo przeoczeń, FN).

Jeden próg zawsze jest złym kompromisem.

**Nasze rozwiązanie — dwa progi (`apply_hysteresis_threshold`, `low=0.004`,
`high=0.02`):**
1. piksele powyżej progu **wysokiego** to **pewne naczynia** („ziarna"),
2. piksele powyżej progu **niskiego** zaliczamy do naczyń **tylko jeśli łączą się**
   (sąsiadują, tworzą ciągłą ścieżkę) z jakimś ziarnem.

**Dlaczego to działa:** prawdziwe cienkie naczynia **wyrastają z** grubszych (są z
nimi połączone) — więc zostają. Natomiast słabe, **oderwane** plamki szumu nie dotykają
żadnego pewnego naczynia — więc wypadają. Dostajemy czułość niskiego progu prawie bez
jego śmieci.

---

## Krok 6. Usunięcie izolowanych kawałków

**Co robimy:** z gotowej maski usuwamy **samotne grupki pikseli** mniejsze niż
~200 px (`remove_small_objects`).

**Dlaczego:** naczynia tworzą jedno wielkie, **połączone „drzewo"**. Cokolwiek jest
małą, osobną wysepką, jest niemal na pewno fałszywym alarmem (szum, drobna plamka).
Usuwamy więc to, co nie trzyma się głównej struktury. To dosprząta resztę FP, które
przeszły przez histerezę.

---

## Co zostaje (uczciwe ograniczenie do pokazania prowadzącemu)

Po tych krokach FP jest **wyraźnie mniej** niż przy jednym progu (patrz tabela
w notebooku i `docs/metryki.md`). Ale jeden rodzaj błędu zostaje: **zmiany chorobowe**
(jasne wysięki, np. na obrazie `im0002`). Ich tekstura wygląda dla Frangiego jak
gąszcz naczyń, tworząc **duży, spójny obszar o silnym sygnale**. Dlatego:
- histereza go **nie** odrzuci (ma mocne „ziarna"),
- usuwanie małych kawałków go **nie** ruszy (to duża, połączona plama).

Tego **nie da się** rzetelnie naprawić samym przetwarzaniem obrazu — algorytm nie „wie",
czym jest patologia. To właśnie powód, dla którego robimy poziomy 4.0 (klasyfikator)
i 5.0 (sieć U-Net): model uczy się na przykładach odróżniać naczynia od zmian
chorobowych. Nasz wynik z 3.0 jest **punktem odniesienia (baseline)** dla tych metod.

---

## Skąd wzięły się liczby (progi, rozmiary)

Wartości `low=0.004`, `high=0.02`, próg rozmiaru `200 px` i zakres `sigmas=1..7`
**dobraliśmy eksperymentalnie**: testowaliśmy różne ustawienia na kilku obrazach i
wybieraliśmy te, które dają najwyższą **średnią geometryczną czułości i swoistości**
(czyli najlepiej zrównoważony wynik, bo klasy są mocno niezrównoważone — patrz
[`docs/metryki.md`](metryki.md)). Nie są to magiczne stałe — to wynik strojenia pod
naszą miarę jakości.
