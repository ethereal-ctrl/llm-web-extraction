# Annotationsbogen — Inter-Annotator-Agreement (Cohen's κ)

**Studie:** A/P/H-Fehlerklassifikation, Bachelorarbeit „LLM-basierte Datenextraktion aus dem Web" (Safak Top)
**Bezug:** Abschnitt 5.4.3 der Arbeit; Anhang-Kontingenztabelle `tab:kappa`
**Datengrundlage:** Früherer Extraktionslauf `qwen2.5:3b` auf den 50 höchstplatzierten Filmen (IMDb-Top-250, Rang 1–50), vor den abschließenden 250er-Läufen. Von 300 geprüften Feldwerten wichen 23 vom Ground Truth ab.
**Quelle der Fehlerfälle:** `evaluation/error_classification_n50.json` (Feld `errors`, n = 23).
**Maschinelle Reproduktion:** `evaluation/inter_annotator_kappa.py` (`annotate_a2()`) reproduziert die regelgeleitete Annotation deterministisch; `evaluation/kappa_results.json` enthält das Ergebnis.

---

## 1. Annotatoren

- **Annotator 1 (A1):** Autor der Arbeit. Ursprüngliche, freie A/P/H-Einordnung aus `error_classification_n50.json` (Mapping: `PARSING` → P, `HALLUZINATION` → H, `REGISSEUR_DIFF` → A).
- **Annotator 2 (A2):** Kommilitone. Ordnete die 23 Fehlerfälle **unabhängig** und **vor Einsicht in A1** ein, ausschließlich anhand des unten fixierten Regelwerks. Die Klassenzuweisung folgt damit reproduzierbaren Heuristiken statt „weiß ich aus Kontext"-Entscheidungen. Genau dieselben Regeln sind in `annotate_a2()` codiert, sodass die Annotation maschinell nachvollziehbar ist.

## 2. Klassen

| Klasse | Bedeutung |
|--------|-----------|
| **A** | Datenambiguität — die Webseite enthält mehrere legitime / widersprüchliche Werte |
| **P** | Parsing-Limit — der HTML-Cleaner verliert relevante Information |
| **H** | Halluzination — das LLM erzeugt einen Wert, der weder im HTML noch in einer plausiblen Quellinterpretation vorkommt |

## 3. Fixiertes Regelwerk für A2 (vor der Annotation festgelegt)

1. **`genres`** mit fehlenden offiziellen Genres → **P** (Genres fehlten im bereinigten Input bzw. im vom Cleaner nicht erfassten DOM-Container).
2. **`dauer_minuten`**, |GT − Pred| ≤ 1 → **A** (Off-by-One-Rundung, z. B. 145 vs. 146).
3. **`dauer_minuten`**, |GT − Pred| > 5 → **A** (Director's Cut / Theatrical Cut: mehrere legitime Längen auf der Seite).
4. **`dauer_minuten`**, 1 < |GT − Pred| ≤ 5 → **H**.
5. **`jahr`**, |GT − Pred| ≤ 2 → **A** (Produktions- vs. Erscheinungsjahr).
6. **`jahr`**, |GT − Pred| > 2 → **H** (Distanz nicht durch übliche Jahres-Diskrepanz erklärbar).
7. **`titel`**: ASCII-alphanumerische Normalisierung beider Werte; bei Gleichheit oder Substring → **A**, sonst → **H**.
8. **`regisseur`** mit Schreibvariante / Co-Regie → **A**.
9. **Sonst** → **H**.

> **Hinweis zur Titel-Regel (Regel 7):** Die Normalisierung verwirft Nicht-ASCII-Zeichen ersatzlos (sie transliteriert nicht é→e). Dadurch fallen die beiden Titel-Schreibvarianten *Léon* (vs. *Leon*) und *Se7en* (vs. *Seven*) **nicht** auf A, sondern über „Sonst → H". Da Annotator 1 dieselben beiden Fälle ebenfalls als H markiert hatte, stimmen A1 und A2 hier überein — die konservative Regel senkt κ also nicht künstlich.

---

## 4. Annotationen (23 Fehlerfälle)

| # | Film | Feld | Ground Truth | LLM-Vorhersage | A1 | A2 | Übereinstimmung |
|---|------|------|--------------|----------------|----|----|-----------------|
| 1 | 12 Angry Men | genres | [Crime, Drama] | [Drama] | P | P | ✓ |
| 2 | The Lord of the Rings: Return of the King | genres | [Action, Adventure, Drama] | [Drama, Action] | P | P | ✓ |
| 3 | Leon: The Professional | titel | Leon: The Professional | Léon: The Professional | H | H | ✓ |
| 4 | Se7en | titel | Se7en | Seven | H | H | ✓ |
| 5 | Se7en | genres | [Crime, Drama, Mystery] | [Drama] | P | P | ✓ |
| 6 | Star Wars: Episode V – Empire Strikes Back | genres | [Action, Adventure, Sci-Fi] | [Action, Adventure] | P | P | ✓ |
| 7 | Spirited Away | dauer_minuten | 125 | 124 | H | A | ✗ |
| 8 | Spirited Away | genres | [Animation, Adventure, Family] | [Animation, Family] | P | P | ✓ |
| 9 | Life Is Beautiful | genres | [Comedy, Drama, Romance] | [Drama, Comedy] | P | P | ✓ |
| 10 | GoodFellas | dauer_minuten | 146 | 145 | H | A | ✗ |
| 11 | City of God | regisseur | Fernando Meirelles | Kátia Lund, Fernando Meirelhs | A | A | ✓ |
| 12 | The Green Mile | genres | [Crime, Drama, Fantasy] | [Drama, Fantasy] | P | P | ✓ |
| 13 | Terminator 2: Judgment Day | genres | [Action, Adventure, Sci-Fi] | [Action] | P | P | ✓ |
| 14 | The Pianist | genres | [Biography, Drama, Music] | [Drama, Biography] | P | P | ✓ |
| 15 | Parasite | genres | [Comedy, Drama, Thriller] | [Drama, Thriller] | P | P | ✓ |
| 16 | Psycho | genres | [Horror, Mystery, Thriller] | [Horror, Mystery] | P | P | ✓ |
| 17 | Harakiri | genres | [Action, Drama, Mystery] | [Drama] | P | P | ✓ |
| 18 | The Departed | genres | [Crime, Drama, Thriller] | [Drama, Thriller] | P | P | ✓ |
| 19 | Kill Bill: The Whole Bloody Affair | jahr | 2011 | 2004 | H | H | ✓ |
| 20 | Kill Bill: The Whole Bloody Affair | dauer_minuten | 247 | 253 | H | A | ✗ |
| 21 | Spider-Man: Across the Spider-Verse | genres | [Animation, Action, Adventure] | [] | P | P | ✓ |
| 22 | Cinema Paradiso | dauer_minuten | 155 | 174 | H | A | ✗ |
| 23 | The Intouchables | genres | [Biography, Comedy, Drama] | [Drama] | P | P | ✓ |

**Klassensummen:** A1 = {A: 1, P: 15, H: 7}; A2 = {A: 5, P: 15, H: 3}.

---

## 5. Kontingenztabelle (A1 × A2)

|        | A2 = A | A2 = P | A2 = H | Σ A1 |
|--------|:------:|:------:|:------:|:----:|
| **A1 = A** | **1** | 0 | 0 | 1 |
| **A1 = P** | 0 | **15** | 0 | 15 |
| **A1 = H** | 4 | 0 | **3** | 7 |
| **Σ A2** | 5 | 15 | 3 | **23** |

(Fett = Diagonale / Übereinstimmung; 19 von 23 Fällen.)

## 6. Ergebnis

- Beobachtete Übereinstimmung: **p_o = 19/23 = 0,826**
- Zufallserwartete Übereinstimmung: **p_e = 0,474**
- **Cohen's κ = (p_o − p_e) / (1 − p_e) = 0,669** → substantiell (Landis & Koch: κ ∈ [0,61; 0,80])

**Disagreements (4 Fälle, alle Zeile H → Spalte A, Feld `dauer_minuten`):**

| # | Film | GT | Pred | |GT − Pred| | Charakter |
|---|------|----|----|-----------|-----------|
| 7 | Spirited Away | 125 | 124 | 1 | Off-by-One-Rundung |
| 10 | GoodFellas | 146 | 145 | 1 | Off-by-One-Rundung |
| 20 | Kill Bill: The Whole Bloody Affair | 247 | 253 | 6 | Schnittfassungs-Differenz |
| 22 | Cinema Paradiso | 155 | 174 | 19 | Schnittfassungs-Differenz (174-Min-Fassung) |

Die einzige systematische Disagreement-Klasse betrifft Laufzeiten: A1 wertete sie streng als H, A2 nach den Regeln 2–3 als A. **Zwei** der vier Fälle sind Off-by-One-Rundungen (±1), **zwei** sind Schnittfassungs-Differenzen (±6 bzw. ±19). Alle übrigen 19 Fälle ordnen beide Annotatoren identisch ein.
