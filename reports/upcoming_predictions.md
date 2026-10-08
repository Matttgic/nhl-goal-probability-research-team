# Prochains matchs NHL — modèle expérimental

Actualisé le **09/10/2026 01:21 (Paris)**. Fenêtre : prochaines 48 heures.

Probabilités conditionnelles à la participation du joueur. Effectifs actuels ; blessures, composition, PP1 et gardiens non confirmés.
Aucune cote de bookmaker utilisée. La cote théorique ne constitue pas une recommandation de pari.

## Validation chronologique

600 matchs historiques. Apprentissage : 10136 lignes, jusqu’au 2026-03-27 ; calibration séparée : 4316 lignes ; test : 3061 lignes du 2026-04-12 au 2026-10-07.

| Modèle | Brier, plus bas préférable | Log loss, plus basse préférable |
|---|---:|---:|
| Référence simple | 0.12545 | 0.40794 |
| Enrichi | 0.12544 | 0.40793 |

Fréquence de but observée : 15.8% ; probabilité moyenne enrichie : 15.5%.
Ce test ne prouve pas la rentabilité. Les joueurs-matchs d’un même match sont corrélés.

Données de temps de jeu en power play indisponibles : ces variables ne sont pas utilisées.

## Matchs du jour dont l’heure de début est passée

**Buteurs : estimations calculées après le début**, uniquement avec l’historique des dates antérieures. Elles concernent l’ensemble du match, pas les buts à venir à partir de maintenant. Le score actuel et le temps restant ne sont pas utilisés ; ce ne sont pas des pronostics verrouillés avant le début.

| Match | Début prévu (Paris) | Statut NHL | Score constaté |
|---|---|---|---|
| UTA chez BOS | 09/10 01:00 | En cours | 0–0 |
| DAL chez BUF | 09/10 01:00 | En cours | 0–0 |
| NSH chez MTL | 09/10 01:00 | En cours | 0–0 |
| PHI chez OTT | 09/10 01:00 | En cours | 0–0 |
| MIN chez TBL | 09/10 01:00 | En cours | 0–2 |
| VAN chez CAR | 09/10 01:00 | En cours | 0–2 |

### Buteurs — UTA chez BOS

| Joueur | Équipe | But sur l’ensemble du match, estimation tardive |
|---|---|---:|
| Nick Schmaltz | UTA | 32.1% |
| Dylan Guenther | UTA | 27.5% |
| Clayton Keller | UTA | 24.8% |
| David Pastrnak | BOS | 31.4% |
| JJ Peterka | BOS | 24.3% |
| Morgan Geekie | BOS | 22.8% |

### Buteurs — DAL chez BUF

| Joueur | Équipe | But sur l’ensemble du match, estimation tardive |
|---|---|---:|
| Jason Robertson | DAL | 36.5% |
| Wyatt Johnston | DAL | 36.2% |
| Mikko Rantanen | DAL | 31.0% |
| Tage Thompson | BUF | 34.2% |
| Jack Quinn | BUF | 28.3% |
| Zach Benson | BUF | 21.8% |

### Buteurs — NSH chez MTL

| Joueur | Équipe | But sur l’ensemble du match, estimation tardive |
|---|---|---:|
| Steven Stamkos | NSH | 32.5% |
| Filip Forsberg | NSH | 29.8% |
| Ryan O'Reilly | NSH | 28.2% |
| Cole Caufield | MTL | 36.8% |
| Juraj Slafkovský | MTL | 28.5% |
| Nick Suzuki | MTL | 28.0% |

### Buteurs — PHI chez OTT

| Joueur | Équipe | But sur l’ensemble du match, estimation tardive |
|---|---|---:|
| Tyson Foerster | PHI | 29.1% |
| Porter Martone | PHI | 20.5% |
| Owen Tippett | PHI | 19.9% |
| Dylan Cozens | OTT | 28.9% |
| Tim Stützle | OTT | 28.6% |
| William Eklund | OTT | 26.7% |

### Buteurs — MIN chez TBL

| Joueur | Équipe | But sur l’ensemble du match, estimation tardive |
|---|---|---:|
| Matt Boldy | MIN | 46.5% |
| Kirill Kaprizov | MIN | 39.4% |
| Joel Eriksson Ek | MIN | 25.1% |
| Brandon Hagel | TBL | 44.7% |
| Jake Guentzel | TBL | 31.5% |
| Nikita Kucherov | TBL | 30.2% |

### Buteurs — VAN chez CAR

| Joueur | Équipe | But sur l’ensemble du match, estimation tardive |
|---|---|---:|
| Jake DeBrusk | VAN | 28.5% |
| Elias Pettersson | VAN | 21.9% |
| Brock Boeser | VAN | 21.5% |
| Sebastian Aho | CAR | 31.7% |
| Andrei Svechnikov | CAR | 31.1% |
| Logan Stankoven | CAR | 26.3% |

## CHI chez NYI — 09/10 01:30 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Connor Bedard | CHI | 32.7% | 3.06 | 3.00 | 34 matchs |
| Anton Frondell | CHI | 27.9% | 3.58 | 2.80 | 16 matchs |
| Patrick Kane | CHI | 25.1% | 3.98 | 1.90 | 37 matchs |
| Brayden Schenn | NYI | 23.8% | 4.21 | 1.60 | 35 matchs |
| Frank Nazar | CHI | 21.2% | 4.71 | 1.60 | 37 matchs |
| Ryan Greene | CHI | 20.0% | 4.99 | 1.20 | 37 matchs |
| Teuvo Teravainen | CHI | 19.2% | 5.21 | 1.20 | 33 matchs |
| Emil Heineman | NYI | 18.6% | 5.37 | 0.90 | 37 matchs |
| Ryan Donato | CHI | 18.4% | 5.43 | 2.00 | 38 matchs |
| Bo Horvat | NYI | 18.0% | 5.54 | 2.70 | 35 matchs |

4 joueurs sans historique suffisant : aucune probabilité inventée.

## SJS chez STL — 09/10 02:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Macklin Celebrini | SJS | 35.0% | 2.85 | 3.60 | 37 matchs |
| Jimmy Snuggerud | STL | 30.4% | 3.29 | 3.10 | 37 matchs |
| Will Smith | SJS | 26.1% | 3.84 | 1.90 | 38 matchs |
| Robert Thomas | STL | 25.7% | 3.90 | 1.70 | 25 matchs |
| Dylan Holloway | STL | 22.9% | 4.37 | 2.50 | 29 matchs |
| Pavel Buchnevich | STL | 21.8% | 4.59 | 1.30 | 36 matchs |
| Mason McTavish | STL | 21.4% | 4.68 | 2.70 | 30 matchs |
| Igor Chernyshov | SJS | 21.1% | 4.74 | 1.50 | 17 matchs |
| Connor McMichael | STL | 19.7% | 5.07 | 1.90 | 32 matchs |
| Alexander Wennberg | SJS | 19.3% | 5.17 | 1.10 | 36 matchs |

7 joueurs sans historique suffisant : aucune probabilité inventée.

## COL chez CGY — 09/10 03:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Martin Necas | COL | 38.3% | 2.61 | 3.60 | 35 matchs |
| Nathan MacKinnon | COL | 30.9% | 3.24 | 3.80 | 37 matchs |
| Jonathan Huberdeau | CGY | 25.4% | 3.93 | 2.43 | 7 matchs |
| Mikael Backlund | CGY | 25.4% | 3.94 | 2.30 | 37 matchs |
| Morgan Frost | CGY | 24.0% | 4.16 | 2.50 | 37 matchs |
| Joel Farabee | CGY | 23.6% | 4.23 | 2.10 | 37 matchs |
| Nazem Kadri | COL | 22.9% | 4.37 | 2.50 | 32 matchs |
| Matt Coronato | CGY | 22.5% | 4.45 | 2.30 | 36 matchs |
| Matvei Gridin | CGY | 20.6% | 4.84 | 1.80 | 35 matchs |
| Brock Nelson | COL | 18.8% | 5.31 | 1.30 | 38 matchs |

3 joueurs sans historique suffisant : aucune probabilité inventée.

## TOR chez VGK — 09/10 04:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Auston Matthews | TOR | 37.5% | 2.67 | 4.20 | 22 matchs |
| Kirill Marchenko | TOR | 30.3% | 3.30 | 2.80 | 36 matchs |
| Jack Eichel | VGK | 30.0% | 3.34 | 2.80 | 38 matchs |
| William Nylander | TOR | 29.9% | 3.35 | 2.60 | 32 matchs |
| Mark Stone | VGK | 27.6% | 3.62 | 2.50 | 33 matchs |
| Mitch Marner | VGK | 26.5% | 3.77 | 2.40 | 38 matchs |
| John Tavares | TOR | 25.4% | 3.94 | 2.30 | 38 matchs |
| Brett Howden | VGK | 20.8% | 4.80 | 1.40 | 23 matchs |
| Tomas Hertl | VGK | 20.1% | 4.96 | 2.40 | 39 matchs |
| Jack Roslovic | TOR | 19.4% | 5.14 | 2.10 | 37 matchs |

7 joueurs sans historique suffisant : aucune probabilité inventée.

## SEA chez DET — 10/10 01:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Dylan Larkin | DET | 36.3% | 2.75 | 3.60 | 26 matchs |
| Alex DeBrincat | DET | 29.7% | 3.36 | 2.60 | 36 matchs |
| Bobby McMann | SEA | 29.3% | 3.42 | 3.70 | 35 matchs |
| Lucas Raymond | DET | 29.1% | 3.43 | 2.60 | 36 matchs |
| Andrew Copp | DET | 23.3% | 4.30 | 1.70 | 33 matchs |
| Jared McCann | SEA | 22.9% | 4.36 | 2.60 | 33 matchs |
| Viktor Arvidsson | DET | 22.5% | 4.44 | 2.40 | 35 matchs |
| Mackie Samoskevich | SEA | 21.2% | 4.71 | 2.20 | 35 matchs |
| Michael Brandsegg-Nygård | DET | 20.2% | 4.94 | 2.00 | 8 matchs |
| Ryan Winterton | SEA | 17.1% | 5.85 | 2.10 | 27 matchs |

7 joueurs sans historique suffisant : aucune probabilité inventée.

## NYR chez WSH — 10/10 01:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Alex Tuch | WSH | 28.7% | 3.48 | 3.00 | 36 matchs |
| Pavel Dorofeyev | NYR | 28.0% | 3.57 | 2.80 | 40 matchs |
| Mika Zibanejad | NYR | 27.3% | 3.66 | 1.90 | 38 matchs |
| Alex Ovechkin | WSH | 24.5% | 4.08 | 3.10 | 36 matchs |
| Tom Wilson | WSH | 24.4% | 4.10 | 1.70 | 34 matchs |
| Alexis Lafrenière | NYR | 23.9% | 4.18 | 1.80 | 38 matchs |
| Ryan Leonard | WSH | 21.9% | 4.57 | 2.70 | 36 matchs |
| Aliaksei Protas | WSH | 21.5% | 4.64 | 1.60 | 33 matchs |
| J.T. Miller | NYR | 20.5% | 4.89 | 1.90 | 33 matchs |
| Will Cuylle | NYR | 19.5% | 5.13 | 2.40 | 38 matchs |

1 joueurs sans historique suffisant : aucune probabilité inventée.

## PIT chez CBJ — 10/10 01:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Rickard Rakell | PIT | 33.0% | 3.03 | 3.20 | 37 matchs |
| Sidney Crosby | PIT | 28.5% | 3.51 | 2.30 | 25 matchs |
| Adam Fantilli | CBJ | 27.5% | 3.63 | 2.30 | 36 matchs |
| Evgeni Malkin | PIT | 26.4% | 3.79 | 2.70 | 28 matchs |
| Zach Werenski | CBJ | 25.0% | 4.00 | 3.30 | 33 matchs |
| Valeri Nichushkin | CBJ | 24.3% | 4.11 | 1.70 | 36 matchs |
| Bryan Rust | PIT | 23.0% | 4.34 | 2.00 | 30 matchs |
| Charlie Coyle | CBJ | 21.6% | 4.62 | 1.60 | 36 matchs |
| Egor Chinakhov | PIT | 20.5% | 4.87 | 2.00 | 38 matchs |
| Matthew Knies | CBJ | 20.3% | 4.92 | 1.40 | 36 matchs |

1 joueurs sans historique suffisant : aucune probabilité inventée.

## ANA chez WPG — 10/10 02:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Kyle Connor | WPG | 54.5% | 1.84 | 5.10 | 39 matchs |
| Leo Carlsson | ANA | 37.4% | 2.67 | 3.00 | 29 matchs |
| Mark Scheifele | WPG | 35.5% | 2.81 | 1.80 | 39 matchs |
| Cutter Gauthier | ANA | 34.3% | 2.92 | 4.40 | 32 matchs |
| Mikael Granlund | ANA | 31.0% | 3.23 | 1.90 | 31 matchs |
| Beckett Sennecke | ANA | 29.3% | 3.41 | 2.90 | 37 matchs |
| Cole Perfetti | WPG | 25.5% | 3.93 | 1.80 | 39 matchs |
| Gabriel Vilardi | WPG | 25.0% | 4.01 | 1.60 | 39 matchs |
| Alex Iafallo | WPG | 19.6% | 5.10 | 2.00 | 36 matchs |
| A.J. Greer | ANA | 18.4% | 5.43 | 1.90 | 34 matchs |

9 joueurs sans historique suffisant : aucune probabilité inventée.

## PHI chez BOS — 10/10 19:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| David Pastrnak | BOS | 36.4% | 2.75 | 2.90 | 37 matchs |
| Tyson Foerster | PHI | 29.1% | 3.44 | 2.40 | 12 matchs |
| Morgan Geekie | BOS | 29.1% | 3.44 | 1.80 | 37 matchs |
| Trevor Zegras | PHI | 24.4% | 4.11 | 1.10 | 38 matchs |
| JJ Peterka | BOS | 24.0% | 4.16 | 2.20 | 37 matchs |
| Christian Dvorak | PHI | 23.7% | 4.22 | 1.60 | 38 matchs |
| Porter Martone | PHI | 20.5% | 4.87 | 1.90 | 13 matchs |
| Pavel Zacha | BOS | 19.3% | 5.19 | 1.50 | 34 matchs |
| James Hagens | BOS | 18.3% | 5.47 | 1.50 | 6 matchs |
| Elias Lindholm | BOS | 18.0% | 5.57 | 1.40 | 34 matchs |

3 joueurs sans historique suffisant : aucune probabilité inventée.

## VAN chez NJD — 10/10 21:30 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Jack Hughes | NJD | 50.7% | 1.97 | 4.20 | 34 matchs |
| Timo Meier | NJD | 35.3% | 2.83 | 3.90 | 37 matchs |
| Jesper Bratt | NJD | 29.0% | 3.45 | 2.60 | 37 matchs |
| Jake DeBrusk | VAN | 28.5% | 3.50 | 3.40 | 36 matchs |
| Nico Hischier | NJD | 27.2% | 3.68 | 2.20 | 37 matchs |
| Connor Brown | NJD | 26.9% | 3.72 | 2.00 | 34 matchs |
| Arseny Gritsyuk | NJD | 25.3% | 3.95 | 2.70 | 25 matchs |
| Luke Evangelista | NJD | 24.1% | 4.15 | 2.70 | 37 matchs |
| Dawson Mercer | NJD | 22.7% | 4.41 | 2.30 | 37 matchs |
| Elias Pettersson | VAN | 21.9% | 4.56 | 2.00 | 38 matchs |

4 joueurs sans historique suffisant : aucune probabilité inventée.

## EDM chez SJS — 10/10 22:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Connor McDavid | EDM | 27.1% | 3.70 | 3.40 | 37 matchs |
| Zach Hyman | EDM | 26.9% | 3.71 | 1.50 | 29 matchs |
| Igor Chernyshov | SJS | 23.3% | 4.29 | 1.50 | 17 matchs |
| Matt Savoie | EDM | 23.2% | 4.30 | 3.00 | 33 matchs |
| Vasily Podkolzin | EDM | 22.9% | 4.37 | 3.00 | 37 matchs |
| Leon Draisaitl | EDM | 21.5% | 4.66 | 3.00 | 21 matchs |
| Macklin Celebrini | SJS | 21.1% | 4.74 | 3.60 | 37 matchs |
| Ryan Nugent-Hopkins | EDM | 21.0% | 4.76 | 2.30 | 32 matchs |
| Alexander Wennberg | SJS | 20.7% | 4.82 | 1.10 | 36 matchs |
| Mason Marchment | SJS | 19.1% | 5.24 | 1.70 | 35 matchs |

8 joueurs sans historique suffisant : aucune probabilité inventée.

## MIN chez FLA — 11/10 00:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Matt Boldy | MIN | 49.1% | 2.04 | 4.20 | 31 matchs |
| Kirill Kaprizov | MIN | 42.0% | 2.38 | 3.30 | 32 matchs |
| Sam Bennett | FLA | 39.0% | 2.56 | 2.80 | 33 matchs |
| Sam Reinhart | FLA | 34.4% | 2.91 | 2.60 | 21 matchs |
| Carter Verhaeghe | FLA | 33.8% | 2.96 | 2.50 | 35 matchs |
| Anton Lundell | FLA | 24.9% | 4.02 | 1.80 | 21 matchs |
| Blake Coleman | MIN | 21.0% | 4.76 | 2.20 | 28 matchs |
| Matthew Tkachuk | FLA | 20.6% | 4.86 | 2.00 | 35 matchs |
| Seth Jones | FLA | 20.4% | 4.91 | 1.50 | 16 matchs |
| Michael McCarron | MIN | 20.0% | 5.00 | 2.00 | 35 matchs |

5 joueurs sans historique suffisant : aucune probabilité inventée.

## UTA chez BUF — 11/10 01:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Tage Thompson | BUF | 34.2% | 2.92 | 3.80 | 37 matchs |
| Jack Quinn | BUF | 32.5% | 3.08 | 3.30 | 38 matchs |
| Nick Schmaltz | UTA | 32.1% | 3.12 | 3.80 | 37 matchs |
| Dylan Guenther | UTA | 27.5% | 3.63 | 3.20 | 34 matchs |
| Zach Benson | BUF | 25.9% | 3.86 | 2.20 | 34 matchs |
| Clayton Keller | UTA | 24.8% | 4.04 | 2.10 | 37 matchs |
| Lawson Crouse | UTA | 20.1% | 4.99 | 2.10 | 36 matchs |
| Logan Cooley | UTA | 19.5% | 5.13 | 1.20 | 29 matchs |
| Vincent Trocheck | UTA | 18.9% | 5.30 | 1.30 | 36 matchs |
| Noah Ostlund | BUF | 18.0% | 5.54 | 1.20 | 28 matchs |

3 joueurs sans historique suffisant : aucune probabilité inventée.

## DET chez MTL — 11/10 01:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Cole Caufield | MTL | 36.8% | 2.72 | 3.40 | 35 matchs |
| Dylan Larkin | DET | 34.6% | 2.89 | 3.60 | 26 matchs |
| Juraj Slafkovský | MTL | 28.5% | 3.51 | 2.40 | 36 matchs |
| Lucas Raymond | DET | 28.2% | 3.54 | 2.60 | 36 matchs |
| Alex DeBrincat | DET | 28.1% | 3.55 | 2.60 | 36 matchs |
| Nick Suzuki | MTL | 28.0% | 3.57 | 2.10 | 36 matchs |
| Andrew Copp | DET | 22.5% | 4.44 | 1.70 | 33 matchs |
| Viktor Arvidsson | DET | 21.9% | 4.57 | 2.40 | 35 matchs |
| Ivan Demidov | MTL | 20.4% | 4.90 | 2.20 | 35 matchs |
| Michael Brandsegg-Nygård | DET | 18.7% | 5.36 | 2.00 | 8 matchs |

6 joueurs sans historique suffisant : aucune probabilité inventée.

## NSH chez OTT — 11/10 01:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Steven Stamkos | NSH | 32.5% | 3.07 | 3.20 | 37 matchs |
| Filip Forsberg | NSH | 29.8% | 3.36 | 2.20 | 37 matchs |
| Dylan Cozens | OTT | 28.9% | 3.46 | 2.90 | 38 matchs |
| Tim Stützle | OTT | 28.6% | 3.50 | 2.30 | 36 matchs |
| Ryan O'Reilly | NSH | 28.2% | 3.54 | 1.70 | 36 matchs |
| Shane Pinto | OTT | 26.6% | 3.76 | 2.10 | 38 matchs |
| William Eklund | OTT | 24.7% | 4.04 | 2.10 | 38 matchs |
| Mavrik Bourque | NSH | 22.5% | 4.44 | 2.30 | 36 matchs |
| Claude Giroux | OTT | 21.7% | 4.61 | 2.10 | 38 matchs |
| Ridly Greig | OTT | 21.1% | 4.75 | 1.90 | 36 matchs |

3 joueurs sans historique suffisant : aucune probabilité inventée.

## DAL chez PIT — 11/10 01:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Rickard Rakell | PIT | 39.2% | 2.55 | 3.20 | 37 matchs |
| Wyatt Johnston | DAL | 36.2% | 2.76 | 2.60 | 36 matchs |
| Sidney Crosby | PIT | 29.0% | 3.45 | 2.30 | 25 matchs |
| Evgeni Malkin | PIT | 26.9% | 3.72 | 2.70 | 28 matchs |
| Bryan Rust | PIT | 25.4% | 3.94 | 2.00 | 30 matchs |
| Tommy Novak | PIT | 24.6% | 4.06 | 2.20 | 39 matchs |
| Matt Duchene | DAL | 24.6% | 4.06 | 1.70 | 33 matchs |
| Jason Robertson | DAL | 22.9% | 4.36 | 3.10 | 36 matchs |
| Mikko Rantanen | DAL | 22.3% | 4.49 | 1.50 | 19 matchs |
| Egor Chinakhov | PIT | 21.4% | 4.67 | 2.00 | 38 matchs |

2 joueurs sans historique suffisant : aucune probabilité inventée.

## CAR chez CHI — 11/10 01:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Connor Bedard | CHI | 34.2% | 2.93 | 3.00 | 34 matchs |
| Andrei Svechnikov | CAR | 31.9% | 3.13 | 3.30 | 34 matchs |
| Sebastian Aho | CAR | 31.5% | 3.18 | 3.50 | 34 matchs |
| Nikolaj Ehlers | CAR | 30.9% | 3.23 | 2.70 | 37 matchs |
| Jackson Blake | CAR | 29.3% | 3.42 | 2.10 | 36 matchs |
| Patrick Kane | CHI | 26.7% | 3.74 | 1.90 | 37 matchs |
| Frank Nazar | CHI | 25.6% | 3.91 | 1.60 | 37 matchs |
| Logan Stankoven | CAR | 22.4% | 4.46 | 2.20 | 36 matchs |
| Tyler Bertuzzi | CHI | 21.5% | 4.65 | 1.30 | 38 matchs |
| Anton Frondell | CHI | 20.8% | 4.80 | 2.80 | 16 matchs |

2 joueurs sans historique suffisant : aucune probabilité inventée.

## CBJ chez STL — 11/10 01:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Adam Fantilli | CBJ | 32.6% | 3.07 | 2.30 | 36 matchs |
| Dylan Holloway | STL | 27.2% | 3.68 | 2.50 | 29 matchs |
| Matthew Knies | CBJ | 25.8% | 3.88 | 1.40 | 36 matchs |
| Robert Thomas | STL | 25.7% | 3.90 | 1.70 | 25 matchs |
| Jimmy Snuggerud | STL | 25.6% | 3.90 | 3.10 | 37 matchs |
| Zach Werenski | CBJ | 23.1% | 4.33 | 3.30 | 33 matchs |
| Charlie Coyle | CBJ | 23.0% | 4.35 | 1.60 | 36 matchs |
| Mason McTavish | STL | 21.2% | 4.72 | 2.70 | 30 matchs |
| Sean Monahan | CBJ | 20.3% | 4.93 | 1.60 | 36 matchs |
| Pius Suter | STL | 20.1% | 4.97 | 1.10 | 30 matchs |

4 joueurs sans historique suffisant : aucune probabilité inventée.

## TOR chez COL — 11/10 01:00 (Paris)

| Joueur | Équipe | But, proba modèle | Cote théorique | Tirs/match, 10 derniers | Historique |
|---|---|---:|---:|---:|---:|
| Martin Necas | COL | 43.0% | 2.33 | 3.60 | 35 matchs |
| Nathan MacKinnon | COL | 41.7% | 2.40 | 3.80 | 37 matchs |
| Auston Matthews | TOR | 37.5% | 2.67 | 4.20 | 22 matchs |
| Kirill Marchenko | TOR | 30.3% | 3.30 | 2.80 | 36 matchs |
| William Nylander | TOR | 29.9% | 3.35 | 2.60 | 32 matchs |
| John Tavares | TOR | 25.4% | 3.94 | 2.30 | 38 matchs |
| Gabriel Landeskog | COL | 23.8% | 4.20 | 2.00 | 22 matchs |
| Artturi Lehkonen | COL | 21.7% | 4.60 | 1.50 | 27 matchs |
| Nazem Kadri | COL | 19.5% | 5.14 | 2.50 | 32 matchs |
| Brock Nelson | COL | 19.2% | 5.20 | 1.30 | 38 matchs |

4 joueurs sans historique suffisant : aucune probabilité inventée.

Tous les joueurs calculés : [CSV](upcoming_predictions.csv). La colonne `prediction_kind` distingue `pregame` et `retrospective_history_only` ; seules les lignes `pregame` sont des prévisions avant le début. Paramètres et validation : [JSON](upcoming_validation.json).

Sources : endpoints publics NHL, matchs terminés et effectifs au moment du calcul. Le statut expérimental s’applique à toutes les lignes.
