# Maintenance prédictive pour installations photovoltaïques

Système de surveillance temps réel et de maintenance prédictive pour installations PV, basé sur des modèles de détection d'anomalies non supervisés.

**PFA — ENSA Agadir | Albarray Consulting | Juillet-Août 2026**

## Architecture

```
Replayer (dataset test) → Kafka "measurements" → Consumer Python (modèle ML)
                                                        │
                                          ┌──────────────┼──────────────┐
                                          ▼                             ▼
                                    TimescaleDB                   Kafka "alarms"
                                          │
                                          ▼
                                       Grafana
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Streaming | Apache Kafka |
| Base de données | PostgreSQL + TimescaleDB |
| Dashboard | Grafana |
| ML | TensorFlow / Keras (autoencodeurs), Scikit-learn (Isolation Forest) |
| Conteneurisation | Docker Compose |

## Lancement rapide

```bash
# Démarrer l'infrastructure
docker compose up -d

# Accéder à Grafana
# http://localhost:3000 (admin / admin)
```

## Structure du projet

```
pfa-photovoltaique/
├── notebooks/
│   └── final.ipynb           # tout le pipeline ML, en 15 sections
├── model_final/              # système déployé (~2,8 Mo), versionné
│   ├── config.json           # features, hyperparamètres, seuil de fusion
│   ├── ae/ae_0..4.keras      # ensemble de 5 autoencodeurs denses
│   ├── lstm/lstm_0..4.keras  # ensemble de 5 autoencodeurs LSTM
│   ├── scaler_ae.pkl         # MinMaxScaler, ajusté sur le train sain
│   ├── scaler_lstm.pkl
│   ├── ref_score_*.npy       # références de rang figées (validation)
│   └── predictions_test.csv  # sorties du système sur le jeu de test
├── data/                     # caches versionnés ; brut et intermédiaires exclus
│   └── replay_test.csv       # partition de test rejouable (600 Ko)
├── src/                      # chaîne temps réel
│   ├── detecteur.py          # DetecteurPV — portage déployable du § 14
│   ├── replayer.py           # rejoue la partition de test sur Kafka
│   ├── consumer.py           # score le flux et alimente TimescaleDB
│   └── db.py                 # insertions measurements / alarms
├── docker/
│   ├── init.sql              # schéma measurements / alarms / maintenance
│   └── migration_01_va_ia.sql
├── figures/                  # régénéré par le notebook, non versionné
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Démo temps réel

```bash
docker compose up -d
python src/consumer.py     # dans un terminal
python src/replayer.py     # dans un autre
```

Le replayer rejoue les 2 975 points de la partition de test à raison d'un par
seconde ; le consumer reconstitue les séquences, score, écrit dans TimescaleDB et
publie les anomalies sur le topic `alarms`. Aucun prérequis : `data/replay_test.csv`
est versionné, et `model_final/` contient déjà les 10 réseaux.

`src/detecteur.py` est le portage de la classe `DetecteurPV` définie en § 14 du
notebook. Le notebook reste la source de vérité ; le portage est vérifié comme
strictement équivalent (écart max 1e-16 sur les 2 813 points scorables du test),
et le chemin streaming redonne exactement le score du chemin par lot.

**Débit.** Un point coûte ~0,9 s à scorer, les 10 réseaux étant interrogés un par
un. C'est le facteur limitant de la démo, et la raison du délai d'une seconde
entre deux messages — soit 300x plus rapide que la cadence réelle des mesures,
qui est de 5 minutes.

**Base déjà initialisée.** `docker/init.sql` n'est joué qu'à la création du
volume PostgreSQL. Sur une base existante, appliquer la migration :

```bash
docker compose exec -T timescaledb psql -U pfa -d photovoltaique < docker/migration_01_va_ia.sql
```

## Dataset

[Zenodo — record 7358042](https://zenodo.org/records/7358042) (article PMC9800176)

- `dt1` : données météo (GTI, DTI, TA, TPV) — **2,14 Go**
- `dt2` : données électriques + étiquettes de pannes (ombrage) — **193 Mo**

## Rejouer le notebook d'analyse

`notebooks/final.ipynb` couvre tout le pipeline : fusion, nettoyage, EDA, feature engineering,
split chronologique, Isolation Forest, autoencodeur dense, autoencodeur LSTM, fusion des
détecteurs et évaluation finale sur le jeu de test.

```bash
jupyter nbconvert --to notebook --execute notebooks/final.ipynb --output /tmp/check.ipynb
```

**Un clone suffit.** Le notebook s'exécute en une dizaine de minutes sans rien réentraîner :
les 10 réseaux du système final, leurs scalers et leurs scores de référence sont versionnés
dans `model_final/` (~2,8 Mo), et les grilles d'hyperparamètres sont mises en cache dans
`data/tuning_*.csv`. Sans ces caches, la même exécution demanderait plus de 10 heures.

### Ce qui est rejoué depuis le cache

Les deux CSV bruts Zenodo (2,3 Go au total) **ne sont pas versionnés** : GitHub refuse tout
fichier au-delà de 100 Mo. Depuis un clone qui ne les contient pas :

| Section | Comportement sans les CSV bruts |
|---|---|
| § 1 — Exploration des CSV bruts | **sautée**, avec un message explicite |
| § 2 — Fusion dt1/dt2 et rééchantillonnage | **rechargée** depuis `data/merged_5min_day.csv`, non recalculée |
| § 3 à § 15 | exécutées normalement, résultats identiques |

Le notebook détecte lui-même la situation (`RAW_DISPONIBLE`, cellule d'imports) et l'annonce
en clair au démarrage. Pour rejouer les sections 1 et 2 en entier, télécharger les deux CSV
depuis Zenodo et les placer dans `data/` sous leurs noms d'origine :

```
data/dt1_solar_and_meteorological_measurement.csv
data/dt2_electrical_production_inverter_1_with_faults.csv
```

### Tout régénérer

Chaque cache a son interrupteur, en tête de la cellule concernée. Les passer à `True` relance
le calcul correspondant et réécrit le fichier de cache.

| Indicateur | Régénère | Coût |
|---|---|---|
| `RUN_TRAINING` | les 10 réseaux de `model_final/` | ~50 min |
| `FORCER_TUNING_A` / `FORCER_TUNING_B` | grilles LSTM Phase A / Phase B | ~3 h 20 / ~5 h 30 |
| `FORCER_TUNING_AE` | grille hyperparamètres AE | ~30 min |
| `FORCER_ABLATION_AE` / `FORCER_TUNING_IF` | ablation features AE / grille Isolation Forest | quelques min |
| `FORCER_LSTM_VIABILITE` / `FORCER_LSTM_SINGLE` | LSTM exploratoires (§ 10.1 / § 10.4) | ~11 min / ~9 min |
| `FORCER_COURBE` | réseau de démonstration de la courbe d'apprentissage | ~1 min |
| `FORCER_RECALCUL` | fusion dt1/dt2 — **exige les CSV bruts Zenodo** | ~10 min |

> `RUN_TRAINING = True` réentraîne les 10 réseaux et **invalide les références de rang et le
> seuil** : la section 15 les recalcule et les réécrit dans `model_final/config.json`.

## Modèles

1. **Isolation Forest** — baseline, détection ponctuelle (§ 8)
2. **Autoencodeur dense** — détection par erreur de reconstruction (§ 9)
3. **Autoencodeur LSTM** — dégradations progressives, séquences de 6 pas (§ 10)

**Le système final n'est aucun des trois pris isolément :** c'est la **fusion**
de l'autoencodeur dense et de l'autoencodeur LSTM (§ 11), chacun étant lui-même
un ensemble de 5 réseaux. Les deux scores sont convertis en **rangs** contre une
référence de validation figée, puis moyennés — ce qui neutralise la différence
d'échelle entre leurs erreurs de reconstruction. Une mesure est déclarée anormale
si ce score dépasse le seuil de `model_final/config.json`, choisi par **F2 max**
sur la validation : en exploitation PV, une panne ratée coûte plus cher qu'une
fausse alerte.

Approche non supervisée : entraînement sur données normales uniquement,
validation sur vraies pannes.

| Jeu | PR-AUC | Précision | Rappel | Épisodes détectés |
|---|---|---|---|---|
| Validation | 0,723 | 0,731 | 0,727 | 4/4 |
| Test | 0,777 | 0,703 | 0,973 | 3/3 |

Le rappel de test dépasse celui de validation parce que les deux jeux ne
contiennent pas les mêmes pannes (test : 3.1 ; validation : 3.2 et 4.0). Le
notebook consacre une cellule de limites à ce point, juste avant la § 15.10.
