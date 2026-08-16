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
| ML | Scikit-learn, PyTorch |
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
├── data/                  # Données (exclu de Git)
├── models/                # Modèles entraînés (.pkl, .pt)
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_fusion_dt1_dt2.ipynb
│   ├── 03_isolation_forest.ipynb
│   ├── 04_autoencoder.ipynb
│   └── 05_lstm_autoencoder.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── replayer.py
│   ├── consumer.py
│   └── db.py
├── docker/
│   └── init.sql
├── tests/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Dataset

[Zenodo — La Réunion PV dataset](https://zenodo.org/records/7358042) (article PMC9800176)

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

1. **Isolation Forest** — baseline, détection ponctuelle
2. **Autoencoder** — détection par erreur de reconstruction
3. **LSTM Autoencoder** — prédiction de dégradations progressives (modèle principal)

Approche non supervisée : entraînement sur données normales uniquement, validation sur vraies pannes.
