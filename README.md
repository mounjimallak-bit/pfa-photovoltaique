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

[(https://zenodo.org/records/7358042)](https://zenodo.org/records/7828879?preview_file=00_metadata_variable.jpg) (article PMC9800176)

- `dt1` : données météo (GTI, DTI, TA, TPV)
- `dt2` : données électriques + étiquettes de pannes (ombrage)

## Modèles

1. **Isolation Forest** — baseline, détection ponctuelle
2. **Autoencoder** — détection par erreur de reconstruction
3. **LSTM Autoencoder** — prédiction de dégradations progressives (modèle principal)

Approche non supervisée : entraînement sur données normales uniquement, validation sur vraies pannes.
