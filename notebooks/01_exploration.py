# %% [markdown]
# # Exploration du dataset Zenodo — PFA Photovoltaïque
# dt1 = météo (2.2 Go), dt2 = électrique + Fault (202 Mo)

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ## 1. Peek dt2 (le plus petit, celui avec les pannes)

# %%
dt2 = pd.read_csv("data/dt2_electrical_production_inverter_1_with_faults.csv", nrows=5)
print("=== Colonnes dt2 ===")
print(dt2.columns.tolist())
print("\n=== 5 premières lignes ===")
dt2

# %%
# Charger dt2 en entier (202 Mo, ça passe en RAM)
dt2 = pd.read_csv("data/dt2_electrical_production_inverter_1_with_faults.csv")
print(f"dt2 : {dt2.shape[0]:,} lignes x {dt2.shape[1]} colonnes")
print(f"Mémoire : {dt2.memory_usage(deep=True).sum() / 1e6:.0f} Mo")
print("\n=== Types ===")
print(dt2.dtypes)
print("\n=== Valeurs manquantes ===")
print(dt2.isnull().sum())
print("\n=== Stats ===")
dt2.describe()

# %%
# Distribution des pannes
print("=== Répartition Fault ===")
print(dt2["Fault"].value_counts().sort_index())

# %% [markdown]
# ## 2. Peek dt1 (2.2 Go — on ne charge que 5 lignes d'abord)

# %%
dt1_peek = pd.read_csv("data/dt1_solar_and_meteorological_measurement.csv", nrows=5)
print("=== Colonnes dt1 ===")
print(dt1_peek.columns.tolist())
print("\n=== 5 premières lignes ===")
dt1_peek

# %%
# Charger un échantillon de 100 000 lignes pour explorer
dt1_sample = pd.read_csv(
    "data/dt1_solar_and_meteorological_measurement.csv",
    nrows=100_000
)
print(f"dt1 sample : {dt1_sample.shape[0]:,} lignes x {dt1_sample.shape[1]} colonnes")
print("\n=== Types ===")
print(dt1_sample.dtypes)
print("\n=== Valeurs manquantes ===")
print(dt1_sample.isnull().sum())
print("\n=== Stats ===")
dt1_sample.describe()

# %% [markdown]
# ## 3. Vérifier le format des dates

# %%
print("=== Format date dt2 ===")
print(dt2.iloc[:3, 0])  # première colonne (probablement le temps)

print("\n=== Format date dt1 ===")
print(dt1_sample.iloc[:3, 0])

# %% [markdown]
# ## 4. Prochaine étape
# Après avoir vu les colonnes et les formats :
# - Identifier la colonne temps dans chaque fichier
# - Vérifier si les formats de date sont identiques
# - Lancer la fusion dans 02_fusion_dt1_dt2.py
