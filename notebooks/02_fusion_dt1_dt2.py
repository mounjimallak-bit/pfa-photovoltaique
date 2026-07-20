
# %%
import pandas as pd
import numpy as np



print("Chargement dt2...")
dt2 = pd.read_csv("data/dt2_electrical_production_inverter_1_with_faults.csv")
dt2["time"] = pd.to_datetime(dt2["time"], utc=True)
dt2 = dt2.set_index("time").sort_index()
print(f"dt2 : {len(dt2):,} lignes")
print(f"Période : {dt2.index.min()} → {dt2.index.max()}")


# %%
print("Chargement dt1 par chunks...")
chunks = []
for i, chunk in enumerate(pd.read_csv(
    "data/dt1_solar_and_meteorological_measurement.csv",
    chunksize=500_000
)):
    chunk["time"] = pd.to_datetime(chunk["time"], utc=True)
    chunk = chunk.set_index("time").sort_index()
    chunks.append(chunk)
    print(f"  chunk {i+1} : {len(chunk):,} lignes")

dt1 = pd.concat(chunks)
print(f"\ndt1 total : {len(dt1):,} lignes")
print(f"Période : {dt1.index.min()} → {dt1.index.max()}")


# %%
print("Rééchantillonnage dt1 à 5 min...")
dt1_5min = dt1.resample("5min").mean()
dt1_5min = dt1_5min.dropna()
print(f"dt1 après resample : {len(dt1_5min):,} lignes")

# %%
print("Rééchantillonnage dt2 à 5 min...")
# Pour Fault : on prend la valeur la plus fréquente dans la fenêtre
fault_5min = dt2["Fault"].resample("5min").apply(
    lambda x: x.mode().iloc[0] if len(x) > 0 and not x.mode().empty else np.nan
)
# Pour les autres colonnes : moyenne
dt2_numeric = dt2.drop(columns=["Fault"]).resample("5min").mean()
dt2_5min = dt2_numeric.join(fault_5min).dropna()
print(f"dt2 après resample : {len(dt2_5min):,} lignes")

# %% [markdown]
# ## 4. Fusion (inner join sur le temps)
# On ne garde que les instants présents dans les DEUX fichiers

# %%
print("Fusion dt1 + dt2...")
df = dt1_5min.join(dt2_5min, how="inner")
print(f"Après fusion : {len(df):,} lignes x {len(df.columns)} colonnes")
print(f"Période : {df.index.min()} → {df.index.max()}")
print(f"\nColonnes : {df.columns.tolist()}")
print(f"\nRépartition Fault :")
print(df["Fault"].value_counts().sort_index())

# %% [markdown]
# ## 5. Filtrer heures de jour (GTI > 50 W/m²)

# %%
df_day = df[df["GTI"] > 50].copy()
print(f"Après filtrage jour (GTI > 50) : {len(df_day):,} lignes")
print(f"Supprimé : {len(df) - len(df_day):,} lignes de nuit/crépuscule")
print(f"\nRépartition Fault après filtrage :")
print(df_day["Fault"].value_counts().sort_index())

# %% [markdown]
# ## 6. Vérifier le résultat

# %%
print("=== Aperçu final ===")
print(df_day.head(10))
print(f"\n=== Stats ===")
print(df_day.describe())
print(f"\n=== Valeurs manquantes ===")
print(df_day.isnull().sum())

# %% [markdown]
# ## 7. Sauvegarder

# %%
output_path = "data/merged_5min_day.csv"
df_day.to_csv(output_path)
print(f"Sauvegardé : {output_path}")
print(f"Taille : {len(df_day):,} lignes x {len(df_day.columns)} colonnes")
print(f"Ce fichier sera utilisé pour l'entraînement et le test.")
