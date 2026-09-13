import pandas as pd
import numpy as np

OUTPUT_DIR = "./data/TCGA-BRCA"

# ─── 1. Charger les données ───────────────────────────────────
clinical = pd.read_csv(f"{OUTPUT_DIR}/clinical_complet.csv")
file_list = pd.read_csv(f"{OUTPUT_DIR}/file_list.csv")

print(f"✅ Clinical : {clinical.shape[0]} patients")
print(f"✅ File list : {file_list.shape[0]} fichiers")

# ─── 2. Créer la variable cible ───────────────────────────────
# Logique :
# "résistant" (1) = décédé avec survie courte (< 3 ans = 1095 jours)
# "sensible"  (0) = vivant OU décédé après longue survie (>= 3 ans)

SEUIL_JOURS = 1095  # 3 ans

def definir_resistance(row):
    vital = row["vital_status"]
    days_death = row["days_to_death"]
    days_follow = row["days_to_last_follow_up"]

    # Décédé avec survie courte → résistant
    if vital == "Dead":
        if pd.notna(days_death) and days_death < SEUIL_JOURS:
            return 1  # résistant
        else:
            return 0  # décédé tardivement → sensible

    # Vivant → sensible
    elif vital == "Alive":
        return 0

    return np.nan  # inconnu

clinical["resistance"] = clinical.apply(definir_resistance, axis=1)

# ─── 3. Statistiques ─────────────────────────────────────────
print(f"\n📊 Distribution de la variable cible :")
print(clinical["resistance"].value_counts())
print(f"\n   0 = Sensible : {(clinical['resistance'] == 0).sum()} patients")
print(f"   1 = Résistant : {(clinical['resistance'] == 1).sum()} patients")
print(f"   NaN = Inconnu : {clinical['resistance'].isna().sum()} patients")

# ─── 4. Fusionner avec file_list ──────────────────────────────
# file_list contient : file_id, file_name, case_id, submitter_id
merged = file_list.merge(
    clinical[["case_id", "submitter_id", "vital_status",
              "days_to_death", "days_to_last_follow_up", "resistance"]],
    on="case_id",
    how="inner"
)

print(f"\n✅ Après fusion : {merged.shape[0]} patients avec RNA-seq + clinique")
print(f"   0 = Sensible  : {(merged['resistance'] == 0).sum()}")
print(f"   1 = Résistant : {(merged['resistance'] == 1).sum()}")

# ─── 5. Sauvegarder ──────────────────────────────────────────
# Table complète
merged.to_csv(f"{OUTPUT_DIR}/labels.csv", index=False)
print(f"\n✅ Labels sauvegardés → {OUTPUT_DIR}/labels.csv")

# Aperçu
print("\nAperçu :")
print(merged[["case_id", "vital_status", "days_to_death",
              "days_to_last_follow_up", "resistance"]].head(10))