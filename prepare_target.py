import pandas as pd

# ─── Configuration ───────────────────────────────────────────
OUTPUT_DIR = "./data/TCGA-BRCA"

# ─── 1. Charger les données cliniques ────────────────────────
print("📂 Chargement des données cliniques...")
clinical = pd.read_csv(f"{OUTPUT_DIR}/clinical_data.tsv", sep="\t")
print(f"   {clinical.shape[0]} patients, colonnes : {list(clinical.columns)}")

# ─── 2. Charger la liste des fichiers (file_id → case_id) ────
file_list = pd.read_csv(f"{OUTPUT_DIR}/file_list.csv")

# ─── 3. Charger la matrice d'expression ──────────────────────
print("📂 Chargement de la matrice d'expression...")
matrix = pd.read_csv(f"{OUTPUT_DIR}/expression_matrix.csv", index_col="gene_name")
print(f"   {matrix.shape[0]} gènes × {matrix.shape[1]} patients")

# ─── 4. Afficher les colonnes cliniques disponibles ──────────
print("\n📋 Colonnes cliniques disponibles :")
for col in clinical.columns:
    print(f"   - {col}")