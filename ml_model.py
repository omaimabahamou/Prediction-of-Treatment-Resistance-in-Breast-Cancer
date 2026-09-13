import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "./data/TCGA-BRCA"

# ─── 1. Charger les données ───────────────────────────────────
print("📂 Chargement des données...")
labels = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")
matrix = pd.read_csv(f"{OUTPUT_DIR}/expression_matrix.csv", index_col="gene_name")

# ─── 2. Aligner patients ──────────────────────────────────────
labels_clean = labels.dropna(subset=["resistance"])
matrix_T = matrix.T
labels_clean = labels_clean.copy()
labels_clean["file_prefix"] = labels_clean["file_name"].apply(
    lambda x: x.split(".")[0]
)
common = labels_clean["file_prefix"][
    labels_clean["file_prefix"].isin(matrix_T.index)
]
labels_aligned = labels_clean[labels_clean["file_prefix"].isin(common)]
matrix_aligned = matrix_T.loc[labels_aligned["file_prefix"].values]

print(f"✅ Patients alignés : {len(labels_aligned)}")
print(f"   Résistants : {(labels_aligned['resistance'] == 1).sum()}")
print(f"   Sensibles  : {(labels_aligned['resistance'] == 0).sum()}")

# ─── 3. Préparer X et y ───────────────────────────────────────
X = matrix_aligned.values.astype(float)
y = labels_aligned["resistance"].values.astype(int)

# ─── 4. Sélection des gènes variables ────────────────────────
print("\n🔬 Sélection des 2000 gènes les plus variables...")
variances = X.var(axis=0)
top_idx = np.argsort(variances)[::-1][:2000]
X = X[:, top_idx]
gene_names = matrix_aligned.columns[top_idx]

# ─── 5. Normalisation log2 ────────────────────────────────────
X = np.log2(X + 1)

# ─── 6. Split TRAIN / TEST (80/20) AVANT SMOTE ───────────────
print("\n✂️  Séparation Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"   Train : {X_train.shape[0]} patients")
print(f"   Test  : {X_test.shape[0]} patients")

# ─── 7. Normalisation StandardScaler (fit sur train seulement) 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)  # ← jamais fit sur le test !

# ─── 8. SMOTE uniquement sur le train ────────────────────────
print("\n⚖️  SMOTE sur train uniquement...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"   Avant : {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"   Après : {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

# ─── 9. Entraînement Random Forest ───────────────────────────
print("\n🌲 Entraînement Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_res, y_train_res)

# ─── 10. Évaluation sur TEST (données jamais vues) ────────────
print("\n📊 Évaluation sur données TEST (jamais vues)...")
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print(f"\n✅ AUC-ROC sur test : {auc:.3f}")
print(f"\n📋 Rapport de classification :")
print(classification_report(y_test, y_pred,
      target_names=["Sensible", "Résistant"]))

print(f"🔲 Matrice de confusion :")
cm = confusion_matrix(y_test, y_pred)
print(f"   Vrais Négatifs  (TN) : {cm[0,0]}")
print(f"   Faux Positifs   (FP) : {cm[0,1]}")
print(f"   Faux Négatifs   (FN) : {cm[1,0]}")
print(f"   Vrais Positifs  (TP) : {cm[1,1]}")

# ─── 11. Validation croisée sur train (robustesse) ────────────
print("\n🔄 Validation croisée 5-fold sur train...")
rf2 = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    random_state=42, n_jobs=-1
)
pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf", rf2)
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train,
                             cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"   AUC par fold : {[round(s,3) for s in cv_scores]}")
print(f"   AUC moyen    : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ─── 12. Top gènes importants ────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=gene_names)
top_genes = importances.sort_values(ascending=False).head(20)
print(f"\n🧬 Top 20 gènes biomarqueurs :")
print(top_genes.to_string())
top_genes.to_csv(f"{OUTPUT_DIR}/top_genes.csv", header=["importance"])
print(f"\n✅ Sauvegardé → {OUTPUT_DIR}/top_genes.csv")