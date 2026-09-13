import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "./data/TCGA-BRCA"

# ─── 1. Charger les données ───────────────────────────────────
print("📂 Chargement...")
labels   = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")
clinical = pd.read_csv(f"{OUTPUT_DIR}/clinical_complet.csv")
matrix   = pd.read_csv(f"{OUTPUT_DIR}/expression_matrix.csv", index_col="gene_name")

# ─── 2. Redéfinir la variable cible (seuil élargi) ───────────
# Nouvelle logique plus équilibrée :
# Résistant (1) = décédé en < 5 ans (1825 jours)
# Sensible  (0) = vivant > 3 ans OU suivi long
SEUIL = 1825  # 5 ans

def definir_resistance_v2(row):
    vital = row["vital_status"]
    days_death  = row["days_to_death"]
    days_follow = row["days_to_last_follow_up"]

    if vital == "Dead" and pd.notna(days_death):
        return 1 if days_death < SEUIL else 0
    elif vital == "Alive" and pd.notna(days_follow):
        return 0 if days_follow > 365 else np.nan
    return np.nan

clinical["resistance_v2"] = clinical.apply(definir_resistance_v2, axis=1)

print(f"\n📊 Nouvelle distribution (seuil 5 ans) :")
print(f"   Sensible  (0) : {(clinical['resistance_v2']==0).sum()}")
print(f"   Résistant (1) : {(clinical['resistance_v2']==1).sum()}")
print(f"   Inconnu   (?) : {clinical['resistance_v2'].isna().sum()}")

# ─── 3. Fusionner avec file_list ─────────────────────────────
file_list = pd.read_csv(f"{OUTPUT_DIR}/file_list.csv")
merged = file_list.merge(
    clinical[["case_id", "resistance_v2"]],
    on="case_id", how="inner"
).dropna(subset=["resistance_v2"])

# ─── 4. Aligner avec la matrice ──────────────────────────────
matrix_T = matrix.T
merged["file_prefix"] = merged["file_name"].apply(lambda x: x.split(".")[0])
merged = merged[merged["file_prefix"].isin(matrix_T.index)]
matrix_aligned = matrix_T.loc[merged["file_prefix"].values]

X = matrix_aligned.values.astype(float)
y = merged["resistance_v2"].values.astype(int)

print(f"\n✅ Dataset final : {len(y)} patients")
print(f"   Résistants : {y.sum()} | Sensibles : {(y==0).sum()}")

# ─── 5. Sélection gènes variables ────────────────────────────
print("\n🔬 Sélection des 2000 gènes variables...")
variances = X.var(axis=0)
top_idx   = np.argsort(variances)[::-1][:2000]
X         = np.log2(X[:, top_idx] + 1)
gene_names = matrix_aligned.columns[top_idx]

# ─── 6. Split Train/Test ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n✂️  Train : {len(y_train)} | Test : {len(y_test)}")

# ─── 7. SMOTE sur train ───────────────────────────────────────
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"   Après SMOTE : {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

# ─── 8. Modèles ───────────────────────────────────────────────
models = {
    "Random Forest"      : RandomForestClassifier(
                               n_estimators=300, class_weight="balanced",
                               max_depth=10, random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(
                               class_weight="balanced", C=0.1,
                               max_iter=1000, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n🤖 Validation croisée 5-fold :")
print(f"{'Modèle':25} | {'AUC':>6} | {'Std':>5}")
print("-" * 45)

best_model = None
best_auc   = 0

for name, model in models.items():
    pipe = Pipeline([("smote", SMOTE(random_state=42)), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train,
                              cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  {name:23} | {scores.mean():.3f} | {scores.std():.3f}")
    if scores.mean() > best_auc:
        best_auc   = scores.mean()
        best_model = (name, model)

# ─── 9. Évaluation finale sur TEST ───────────────────────────
print(f"\n🏆 Meilleur modèle : {best_model[0]}")
best_model[1].fit(X_train_res, y_train_res)
y_proba = best_model[1].predict_proba(X_test)[:, 1]

# Seuil optimal via courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
optimal_idx       = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"   Seuil optimal : {optimal_threshold:.3f}")

y_pred = (y_proba >= optimal_threshold).astype(int)
auc    = roc_auc_score(y_test, y_proba)

print(f"\n✅ AUC-ROC final : {auc:.3f}")
print(f"\n📋 Rapport de classification :")
print(classification_report(y_test, y_pred,
      target_names=["Sensible", "Résistant"]))

cm = confusion_matrix(y_test, y_pred)
print(f"🔲 Matrice de confusion :")
print(f"   Vrais Négatifs  (TN) : {cm[0,0]}")
print(f"   Faux Positifs   (FP) : {cm[0,1]}")
print(f"   Faux Négatifs   (FN) : {cm[1,0]}")
print(f"   Vrais Positifs  (TP) : {cm[1,1]}")

# ─── 10. Top gènes ───────────────────────────────────────────
if hasattr(best_model[1], "feature_importances_"):
    importances = pd.Series(best_model[1].feature_importances_, index=gene_names)
else:
    importances = pd.Series(np.abs(best_model[1].coef_[0]), index=gene_names)

top_genes = importances.sort_values(ascending=False).head(20)
print(f"\n🧬 Top 20 gènes biomarqueurs :")
print(top_genes.to_string())
top_genes.to_csv(f"{OUTPUT_DIR}/top_genes_final.csv", header=["importance"])
print(f"\n✅ Résultats sauvegardés → {OUTPUT_DIR}/top_genes_final.csv")