import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "./data/TCGA-BRCA"

# ─── 1. Charger et aligner ───────────────────────────────────
print("📂 Chargement...")
labels = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")
matrix = pd.read_csv(f"{OUTPUT_DIR}/expression_matrix.csv", index_col="gene_name")

labels_clean = labels.dropna(subset=["resistance"]).copy()
matrix_T = matrix.T
labels_clean["file_prefix"] = labels_clean["file_name"].apply(lambda x: x.split(".")[0])
labels_aligned = labels_clean[labels_clean["file_prefix"].isin(matrix_T.index)]
matrix_aligned = matrix_T.loc[labels_aligned["file_prefix"].values]

X = matrix_aligned.values.astype(float)
y = labels_aligned["resistance"].values.astype(int)

# ─── 2. Sélection gènes + normalisation ──────────────────────
variances = X.var(axis=0)
top_idx = np.argsort(variances)[::-1][:2000]
X = np.log2(X[:, top_idx] + 1)
gene_names = matrix_aligned.columns[top_idx]

# ─── 3. Split train/test ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─── 4. Tester plusieurs seuils de décision ──────────────────
print("\n🎯 Optimisation du seuil de décision...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
y_proba = rf.predict_proba(X_test)[:, 1]

print(f"\n{'Seuil':>8} | {'AUC':>6} | {'Recall Résistant':>16} | {'Précision':>10} | {'F1':>6}")
print("-" * 60)
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_proba >= threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    rec  = recall_score(y_test, y_pred_t, zero_division=0)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    f1   = f1_score(y_test, y_pred_t, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    print(f"  {threshold:>6.1f} | {auc:>6.3f} | {rec:>16.3f} | {prec:>10.3f} | {f1:>6.3f}")

# ─── 5. Comparer plusieurs modèles ───────────────────────────
print("\n\n🤖 Comparaison de modèles (5-fold CV)...")
models = {
    "Random Forest"       : RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression" : LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Modèle':25} | {'AUC moyen':>10} | {'Std':>6}")
print("-" * 50)
for name, model in models.items():
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])
    scores = cross_val_score(pipeline, X_train, y_train,
                              cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  {name:23} | {scores.mean():>10.3f} | {scores.std():>6.3f}")

# ─── 6. Top gènes biomarqueurs ───────────────────────────────
print("\n🧬 Top 20 gènes biomarqueurs :")
importances = pd.Series(rf.feature_importances_, index=gene_names)
top_genes = importances.sort_values(ascending=False).head(20)
print(top_genes.to_string())
top_genes.to_csv(f"{OUTPUT_DIR}/top_genes.csv", header=["importance"])
print(f"\n✅ Sauvegardé → {OUTPUT_DIR}/top_genes.csv")