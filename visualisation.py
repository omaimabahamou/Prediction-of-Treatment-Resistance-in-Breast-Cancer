import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR  = "./data/TCGA-BRCA"
FIGURES_DIR = "./figures"
import os; os.makedirs(FIGURES_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# 1. RECHARGER ET PRÉPARER LES DONNÉES
# ════════════════════════════════════════════════════════════════
print("📂 Chargement des données...")
labels   = pd.read_csv(f"{OUTPUT_DIR}/labels.csv")
clinical = pd.read_csv(f"{OUTPUT_DIR}/clinical_complet.csv")
matrix   = pd.read_csv(f"{OUTPUT_DIR}/expression_matrix.csv", index_col="gene_name")

SEUIL = 1825
def definir_resistance(row):
    vital = row["vital_status"]
    days_death  = row["days_to_death"]
    days_follow = row["days_to_last_follow_up"]
    if vital == "Dead" and pd.notna(days_death):
        return 1 if days_death < SEUIL else 0
    elif vital == "Alive" and pd.notna(days_follow):
        return 0 if days_follow > 365 else np.nan
    return np.nan

clinical["resistance"] = clinical.apply(definir_resistance, axis=1)
file_list = pd.read_csv(f"{OUTPUT_DIR}/file_list.csv")
merged = file_list.merge(clinical[["case_id","resistance","vital_status",
                                    "days_to_death","days_to_last_follow_up"]],
                          on="case_id", how="inner").dropna(subset=["resistance"])

matrix_T = matrix.T
merged["file_prefix"] = merged["file_name"].apply(lambda x: x.split(".")[0])
merged = merged[merged["file_prefix"].isin(matrix_T.index)]
matrix_aligned = matrix_T.loc[merged["file_prefix"].values]

X = matrix_aligned.values.astype(float)
y = merged["resistance"].values.astype(int)

variances  = X.var(axis=0)
top_idx    = np.argsort(variances)[::-1][:2000]
X          = np.log2(X[:, top_idx] + 1)
gene_names = matrix_aligned.columns[top_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_tr, y_tr = smote.fit_resample(X_train, y_train)

model = LogisticRegression(class_weight="balanced", C=0.1,
                            max_iter=1000, random_state=42)
model.fit(X_tr, y_tr)
y_proba = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

threshold = 0.104
y_pred = (y_proba >= threshold).astype(int)
cm     = confusion_matrix(y_test, y_pred)

top_genes = pd.read_csv(f"{OUTPUT_DIR}/top_genes_final.csv",
                         index_col="gene_name")

print("✅ Données prêtes — génération des figures...\n")

# Style global
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})
BLUE  = "#2563EB"
RED   = "#DC2626"
GREEN = "#16A34A"
GRAY  = "#6B7280"

# ════════════════════════════════════════════════════════════════
# FIGURE 1 — COURBE ROC
# ════════════════════════════════════════════════════════════════
print("📈 Figure 1 : Courbe ROC...")
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
opt_idx = np.argmax(tpr - fpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=BLUE, lw=2.5,
        label=f"Logistic Regression (AUC = {auc:.3f})")
ax.plot([0,1],[0,1], color=GRAY, lw=1.5, linestyle="--", label="Hasard (AUC = 0.500)")
ax.scatter(fpr[opt_idx], tpr[opt_idx], color=RED, s=120, zorder=5,
           label=f"Seuil optimal = {thresholds[opt_idx]:.3f}")
ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
ax.set_xlabel("Taux de Faux Positifs (1 - Spécificité)", fontsize=12)
ax.set_ylabel("Taux de Vrais Positifs (Sensibilité)", fontsize=12)
ax.set_title("Courbe ROC — Prédiction de Résistance au Traitement\nCancer du Sein (TCGA-BRCA)",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/1_courbe_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé → figures/1_courbe_roc.png")

# ════════════════════════════════════════════════════════════════
# FIGURE 2 — MATRICE DE CONFUSION
# ════════════════════════════════════════════════════════════════
print("📊 Figure 2 : Matrice de confusion...")
fig, ax = plt.subplots(figsize=(6, 5))
labels_cm = ["Sensible", "Résistant"]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_cm, yticklabels=labels_cm,
            linewidths=2, linecolor="white",
            annot_kws={"size": 18, "weight": "bold"}, ax=ax)
ax.set_xlabel("Classe Prédite",  fontsize=12, labelpad=10)
ax.set_ylabel("Classe Réelle",   fontsize=12, labelpad=10)
ax.set_title("Matrice de Confusion\n(Seuil optimal = 0.104)",
             fontsize=13, fontweight="bold", pad=15)

# Annotations
tn, fp, fn, tp = cm.ravel()
ax.text(1.35, 0.5, f"Sensibilité\n{tp/(tp+fn):.1%}",
        transform=ax.transAxes, fontsize=10, color=GREEN,
        va="center", ha="center", fontweight="bold")
ax.text(1.35, 0.85, f"Spécificité\n{tn/(tn+fp):.1%}",
        transform=ax.transAxes, fontsize=10, color=BLUE,
        va="center", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/2_matrice_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé → figures/2_matrice_confusion.png")

# ════════════════════════════════════════════════════════════════
# FIGURE 3 — TOP 20 GÈNES BIOMARQUEURS
# ════════════════════════════════════════════════════════════════
print("🧬 Figure 3 : Top 20 gènes biomarqueurs...")
top20 = top_genes.head(20).sort_values("importance")
colors = [RED if v > top20["importance"].median() else BLUE
          for v in top20["importance"]]

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(top20.index, top20["importance"], color=colors,
               edgecolor="white", height=0.7)
for bar, val in zip(bars, top20["importance"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9, color="#374151")
ax.set_xlabel("Importance du Gène (|Coefficient|)", fontsize=12)
ax.set_title("Top 20 Gènes Biomarqueurs de Résistance\n(Logistic Regression — TCGA-BRCA)",
             fontsize=13, fontweight="bold", pad=15)
ax.axvline(top20["importance"].median(), color=GRAY,
           linestyle="--", lw=1.5, label="Médiane")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/3_top_genes.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé → figures/3_top_genes.png")

# ════════════════════════════════════════════════════════════════
# FIGURE 4 — DISTRIBUTION DES PROBABILITÉS
# ════════════════════════════════════════════════════════════════
print("📉 Figure 4 : Distribution des probabilités...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y_proba[y_test == 0], bins=30, alpha=0.6,
        color=BLUE, label="Sensible (0)", edgecolor="white")
ax.hist(y_proba[y_test == 1], bins=15, alpha=0.7,
        color=RED,  label="Résistant (1)", edgecolor="white")
ax.axvline(threshold, color="black", linestyle="--", lw=2,
           label=f"Seuil = {threshold}")
ax.set_xlabel("Probabilité Prédite de Résistance", fontsize=12)
ax.set_ylabel("Nombre de Patients", fontsize=12)
ax.set_title("Distribution des Scores de Prédiction\npar Classe Réelle",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/4_distribution_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé → figures/4_distribution_scores.png")

# ════════════════════════════════════════════════════════════════
# FIGURE 5 — HEATMAP EXPRESSION TOP GÈNES
# ════════════════════════════════════════════════════════════════
print("🔥 Figure 5 : Heatmap expression des top gènes...")
top10_genes = top_genes.head(10).index.tolist()
gene_idx    = [list(gene_names).index(g) for g in top10_genes if g in gene_names]

X_all = np.log2(matrix_aligned.values.astype(float)[:, top_idx] + 1)
X_top = X_all[:, [list(gene_names).index(g)
                   for g in top10_genes if g in gene_names]]

df_heat = pd.DataFrame(X_top,
                        columns=[g for g in top10_genes if g in gene_names])
df_heat["resistance"] = y
df_heat = df_heat.sort_values("resistance")

sample_s = df_heat[df_heat["resistance"]==0].sample(50, random_state=42)
sample_r = df_heat[df_heat["resistance"]==1].sample(
               min(50, (df_heat["resistance"]==1).sum()), random_state=42)
df_plot  = pd.concat([sample_s, sample_r])

heat_data = df_plot.drop("resistance", axis=1).T
col_colors = ["#BFDBFE" if r == 0 else "#FCA5A5"
              for r in df_plot["resistance"]]

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(heat_data, cmap="RdBu_r", center=0,
            xticklabels=False, yticklabels=True,
            cbar_kws={"label": "Expression log2(TPM+1)"},
            ax=ax, linewidths=0)

# Barre de couleur patients
for i, c in enumerate(col_colors):
    ax.add_patch(plt.Rectangle((i, -0.8), 1, 0.5,
                 color=c, clip_on=False, transform=ax.transData))

ax.set_title("Heatmap d'Expression — Top 10 Gènes Biomarqueurs\n"
             "🔵 Sensible   🔴 Résistant",
             fontsize=13, fontweight="bold", pad=20)
ax.set_ylabel("Gènes", fontsize=11)
ax.set_xlabel("Patients (50 sensibles + résistants)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/5_heatmap_genes.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé → figures/5_heatmap_genes.png")

# ════════════════════════════════════════════════════════════════
# FIGURE 6 — TABLEAU RÉCAPITULATIF
# ════════════════════════════════════════════════════════════════
print("📋 Figure 6 : Tableau récapitulatif...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")

data_table = [
    ["Patients total",           "992"],
    ["Sensibles (0)",            "858"],
    ["Résistants (1)",           "134"],
    ["Gènes analysés",           "60 660"],
    ["Gènes sélectionnés",       "2 000"],
    ["Modèle",                   "Logistic Regression"],
    ["AUC-ROC (test)",           "0.810"],
    ["Sensibilité",              "70.4%"],
    ["Spécificité",              "81.4%"],
    ["Seuil optimal",            "0.104"],
    ["Top biomarqueur",          "UGT2B4"],
]

table = ax.table(cellText=data_table,
                 colLabels=["Paramètre", "Valeur"],
                 cellLoc="left", loc="center",
                 colWidths=[0.55, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1E40AF")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#EFF6FF")
    cell.set_edgecolor("white")

ax.set_title("Résumé du Projet — Prédiction Résistance Traitement\nCancer du Sein (TCGA-BRCA)",
             fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/6_tableau_resume.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé → figures/6_tableau_resume.png")

# ════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("🎉 Toutes les figures générées dans ./figures/")
print("="*55)
print("   1_courbe_roc.png")
print("   2_matrice_confusion.png")
print("   3_top_genes.png")
print("   4_distribution_scores.png")
print("   5_heatmap_genes.png")
print("   6_tableau_resume.png")