# Breast Cancer Treatment Resistance Prediction (TCGA-BRCA)

Machine learning pipeline that predicts treatment resistance in breast cancer patients using RNA-seq gene expression data from the TCGA-BRCA cohort.

## Overview

- **Dataset**: TCGA-BRCA (1,231 tumor samples, 60,660 genes)
- **Model**: Logistic Regression with SMOTE class balancing
- **Performance**: AUC-ROC = 0.810, Sensitivity = 70.4%, Specificity = 81.4%
- **Top biomarker**: UGT2B4 (drug metabolism gene)

## How it works

1. RNA-seq expression data (log2-normalized) pulled from the GDC API
2. Resistance label built from survival data (died within 5 years = resistant)
3. Top 2,000 most variable genes selected as features
4. SMOTE used to balance the training set (134 resistant vs 858 sensitive patients)
5. Logistic Regression trained and evaluated with 5-fold cross-validation
6. Decision threshold optimized using the Youden index

## Key results

| Metric | Value |
|---|---|
| AUC-ROC (test set) | 0.810 |
| Sensitivity | 70.4% |
| Specificity | 81.4% |
| Top genes | UGT2B4, SMR3B, MANBAL, CGA, S100P |



## Reference

Data source: The Cancer Genome Atlas (TCGA), via the GDC Data Portal.
