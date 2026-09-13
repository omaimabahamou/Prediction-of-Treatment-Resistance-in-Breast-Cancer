import requests
import json
import pandas as pd

BASE_URL = "https://api.gdc.cancer.gov"
OUTPUT_DIR = "./data/TCGA-BRCA"

def download_clinical_complet():
    filters = {
        "op": "=",
        "content": {
            "field": "cases.project.project_id",
            "value": "TCGA-BRCA"
        }
    }

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join([
            "case_id",
            "submitter_id",
            # Survie
            "diagnoses.vital_status",
            "diagnoses.days_to_death",
            "diagnoses.days_to_last_follow_up",
            "diagnoses.days_to_last_known_disease_status",
            "diagnoses.last_known_disease_status",
            # Tumeur
            "diagnoses.tumor_stage",
            "diagnoses.tumor_grade",
            "diagnoses.morphology",
            "diagnoses.primary_diagnosis",
            # Réponse traitement
            "diagnoses.progression_or_recurrence",
            "diagnoses.days_to_recurrence",
            "diagnoses.days_to_best_overall_response",
            "diagnoses.best_overall_response",
            # Traitements
            "treatments.treatment_type",
            "treatments.treatment_or_therapy",
            "treatments.days_to_treatment_start",
            "treatments.days_to_treatment_end",
            "treatments.therapeutic_agents",
            "treatments.treatment_intent_type",
            "treatments.treatment_outcome",
            # Démographie
            "demographic.gender",
            "demographic.age_at_index",
            "demographic.race",
            "demographic.ethnicity",
            "demographic.vital_status",
            "demographic.days_to_death",
        ]),
        "format": "JSON",
        "size": 2000,
        "expand": "diagnoses,treatments,demographic"
    }

    print("📥 Téléchargement des données cliniques complètes...")
    response = requests.get(f"{BASE_URL}/cases", params=params)
    data = response.json()
    cases = data["data"]["hits"]
    print(f"✅ {len(cases)} patients récupérés")

    # ─── Aplatir les données ─────────────────────────────────
    rows = []
    for case in cases:
        row = {
            "case_id": case.get("case_id"),
            "submitter_id": case.get("submitter_id"),
        }

        # Démographie
        demo = case.get("demographic", {})
        row["gender"] = demo.get("gender")
        row["age_at_index"] = demo.get("age_at_index")
        row["race"] = demo.get("race")
        row["vital_status"] = demo.get("vital_status")
        row["days_to_death"] = demo.get("days_to_death")

        # Diagnostic (premier)
        diagnoses = case.get("diagnoses", [{}])
        diag = diagnoses[0] if diagnoses else {}
        row["tumor_stage"] = diag.get("tumor_stage")
        row["tumor_grade"] = diag.get("tumor_grade")
        row["primary_diagnosis"] = diag.get("primary_diagnosis")
        row["progression_or_recurrence"] = diag.get("progression_or_recurrence")
        row["days_to_recurrence"] = diag.get("days_to_recurrence")
        row["days_to_last_follow_up"] = diag.get("days_to_last_follow_up")
        row["last_known_disease_status"] = diag.get("last_known_disease_status")
        row["best_overall_response"] = diag.get("best_overall_response")

        # Traitements (tous)
        treatments = case.get("treatments", [])
        row["nb_treatments"] = len(treatments)
        row["treatment_types"] = "|".join([t.get("treatment_type", "") for t in treatments])
        row["therapeutic_agents"] = "|".join([t.get("therapeutic_agents", "") or "" for t in treatments])
        row["treatment_outcomes"] = "|".join([t.get("treatment_outcome", "") or "" for t in treatments])
        row["treatment_or_therapy"] = "|".join([t.get("treatment_or_therapy", "") or "" for t in treatments])

        rows.append(row)

    df = pd.DataFrame(rows)

    # ─── Sauvegarder ─────────────────────────────────────────
    output_path = f"{OUTPUT_DIR}/clinical_complet.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Sauvegardé → {output_path}")
    print(f"\n📋 Colonnes disponibles :")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"   - {col:40s} {non_null}/{len(df)} patients renseignés")

    return df

if __name__ == "__main__":
    df = download_clinical_complet()
    print("\nAperçu :")
    print(df.head())