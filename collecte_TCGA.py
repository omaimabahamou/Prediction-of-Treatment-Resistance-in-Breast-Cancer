import requests
import pandas as pd
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Configuration ───────────────────────────────────────────
BASE_URL = "https://api.gdc.cancer.gov"
OUTPUT_DIR = "./data/TCGA-BRCA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. Récupérer tous les fichiers RNA-seq ──────────────────
def get_rnaseq_files():
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "RNA-Seq"}},
            {"op": "=", "content": {"field": "data_format", "value": "TSV"}},
            {"op": "=", "content": {"field": "access", "value": "open"}}
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.case_id,cases.submitter_id",
        "format": "JSON",
        "size": 2000
    }

    response = requests.get(f"{BASE_URL}/files", params=params)
    data = response.json()
    files = data["data"]["hits"]
    print(f"✅ {len(files)} fichiers RNA-seq trouvés")
    return files

# ─── 2. Télécharger un seul fichier ──────────────────────────
def download_file(file_id, file_name, output_dir, retries=3):
    filepath = os.path.join(output_dir, file_name)

    # Sauter si déjà téléchargé
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return f"⏭️  Déjà téléchargé : {file_name}"

    for attempt in range(retries):
        try:
            url = f"{BASE_URL}/data/{file_id}"
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return f"✅ {file_name}"

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return f"❌ Échec ({file_name}) : {e}"

# ─── 3. Télécharger tous les fichiers en parallèle ───────────
def download_all(files, max_workers=5):
    total = len(files)
    print(f"\n📥 Téléchargement de {total} fichiers (parallèle x{max_workers})...")
    print(f"⏱️  Durée estimée : {total // 60} à {total // 30} minutes\n")

    success = 0
    failed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, f["file_id"], f["file_name"], OUTPUT_DIR): f
            for f in files
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()

            if "✅" in result:
                success += 1
            elif "⏭️" in result:
                skipped += 1
            else:
                failed += 1

            # Afficher progression toutes les 50 fichiers
            if i % 50 == 0 or i == total:
                print(f"  [{i}/{total}] ✅ {success} | ⏭️ {skipped} | ❌ {failed}")

    print(f"\n{'='*50}")
    print(f"✅ Succès  : {success}")
    print(f"⏭️  Ignorés : {skipped} (déjà téléchargés)")
    print(f"❌ Échecs  : {failed}")
    print(f"{'='*50}")
    print(f"📁 Fichiers dans : {OUTPUT_DIR}")

# ─── 4. Sauvegarder la liste des fichiers ────────────────────
def save_file_list(files):
    rows = []
    for f in files:
        rows.append({
            "file_id": f["file_id"],
            "file_name": f["file_name"],
            "case_id": f["cases"][0]["case_id"] if f.get("cases") else None,
            "submitter_id": f["cases"][0]["submitter_id"] if f.get("cases") else None
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/file_list.csv", index=False)
    print(f"✅ Liste sauvegardée → {OUTPUT_DIR}/file_list.csv")
    return df

# ─── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Récupérer la liste complète
    files = get_rnaseq_files()

    # 2. Sauvegarder la liste
    df_files = save_file_list(files)
    print(df_files.head())

    # 3. Télécharger TOUS les fichiers
    download_all(files, max_workers=5)

    print("\n🎉 Téléchargement complet terminé !")
    print("➡️  Prochaine étape : python combine.py")