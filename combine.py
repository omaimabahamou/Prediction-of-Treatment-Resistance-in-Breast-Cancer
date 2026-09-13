import pandas as pd
import glob
import os

OUTPUT_DIR = "./data/TCGA-BRCA"

def build_expression_matrix(data_dir):
    # Exclure le fichier clinique
    all_files = [f for f in glob.glob(f"{data_dir}/*.tsv")
                 if "clinical_data" not in f]

    total = len(all_files)
    print(f"📂 {total} fichiers RNA-seq trouvés...")

    dfs = []
    for i, f in enumerate(all_files, 1):
        try:
            df = pd.read_csv(f, sep="\t", comment="#",
                             usecols=["gene_name", "tpm_unstranded"])
            sample_id = os.path.basename(f).split(".")[0]
            df = df.rename(columns={"tpm_unstranded": sample_id})
            df = df.dropna(subset=["gene_name"])
            dfs.append(df.set_index("gene_name"))

            # Progression toutes les 100 fichiers
            if i % 100 == 0 or i == total:
                print(f"  [{i}/{total}] fichiers traités...")

        except Exception as e:
            print(f"  ⚠️ Erreur sur {f} : {e}")

    print("🔗 Fusion en cours (peut prendre 2-3 minutes)...")
    matrix = pd.concat(dfs, axis=1)

    output_path = f"{data_dir}/expression_matrix.csv"
    matrix.to_csv(output_path)

    print(f"\n✅ Matrice sauvegardée !")
    print(f"   📊 {matrix.shape[0]} gènes × {matrix.shape[1]} patients")
    print(f"   📄 Fichier → {output_path}")
    return matrix

if __name__ == "__main__":
    matrix = build_expression_matrix(OUTPUT_DIR)
    print("\nAperçu :")
    print(matrix.head())