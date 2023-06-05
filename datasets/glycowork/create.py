import shutil

import pandas as pd
import os
from glyles.converter import convert
from Bio.PDB import PDBParser, PDBIO
import sys
import rdkit


def extract_chain(pdb_name, pdb_file, search_chain, output_file):
    io = PDBIO()
    pdb = PDBParser().get_structure(pdb_name, pdb_file)

    for chain in pdb.get_chains():
        if chain.get_id() == search_chain:
            io.set_structure(chain)
            io.save(output_file)
            return


def create(target_path, min_id=0.75, min_cov=0.6):
    os.makedirs(os.path.join(target_path, "tables"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "structures"), exist_ok=True)

    data = open("all_arrays.csv").readlines()
    lectins = pd.read_csv("model_summary.tsv", sep="\t")
    lectin_lines = open("Lectins.fasta").readlines()
    lectin_dict = {}
    for i in range(0, len(lectin_lines), 2):
        lectin_dict[lectin_lines[i + 1].strip()] = lectin_lines[i].strip()[1:]
    omega_lectins = os.listdir("omegafold")

    inter = open(os.path.join(target_path, "tables", "inter.tsv"), "w")
    inter.write("Drug_ID\tTarget_ID\tY\n")
    lig = open(os.path.join(target_path, "tables", "lig.tsv"), "w")
    lig.write("Drug_ID\tDrug\tIUPAC\n")
    prots = open(os.path.join(target_path, "tables", "prot.tsv"), "w")
    prots.write("Target_ID\tTarget\n")
    weights = open(os.path.join(target_path, "tables", "weights.tsv"), "w")
    weights.write("Target_ID\tCount\n")

    glycans = []
    for d, line in enumerate(data):
        if d == 0:
            glycans = [x.strip() for x in line.split(",")[:-1]]
            for i, glycan in enumerate(glycans):
                smiles = convert(glycan, returning=True)[0][1]
                if rdkit.Chem.MolFromSmiles(smiles):
                    lig.write(f"Gly{(i + 1):05d}\t{smiles}\t{glycan}\n")
        else:
            parts = line.strip().split(",")
            seq = parts[-1]
            if seq not in lectin_dict or any(not x.isalpha() or not x.isupper() for x in seq):
                print("exit -", seq)
                continue
            lectin = lectin_dict[seq]
            print(lectin)
            row = lectins[lectins["Input ID"] == lectin]
            if len(row) < 1 or (((row["Sequence identity"] < min_id).bool() or (row["Coverage"] < min_cov).bool()) and lectin + ".pdb" not in omega_lectins):
                continue
            prots.write(f"{lectin}\t{seq}\n")
            if lectin + ".pdb" in omega_lectins:
                shutil.copy(os.path.join("omegafold", lectin + ".pdb"), os.path.join(target_path, "structures", lectin + ".pdb"))
            else:
                extract_chain(lectin, f"models/{lectin}_B99990001_refined.pdb", row["Model chain"].values[0], os.path.join(target_path, "structures", lectin + ".pdb"))

            counter = 0
            for i, (glycan, value) in enumerate(zip(glycans, parts[:-1])):
                if len(value) > 0:
                    counter += 1
                    value = float(value)
                    inter.write(f"Gly{(i + 1):05d}\t{lectin}\t{value}\n")
            weights.write(f"{lectin}\t{counter}\n")
    inter.close()
    lig.close()
    prots.close()
    weights.close()

                
if __name__ == "__main__":
    create(sys.argv[1])  # , float(sys.argv[2]), float(sys.argv[3]))
