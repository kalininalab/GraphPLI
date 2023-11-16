import shutil
from pathlib import Path

import pandas as pd
import os

from glyles import Glycan
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


def create(source_path, target_path, min_id=0.75, min_cov=0.6, max_mono=4, max_aas=350):
    os.makedirs(target_path / "tables", exist_ok=True)
    os.makedirs(target_path / "structures", exist_ok=True)

    data = open(source_path / "all_arrays.csv").readlines()
    lectins = pd.read_csv(source_path / "model_summary.tsv", sep="\t")
    lectin_lines = open(source_path / "Lectins.fasta").readlines()
    lectin_dict = {}
    for i in range(0, len(lectin_lines), 2):
        lectin_dict[lectin_lines[i + 1].strip()] = lectin_lines[i].strip()[1:]
    omega_lectins = os.listdir(source_path / "omegafold")

    inter = open(target_path / "tables" / "inter.tsv", "w")
    print("Drug_ID", "Target_ID", "Y", sep="\t", file=inter)
    lig = open(target_path / "tables" / "lig.tsv", "w")
    print("Drug_ID", "Drug", sep="\t", file=lig)
    prots = open(target_path / "tables" / "prot.tsv", "w")
    print("Target_ID", "Target", sep="\t", file=prots)

    glycans = set()
    for d, line in enumerate(data):
        if d == 0:
            iupacs = [x.strip() for x in line.split(",")[:-1]]
            for i, iupac in enumerate(iupacs):
                try:
                    glycan = Glycan(iupac, tree_only=True)
                    if glycan.get_tree().number_of_nodes() > max_mono:
                        continue
                    smiles = glycan.get_smiles()
                    if rdkit.Chem.MolFromSmiles(smiles):
                        name = f"Gly{(i + 1):05d}"
                        glycans.add(name)
                        print(name, smiles, sep="\t", file=lig)
                except:
                    continue
        else:
            parts = line.strip().split(",")
            seq = parts[-1]
            if seq not in lectin_dict or any(not x.isalpha() or not x.isupper() for x in seq) or len(seq) > max_aas:
                continue
            lectin = lectin_dict[seq]
            print(lectin)
            row = lectins[lectins["Input ID"] == lectin]
            if len(row) < 1 or (((row["Sequence identity"] < min_id).bool() or (row["Coverage"] < min_cov).bool()) and lectin + ".pdb" not in omega_lectins):
                continue
            prots.write(f"{lectin}\t{seq}\n")
            if lectin + ".pdb" in omega_lectins:
                shutil.copy(source_path / "omegafold" / f"{lectin}.pdb", target_path / "structures" / f"{lectin}.pdb")
            else:
                extract_chain(
                    lectin,
                    str(source_path / "models" / f"{lectin}_B99990001_refined.pdb"),
                    row["Model chain"].values[0],
                    str(target_path / "structures" / f"{lectin}.pdb")
                )

            counter = 0
            for i, value in enumerate(parts[:-1]):
                if len(value) > 0:
                    counter += 1
                    value = float(value)
                    glycan = f"Gly{(i + 1):05d}"
                    if glycan in glycans:
                        print(glycan, lectin, value, sep="\t", file=inter)
    inter.close()
    lig.close()
    prots.close()

                
if __name__ == "__main__":
    create(Path(sys.argv[1]), Path(sys.argv[2]))  # , float(sys.argv[2]), float(sys.argv[3]))
