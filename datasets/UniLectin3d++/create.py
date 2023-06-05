import os
import re
import shutil
from os.path import join as pj
import sys
import urllib.request
from urllib.error import HTTPError

import glyles
import pandas as pd

threetoone = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}


def pdb_to_sequence(pdb_filename: str) -> str:
    """Extract sequence from PDB file and return it as a string."""
    sequence = ""
    with open(pdb_filename, "r") as file:
        for line in file.readlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                sequence += threetoone.get(line[17:20].strip(), "A")
    return sequence


def extract_pdb_chain(pdb_file, chain_id, output_file):
    valid = False
    if isinstance(chain_id, str):
        chain_ids = set(chain_id.split(","))
    else:
        chain_ids = {str(chain_id)}
    with open(pdb_file, "r") as pdb, open(output_file, "w") as chain:
        for line in pdb:
            if line.startswith("ATOM") and line[21].strip() in chain_ids:
                valid = True
                print(line, file=chain)
    if not valid:
        os.remove(output_file)
    return valid


def create(target_path, glycowork_path):
    bnd_strct_folder = os.path.join(target_path, "resources", "bound_structs")
    strct_folder = os.path.join(target_path, "resources", "structures")
    table_folder = os.path.join(target_path, "resources", "tables")
    os.makedirs(bnd_strct_folder, exist_ok=True)
    os.makedirs(strct_folder, exist_ok=True)
    os.makedirs(table_folder, exist_ok=True)

    df = pd.read_csv("unilectin3D.csv")
    lig_map = dict()
    prot_map = dict()
    inter_count = 0
    with open(pj(table_folder, "inter.tsv"), "w") as inter:
        print("Drug_ID", "Target_ID", "Y", sep="\t", file=inter)
        for index, row in df.iterrows():
            if row["iupac"] != row["iupac"]:
                continue
            for iupac in re.split(',|;', row["iupac"]):
                iupac = iupac.replace("DmanHep", "DManHep")
                smiles = glyles.convert(iupac)[0][1]
                if smiles == "":
                    continue
                if smiles not in lig_map:
                    lig_map[smiles] = f"Gly{len(lig_map) + 1:05d}"
                lig_id = lig_map[smiles]
                try:
                    if "+0" in row["pdb"]:
                        row["pdb"] = row["pdb"].replace("+0", "")
                    if not os.path.exists(pj(bnd_strct_folder, f"{row['pdb']}.pdb")):
                        urllib.request.urlretrieve(
                            f"http://files.rcsb.org/download/{row['pdb']}.pdb",
                            pj(bnd_strct_folder, f"{row['pdb']}.pdb"),
                        )
                    if os.path.exists(pj(bnd_strct_folder, f"{row['pdb']}.pdb")) and not os.path.exists(pj(strct_folder, f"{row['pdb']}.pdb")):
                        if not extract_pdb_chain(
                                pj(bnd_strct_folder, f"{row['pdb']}.pdb"),
                                row["chain"],
                                pj(strct_folder, f"{row['pdb']}.pdb"),
                        ):
                            # os.remove(pj(bnd_strct_folder, f"{row['pdb']}.pdb"))
                            continue
                    prot_map[row["pdb"]] = pdb_to_sequence(pj(strct_folder, f"{row['pdb']}.pdb"))
                    print(lig_id, row["pdb"], 1, sep="\t", file=inter)
                    inter_count += 1
                except HTTPError:
                    print("HTTP-Error with:", row["pdb"])

        with open(pj(glycowork_path, "tables", "inter.tsv"), "r") as data:
            gw_inter = sorted((line.strip().split("\t") for line in data.readlines()[1:]), key=lambda x: x[2])
        with open(pj(glycowork_path, "tables", "lig.tsv"), "r") as data:
            gw_ligands = dict(line.strip().split("\t")[:2] for line in data.readlines()[1:])
        for drug_id, pdb, _ in gw_inter[:inter_count]:
            shutil.copyfile(pj(glycowork_path, "structures", f"{pdb}.pdb"), pj(strct_folder, f"{pdb}.pdb"))
            prot_map[pdb] = pdb_to_sequence(pj(strct_folder, f"{pdb}.pdb"))
            smiles = gw_ligands[drug_id]
            if smiles not in lig_map:
                lig_map[smiles] = f"Gly{len(lig_map) + 1:05d}"
            print(lig_map[smiles], pdb, 0, file=inter, sep="\t")

    with open(pj(table_folder, "lig.tsv"), "w") as out:
        print("Drug_ID", "Drug", sep="\t", file=out)
        for smiles, lig_id in lig_map.items():
            print(lig_id, smiles, sep="\t", file=out)

    with open(pj(table_folder, "prot.tsv"), "w") as out:
        print("Target_ID", "Target", sep="\t", file=out)
        for pdb, seq in prot_map.items():
            print(pdb, seq, sep="\t", file=out)


if __name__ == '__main__':
    create(*sys.argv[1:3])
