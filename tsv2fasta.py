with open("/scratch/SCRATCH_SAS/roman/rindti/datasets/unilectin3d++/resources/tables/prot.tsv", "r") as data, \
        open("datasets/UniLectin3d++/Lectins.fasta", "w") as out:
    for line in data.readlines()[1:]:
        parts = line.strip().split("\t")
        print(f">{parts[0]}", parts[1], sep="\n", file=out)
