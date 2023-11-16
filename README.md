# GraphPLI - Graph-based Prediction of Protein-Ligand Interactions


## Download and Installation

Store the code locally by cloning this GitHub repository

```shell
git clone git@github.com:kalininalab/SIPILG.git
```

The entire code can be executed in the same conda-environment. To install the conda environment, move into the project 
folder and install the conda environment from the `environment.yaml` file. Finally, activate it to run the code.

```shell
cd SIPILG
conda env create -f environment.yaml
conda activate sipilg
```

## Data Preparation

To prepare the data for training or inference, run

```shell
snakemake -j <no_cpus> --configfile config/snakemake/unilectin.yaml
```

in the root folder of the project. To run these snakemake pipelines, the datasets have to have a specific structure 
which is explained below.

## Training

To run the training, just run

```shell
python -m src.train config/train/default.yaml
```

in the root folder of this project.

All training configs for the Master's thesis are included in the `config/thesis_train` folder.


## Dataset Structure

The datasets used for training and testing of the models have to have a specific structure to be preprocessed by the 
snakemake pipeline. 

```
dataset/
├── structures/
│       └── <pdb structures>
└── tables/
        ├── inter.tsv
        ├── lig.tsv
        └── prot.tsv
```

The `lig.tsv` file needs a header line and has to contain a column `Drug_ID` and a column `SMILES` mapping `Drug_ID`s 
to their SMILES strings. Similarly, `prot.tsv` needs a header line and two columns, namely `Target_ID` and `Target`. 
This file maps `Target_ID`s to their FASTA sequence. Furthermore, the Target_IDs have to match the filenames in the 
`structures` folder. Finally, the inter.tsv file contains the actual pairwise interactions of `Drug_ID`s and 
`Target_ID`s in a TSV format. Again a header line is required as well as a third column, `Y`, storing a proxy for the 
binding affinity. This can either be a regression value or classification labels.
