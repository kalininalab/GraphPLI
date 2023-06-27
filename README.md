# Predicting Lectin-Glycan Interaction using Graph Neural Networks on Structural Data

### Master Thesis by Roman Joeres

---

## Quick Start for Thesis Evaluation

As you see this file, I assume, you already downloaded the code from the UdS filesharing system. If not, please refer 
to the "Cloning"-section below. The next steps are
1. Unpack the pre-build dataset files into a `prebuild` folder:
   ```shell
   tar -xvf prebuild
   ```
   The prebuild folder contains the datasets for structure-based prediction with GNNs, the ESM-1b and ESM-2 (3B) 
   embeddings of the Lectins, and the preprocessed pickle-files as they are produces by the snakemake pipeline.
2. Create and activate the environment from ``environment.yaml`` by running
   ```shell
   mamba env create -f environment.yaml -y
   conda activate siplgi
   ```
   This step assumes, mamba is installed in your system. If not, please refer to the 
   [Mamba installation guide](https://mamba.readthedocs.io/en/latest/installation.html#existing-conda-install)
3. Log-in with WandB (Weights & Biases is our tool to report metrics)
   To do so, please follow the [quickstart instructions](https://docs.wandb.ai/quickstart).
4. If you don't want to store the datafiles of the splits with all the graphs (up to 10G per dataset and split), change 
   line 61 in `src/training/data/datasets.py` from
   ```shell
   return os.path.join("data", exp_name, basefilename)
   ```
   to
   ```shell
   return os.path.join(<folder_path>, exp_name, basefilename)
   ```
5. Start re-training your favourite model by running
   ```shell
   python train.py config/train/<config>.yaml
   ```

## Cloning

Store the code locally by cloning this GitHub repository

```shell
git clone -b thesis_roman git@github.com:kalininalab/GraphPLI.git
```

## Data preparation

To prepare the data for training or inference, run

```shell
snakemake -j <no_cpus> --configfile config/snakemake/roman.yaml
```

in the root folder of the project. To run these snakemake pipelines, the datasets have to have a specific structure 
which is explained below.

**Because this dataset creation is a bit tricky, I will provide all build files in a tar.gz file**
The collected, structure-based dataset is also provided in that file. To get the code, I used to convert the 
LectinOracle dataset into a sequence-based one, please contact me, as this needs additional explanation.

## Training

To run the training just run

```shell
python train.py config/train/<config>.yaml
```

in the root folder of this project.


## Important Snakemake Config Parameters

The following explains the most important parameters to be twisted in the Snakemake pipeline.

| Field parents   | Field name  | Value      | Explanations                                               |
|-----------------|-------------|------------|------------------------------------------------------------|
|                 | source:     | <filepath> | Filepath to the resources folder of the dataset            |
| prots:features: | method:     | rinerator  | Use RINerator to compute protein graphs                    |
|                 |             | distance   | Use distance thresholds between Calpha atoms               |
|                 |             | esm        | Use the ESM embeddings for protein encoding                |
| drugs:          | node_feats: | IUPAC      | Use sweetnet for glycan encoding; therefore, store IUPACs  |
|                 |             | label      | Use atom labels as atom node featurizers                   |
|                 |             | onehot     | Use one-hot representation of atom types for featurization |
|                 |             | glycan     | Use extended one-hot reps. of atoms for featurization      |
| split_data:     | method:     | random     | Split interactions randomly                                |
|                 |             | target     | Split interactions along the lectins                       |
|                 |             | drug       | Split interactions along the drugs                         |

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
