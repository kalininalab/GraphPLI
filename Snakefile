from snakemake.utils import validate

from src.data_prep.scripts.snakemake_helper import SnakemakeHelper


configfile: "config/snakemake/default.yaml"


validate(config, schema="src/data_prep/schemas/config.schema.yaml")
sh = SnakemakeHelper(config, 8)


include: "src/data_prep/rules/sanity_checks.smk"
include: "src/data_prep/rules/prots.smk"
include: "src/data_prep/rules/drugs.smk"
include: "src/data_prep/rules/data.smk"


if config["only_prots"] == "both":
    output = [pretrain_prot_data, final_output]
elif config["only_prots"]:
    output = [pretrain_prot_data]
else:
    output = [final_output]


rule all:
    input:
        output,
