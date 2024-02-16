#!/bin/bash

export PYTHONPATH=`pwd`

python bin/preprocess_mimic.py
python bin/generate_mini_holdout.py

python bin/generate_d2v_visualisation_tasks.py
python bin/train_liltab.py
python bin/train_d2v.py

cp `find results/liltab_encoder_training -name model-epoch\*.ckpt -print` models/liltab.ckpt
cp `find results/dataset2vec -name epoch\*.ckpt -print` models/d2v.ckpt

python bin/generate_plots_uci.py
python bin/generate_plots_mimic.py

python bin/generate_hpo_random_base_uci.py
python bin/generate_hpo_random_base_mimic.py

python bin/generate_hp_base_index_uci.py
python bin/generate_hp_base_index_mimic.py

python bin/warmstart.py \
    --experiment=mimic \
    --index-path=results/hyperparameters_index_mimic/index.parquet \
    --ranks-path=results/hyperparameters_index_mimic/ranks.parquet \
    --datasets-path=data/mimic/mini_holdout \
    --output-db-name=mimic \
    --optimisation-iterations=30 \
    --warmstart-trials-count=10 

for i in $(seq 1 20); do
    echo $i;
    python bin/warmstart.py \
        --experiment=uci \
        --index-path=results/hyperparameters_index_uci/index.parquet \
        --ranks-path=results/hyperparameters_index_uci/ranks.parquet \
        --datasets-path=data/uci/splitted/test \
        --output-db-name=uci_$i \
        --output-path=results/warmstart_dbs/uci \
        --optimisation-iterations=30 \
        --warmstart-trials-count=10 \
        --fix-seed
done