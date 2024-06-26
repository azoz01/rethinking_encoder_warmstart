#!/bin/bash
# Commented lines run scripts depending on MIMIC dataset
# which is not in public domain

export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=""

# python bin/preprocess_mimic.py
# python bin/generate_mini_holdout.py

python bin/generate_uci_visualisation_tasks.py
python bin/train_liltab.py
python bin/train_d2v.py

mkdir models

cp `find results/liltab_encoder_training -name model-epoch\*.ckpt -print` models/liltab.ckpt
cp `find results/dataset2vec -name epoch\*.ckpt -print` models/d2v.ckpt

python bin/generate_plots_uci.py
# python bin/generate_plots_mimic.py

python bin/calculate_representation_metrics_uci.py
# python bin/calculate_representation_metrics_mimic.py

python bin/generate_hpo_random_base_uci.py
# python bin/generate_hpo_random_base_mimic.py

python bin/generate_hp_base_index_uci.py
# python bin/generate_hp_base_index_mimic.py

for i in $(seq 0 4); do
    echo $i;
    python bin/warmstart.py \
        --experiment=uci \
        --index-path=results/hyperparameters_index_uci \
        --datasets-path=data/uci/splitted \
        --output-db-name=uci_fold_$i \
        --output-path=results/warmstart_dbs/uci \
        --optimisation-iterations=30 \
        --warmstart-trials-count=10 \
        --fold=$i
done

# for i in $(seq 0 3); do
#     echo $i;
#     python bin/warmstart.py \
#         --experiment=mimic \
#         --index-path=results/hyperparameters_index_mimic \
#         --datasets-path=data/mimic/mini_holdout \
#         --output-db-name=mimic_fold_$i \
#         --output-path=results/warmstart_dbs/mimic \
#         --optimisation-iterations=30 \
#         --warmstart-trials-count=10 \
#         --fold=$i
# done

mkdir results/warmstart_plots

python bin/generate_warmstart_plots_uci.py
# python bin/generate_warmstart_plots_mimic.py