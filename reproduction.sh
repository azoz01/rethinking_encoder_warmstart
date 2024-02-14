#!/bin/bash

export PYTHONPATH=`pwd`

# python bin/preprocess_mimic.py
# python bin/generate_mini_holdout.py

python bin/generate_d2v_visualisation_tasks.py
python bin/train_liltab.py
python bin/train_d2v.py

cp `find results/liltab_encoder_training -name model-epoch\*.ckpt -print` models/liltab.ckpt
cp `find results/dataset2vec -name epoch\*.ckpt -print` models/d2v.ckpt

python bin/generate_plots_uci.py
python bin/generate_plots_mimic.py
python bin/generate_hpo_random_base.py