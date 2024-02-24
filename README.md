Repository for reproduction of paper "Rethinking of Encoder-Based Warmstart Methods in Hyperparameter Optimization"

# Running experiments
## *Full dataset*
To reproduce experiments using full you need to have computer with linux opearing system and Python with version 3.10.12. First step is to install requirements using command
```
pip install -r requirements.txt
```

Second step is to run command which runs all steps and experiments:
```
./reproduction.sh
```
After that all results should be in directory `results/`. If you have permission denied error you need to run:
```
    chmod +x reproduction.sh
```

## *Toy dataset*
To reproduce experiments with toy dataset you need:
* Rename directory `data_toy` to `data` i. e. `mv data_toy data`. **NOTE:** you need to rename current `data` folder to something else 
* Rename file `config/uci_splits_toy.json` to `config/uci_splits.json` i. e. `mv config/uci_splits_toy.json config/uci_splits.json`. **NOTE:** you need to rename current `config/uci_splits.json` folder to something else.
* Run all steps from section *Full dataset*

# Repository contents
```
├── bin - directory with all executable files used to reproduce results
├── config - files with models and data configurations
├── data - directory with data from openml and uci
├── data_toy - minimal example needed to run experiments
├── engine - directory with files containing some logic behind our experiments
│   ├── cd_plot.py - generating critical distance plots
│   ├── dataset2vec - dataset2vec implementation
│   └── random_hpo - generating random hyperparameters using conditional spaces
├── models - pretrained models
├── README.md
├── reproduction.sh - scripts to reproduce experiments
├── requirements.txt - requirements to python environment needed for experiments run
```