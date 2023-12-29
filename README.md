# SMTRI
A deep learning based approach for predicting Small Molecules Targeting miRNA-mRNA Interactions (SMTRI).

## Requirements
* Python 3.8
* tensorflow 2.11
* e3fp
* matplotlib
* numpy
* pandas
* rdkit
* scikit_learn
* scipy=
* xgboost

You can install the dependencies with the versioins specified in requirements.txt. 

## Dataset
The datasets used in training and testing are provided in ./data folder. We also provide valid RNA motif-SM associations with molecular properties in [Zenodo](https://zenodo.org/records/10439440). 

## Usage
You can train or test the trained SMTRI model:
```
$ python main.py
```

## Evaluation
You can reproduce the Case Studies and Performance Comparisons in [case_study](https://github.com/huan-xiao/SMTRI/blob/main/case_study.ipynb) and [performance_comparison](https://github.com/huan-xiao/SMTRI/blob/main/performance_comparison.ipynb).

