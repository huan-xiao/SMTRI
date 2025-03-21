# SMTRI
A deep learning based approach for predicting Small Molecules Targeting miRNA-mRNA Interactions (SMTRI).
![title](Title.jpg)

## Requirements
* Python 3.8
* tensorflow 2.11
* e3fp
* matplotlib
* numpy
* pandas
* rdkit
* scikit_learn
* scipy
* xgboost

You can install the dependencies with the versioins specified in requirements.txt. 

## Dataset
The datasets used in training and testing are provided in ./data folder. We also provide all models, predictions, and additional files in [Zenodo](https://zenodo.org/records/10897739).

## Usage
You can train the SMTRI model:
```
$ python main.py
```

We also prepared the docker image [smtri](https://hub.docker.com/u/hilaryhsiao), which can be pulled from Docker Hub and run locally:
```
$ docker image pull hilaryhsiao/smtri:2.0
$ docker run hilaryhsiao/smtri:2.0 <args>
```
The parameters in "args" above are miRNA sequence, mRNA sequence, and SM SMILES. For example:
1. miRNA, a sequence made up of 'A,C,G,U'.
```
"AACGCACACUGCGCUGCUU"
```
2. mRNA, a sequence made up of 'A,C,G,U'.
```
"GGGGGGGCCCCCCCCCCCAGACCCACUGUGCGUUUUUUUUUU"
```
3. SMs, one SMILES per line, the length can be arbitary.
```
"C[n+]1ccc(c2c1cccc2)Cc3[n+](c4ccccc4s3)CC(=O)NCCOC
CC[C@H](C)[C@H](NC(=O)[C@H](CC(=O)OCc1ccccc1)NC(=O)[C@H](N)CC(C)C)C(=O)N[C@@H](C)C(N)=O
CC(C)C[C@@H]1NC(=O)[C@@H]2CCC[C@H]2NC(=O)[C@H](C)NC(=O)[C@@H]2C[C@@H](NC(=O)[C@@H]3CCCN4CCCC[C@H]34)CN2C(=O)[C@H](Cc2ccccc2)NC(=O)[C@H]([C@@H](C)O)NC1=O
CN1CCNC(=O)C[C@@H]2CN(S(=O)(=O)c3ccc4c(c3)OCCCO4)C[C@@H]2n2cc(nn2)Cn2cccc(c2=O)C1=O
O=C1OC2=CC(=CC(OC3OC(CO)C(O)C(O)C3O)=C2C4=C1CCC4)C
O=C1C=C(OC=2C1=C(O)C=C(O)C2C(C=3C=CC=CC3)CC(=O)OC)C=4C=CC=CC4"
```
So, the command for running the docker image can be like: (selecting targeting SMs for miRNA-mRNA interactions from a list of 6 SMs)
```
docker run hilaryhsiao/smtri:2.0 "AACGCACACUGCGCUGCUU" "GGGGGGGCCCCCCCCCCCAGACCCACUGUGCGUUUUUUUUUU" "C[n+]1ccc(c2c1cccc2)Cc3[n+](c4ccccc4s3)CC(=O)NCCOC
CC[C@H](C)[C@H](NC(=O)[C@H](CC(=O)OCc1ccccc1)NC(=O)[C@H](N)CC(C)C)C(=O)N[C@@H](C)C(N)=O
CC(C)C[C@@H]1NC(=O)[C@@H]2CCC[C@H]2NC(=O)[C@H](C)NC(=O)[C@@H]2C[C@@H](NC(=O)[C@@H]3CCCN4CCCC[C@H]34)CN2C(=O)[C@H](Cc2ccccc2)NC(=O)[C@H]([C@@H](C)O)NC1=O
CN1CCNC(=O)C[C@@H]2CN(S(=O)(=O)c3ccc4c(c3)OCCCO4)C[C@@H]2n2cc(nn2)Cn2cccc(c2=O)C1=O
O=C1OC2=CC(=CC(OC3OC(CO)C(O)C(O)C3O)=C2C4=C1CCC4)C
O=C1C=C(OC=2C1=C(O)C=C(O)C2C(C=3C=CC=CC3)CC(=O)OC)C=4C=CC=CC4"
```

The results are presented in three columns [InchiKey, SMILES, Probability]. Those not listed in the results are deemed not targeting. For example:
```
['WEFGJTSJMBBARG-UHFFFAOYSA-O', 'C[n+]1ccc(c2c1cccc2)Cc3[n+](c4ccccc4s3)CC(=O)NCCOC', 0.9706]
```

## Evaluation
You can reproduce the Case Studies and Performance Comparisons in [case_study](https://github.com/huan-xiao/SMTRI/blob/main/case_study.ipynb) and [SOTA_comparison](https://github.com/huan-xiao/SMTRI/blob/main/SOTA_comparison.ipynb).

## Citation
Please cite the following [paper](https://www.sciencedirect.com/science/article/pii/S2162253124001902?via%3Dihub):  
Xiao, H., Zhang, Y., Yang, X., Yu, S., Chen, Z., Lu, A., Zhang, Z., Zhang, G., and Zhang, B.-T. (2024). SMTRI: A deep learning-based web service for predicting small molecules that target miRNA-mRNA interactions. Molecular Therapy - Nucleic Acids. 35(3).

The user-friendly website of SMTRI is on: http://www.smtri.net/
