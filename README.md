# DataMiningTrail
This is a Pytorch implementation of the paper: Molecular Property Prediction with Hierarchical Message Passing Neural Networks
## Overview of the Framework
HMPNN is a novel molecular representation learning framework, consisting of Atom-MPNN and Substructure-MPNN and adopts a local augmentation strategy to improve the performance of the downstream molecular property prediction tasks.

<p align="center">
<img  src="fig/KPGT.png"> 
</p>

## **Setup Environment**

Setup the required environment using `hmpnn.yaml` with Anaconda. While in the project directory run:

    conda env create -f hmpnn.yaml

Activate the environment

    conda activate HeinzTorch
## Training
To train a model from scratch, run:

`python train_evaluate.py --dataset <datasetname> --num_folds 10 --device cuda:0 --epoch 150`

To train a model with optimized hyperparameters,run:
```linux
bash ./scripts/<datasetname>.sh
```

## The Results

- The classification datasets.

Model      | BBBP | SIDER | ClinTox | BACE | Tox21 
----       | ---|----       |----       |----        |----            
HMPNN|  0.935(0.017) | 0.667(0.071) | 0.927(0.029) | 0.907(0.029) | 0.837(0.011) 

- The regression datasets.

Model | FreeSolv | ESOL | Lipo 
----- | ---- | ---- | ---- 
HMPNN  | 0.986(0.171) | 0.642(0.068) |  0.582(0.014) 
 
