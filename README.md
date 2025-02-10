# MTNet
This is the pytorch implementation of MTNet.

## Required Packages

```
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```

## Datasets
- Manchester
- PEMS04
- PEMS08

If you need the original datasets, please refer to [STSGCN](https://github.com/Davidham3/STSGCN) (including PEMS04, and PEMS08)

## Training Commands

```bash
cd model/
python train.py -d <dataset> 
```

