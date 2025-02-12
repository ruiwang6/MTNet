# MTNet
This is the pytorch implementation of MTNet. MTNet is a multi-task learning framework for joint traffic flow and speed prediction, built on a Transformer-like Encoder-Decoder architecture. 

<img src="Figures/model.jpg" height="300"/>

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

Download PEMS04 and PEMS08 datasets provided by [STSGCN](https://github.com/Davidham3/STSGCN).

## Training Commands

```bash
cd scripts/
python train.py -d <dataset> -g <gpu_id>
```

`<dataset>`:
- PEMS04
- PEMS08
- Manchester

