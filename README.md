# MTNet
This is the pytorch implementation of MTNet. MTNet is a multi-task learning framework for joint traffic flow and speed prediction, built on a Transformer-like Encoder-Decoder architecture. 

<img src="Figures/model.jpg" height="300"/>

## Model Comparison
The folder 'Model Comparison' contains the training logs of MTNet and Benchmarks.

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

We use the following datasets in our experiments:

- **Manchester**  
  Download: [Baidu NetDisk](https://pan.baidu.com/s/1YpZa1mYI3uOHl7lKKHjM_Q) (extraction code: `jtmc`)  
  Preprocess with: `prepareManchester.py`

- **PEMS04 and PEMS08**  
  Download from: [STSGCN](https://github.com/Davidham3/STSGCN)  
  Follow STSGCN instructions for preprocessing.


## Training Commands

```bash
cd model/
python train.py -d <dataset> -g <gpu_id>
```

`<dataset>`:
- PEMS04
- PEMS08
- Manchester

