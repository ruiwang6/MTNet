## MTNet: A Multi-task Learning Framework that Integrates Intra-task and Task-specific Dependencies for Traffic Forecasting

#### Shaokun Zhang, Rui Wang, Hongjun Tang, Kaizhong Zuo, Peng Jiang,  Peng Hu,  Wenjie Li,  Biao Jie, and Peize Zhao#, "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting", Proc. of 32nd ACM International Conference on Information and Knowledge Management (CIKM), 2023. (*Equal Contribution, #Corresponding Author)

#### Required Packages

```
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```

#### Training Commands

```bash
cd model/
python train.py -d <dataset> 
```

`<dataset>`:
- Manchester
- PEMS04
- PEMS08
