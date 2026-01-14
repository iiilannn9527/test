# Continuous Subspace Optimization for Continual Learning

<div align="justify">
The implementation of our paper "Continuous Subspace Optimization for Continual Learning".
</div>

## Requisite

This code is implemented in PyTorch, and we perform the experiments under the following environment settings:

- python = 3.9
- pytorch = 2.5.1
- torchvision = 0.20.1
- timm = 0.6.7

The code may run under other versions of the environment, but I haven't tried.


## Dataset preparation
 * Create a folder `data/`
 * **CIFAR 100**: should automatically be downloaded.
 * **ImageNet-R**: download dataset from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar. After unzipping, place it into `data/` folder.
 * **DomainNet**: download from http://ai.bu.edu/M3SDA/, place it into `data/` folder.

## Training
All commands should be run under the project root directory. Currently, the code has been validated on 1 A6000 GPU (48G).

### CIFAR100:
#### For CoSO
```
python main.py --device your_device --config exps/coso_cifar.json 
```

### ImageNet-R (5 Tasks):
#### For CoSO
```
python main.py --device your_device --config exps/coso_inr5.json 
```

### ImageNet-R (10 Tasks):
#### For CoSO
```
python main.py --device your_device --config exps/coso_inr10.json 
```

### ImageNet-R (20 Tasks):
#### For CoSO
```
python main.py --device your_device --config exps/coso_inr20.json 
```

### DomainNet:
#### For CoSO
```
python main.py --device your_device --config configs/coso_domain.json
```