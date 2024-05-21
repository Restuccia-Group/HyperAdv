# HyperAdv
Official implementation of "HyperAdv: Dynamic Defense Against Adversarial Radio Frequency Machine Learning Systems"

## Dataset

Experiments are based on [RadioML2018.01A](https://www.deepsig.ai/datasets/) dataset. Please download it from official link and extract to a propriate directory.

## Environment

```
conda env create -n rfadv --file requirements.yml
conda activate rfadv
```

## Models

- DNN:
    - baseline: basic 1d cnn model 
    - hyper: our hypernet model

- Training Approach:
    - nt: natural training with cross entropy
    - at: pgd-based adverasarial training [Paper](https://arxiv.org/pdf/1706.06083)
    - trade: trades-style training [Paper](http://proceedings.mlr.press/v97/zhang19p/zhang19p-supp.pdf)

## Code Usage

python run_experiment.py -h
usage: run_experiment.py [-h] [-dnn] [-md] [-dp] [-cp] [-d] [-bs] [-t]

options:
  -h, --help           show this help message and exit
  -dnn , --DNN         specify the classifier (cnn, hyper) (default: cnn)
  -md , --Mode         specify the training mode (nt, at, trades) (default:
                       nt)
  -dp , --Data_path    specify the dataset directory (default: ../dataset/2018
                       .01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.
                       hdf5)
  -cp , --Ckpt_path    specify the checkpoint directory (default: resource/)
  -d , --device        specify the gpu device, -1 means cpu (default: 0)
  -bs , --Batch_size   specify the batchsize (default: 1024)
  -t, --Test_only      Specify training or testing (default: False)