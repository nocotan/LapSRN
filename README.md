## LaSRN
Implementation of CVPR2017 Paper: "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution"(http://vllab1.ucmerced.edu/~wlai24/LapSRN/) in Chainer.

### Usage

#### Training

```bash
$ python train.py -h
usage: train.py [-h] [--dataset DATASET] [--outdirname OUTDIRNAME]
                [--scale SCALE] [--batchsize BATCHSIZE] [--epoch EPOCH]
                [--steps_per_epoch STEPS_PER_EPOCH] [--model MODEL] [--gpu GPU]

LapSRN

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET
  --outdirname OUTDIRNAME
  --scale SCALE
  --batchsize BATCHSIZE
  --epoch EPOCH
  --steps_per_epoch STEPS_PER_EPOCH
  --model MODEL
  --gpu GPU
```

#### Inferring

```bash
$ python sr.py -h
usage: sr.py [-h] [--model MODEL] [--image IMAGE] [--scale SCALE] [--gpu GPU]

LapSRN Super-Resolution

optional arguments:
  -h, --help     show this help message and exit
  --model MODEL
  --image IMAGE
  --scale SCALE
  --gpu GPU
```
