# Code Execution Manual for FACC

## Paper Information

> **Few-Shot Continual Learning for Activity Recognition in Classroom Surveillance Images** <br>Yilei Qian, Kanglei Geng, Kailong Chen, Shaoxu Cheng, Linfeng Xu, Fanman Meng, Qingbo Wu, Hongliang Li
> The 2025 3rd International Conference on Intelligent Perception and Computer Vision

## Dependencies and Installation

**Depedencies**

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Pytorch 2.0 or later. See [Pytorch]( https://pytorch.org) for install instructions.
- Linux (Ubuntu 18.04.3)

**Installation**

First, you can clone this repo using the command:

```shell 
git clone https://github.com/mon1ystar/FACC.git
```

Then, you can create a virtual environment using conda, as follows:

```shell
conda env create -f environment.yaml
conda activate ckl
```

## Data preparation

We provide source about the datasets we use in our experiment as below:

| Dataset | Dataset                                                   |
| ------- | --------------------------------------------------------- |
| ARIC    | [ARIC](https://ivipclab.github.io/publication_ARIC/ARIC/) |

## Training

Run the following command to train the model sequentially:

```shell
bash ./train_classroom.sh
```

After training, you can get model checkpoints in the folder `./checkpoint`, you can modify the train_mode in 'resnet_a1' and 'resnet_a2' to choose the task setting.

## Evaluation

After completing training, the model's performance can be tested using the following command:

```shell
bash ./evaluation.sh
```

The result will be saved in the folder `./checkpoint`.