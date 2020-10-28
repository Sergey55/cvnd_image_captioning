[//]: # (Image References)

[img_dataset_example]: ./images/coco-examples.jpg "Dataset examples"
[img_encoder]: ./images/encoder.png "Encoder"
[img_decoder]: ./images/decoder.png "Decoder"
[img_result]: ./images/result.png "Result"

# Image captioning

## Project Overview

A CNN-RNN model which extracts features from image using CNN encoder and converts them into text using RNN decoder.

## Data

This project uses [COCO](https://cocodataset.org) dataset. The Microsoft Common Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.

You can read more about the dataset on the [website](http://cocodataset.org/#home) or in the [research paper](https://arxiv.org/pdf/1405.0312.pdf).

![Dataset examples][img_dataset_example]

Model in this project trained using [2014 Train images](http://images.cocodataset.org/zips/train2014.zip) and [2014 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) files.

Both files are supposed to be downloaded and extracted to `coco` subdirectory. Having this done directory hierarchy should look like below:

```
project_home
|
├── coco
|   |
|   ├── annotations
|   |   ├── captions_train2014.json
|   |   ├── ...
|   |   └── person_keypoints_val2014.json
|   |
|   ├── images
|   |   ├── train2014
|   |   |   ├── COCO_train2014_000000000009.jpg
|   |   |   ├── ...
|   |   |   └── COCO_train2014_000000581921.jpg
```

## Architecture

In a nut shell the model in this project consists of encoder and decoder. Encoder extracts features from an image using pre-trained CNN model. Decoder converts extracted features into text.

Network architecture:

![alt Network Architecture](./images/encoder-decoder.png)

### Encoder:

![Encoder][img_encoder]

Encoder consistis of a pre-trained ResNet-50 network and a following Embedding layer. ResNet-50 is a convolutional neural network which is trained on more that a million images from ImageNet dataset. Embedding layer is used for co-embedding images and sentences together.

### Decoder:

![Decoder][img_decoder]

Decoder consists of a single LSTM layer followed by a fully-connected layer.

## Training

The model trained using `AI Platform Notebook` instance in Google Cloud Platform. This project requires `PyTorch Lightning` package to be pre-installed. Since the model implemented using PyTorch Lightning it can be trained using both GPU and TPU. Training process can be started using `train.sh` file. It automatically stops an instance at the end of the training.

## Evaluation

This repository contains weights for trained encoder and decoder. To get a caption the following command should be executed:

```
python evaluate.py --image_path './coco/images/test2014/COCO_test2014_000000000016.jpg'
```

Result looks like:

```
A man is holding a tennis racket on a court.
```

## Files

Repository contains next files and folders:

* [images](./images) - Images used in notebook and this readme.
* [models](./models) - Contains weights for trained models.
* [datamodule.py](./datamodule.py) - Pytorch Lightning datamodule which can be used for obtaining dataloader for train/test dataset.
* [dataset.py](./dataset.py) - CoCoDataset implementation.
* [environment.yml](./environment.yml) - Conda environment info.
* [evaluate.py](./evaluate.py) - Evaluation script.
* [model.py](./model.py) - Model implementation.
* [pkgs.txt](./pkgs.txt) - List of required packages.
* [README.md](./README.md) - This file.
* [Test.ipynb](./Test.ipynb) - Notebook, containing evaluation logic.
* [train.py](./train.py) - Script for training network
* [train.sh](./train.sh) - Shell script for starting trainig process from terminal
* [util.py](./util.py) - Utility functions.
* [vocab.pkl](./vocab.pkl) - Vocabulary
* [vocabulary.py](./vocabulary.py) - Vocabulary class which constructs vocabulary from the captions in the training dataset

## Resources

* [Dataset details](https://cocodataset.org)
* [Research paper](https://arxiv.org/pdf/1411.4555.pdf)

## Results

As a result I obtained a neural network which can generate text description of image file.

![Result][img_result]
