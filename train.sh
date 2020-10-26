#!/bin/bash
python train.py --max_epochs 10 --gpus=1

gcloud compute instances stop $HOSTNAME