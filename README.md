# VTH-VPG
========
This is the official implementation of [Visual-Textual Harmony: Multi-Dimensional Congruity for Weakly Supervised Video Paragraph Grounding].



## Dataset
Please download the visual features from the official website of ActivityNet: [Official C3D Feature](http://activity-net.org/download.html). And you can download preprocessed annotation files [here](https://github.com/baopj/DenseEventsGrounding/blob/main/DepNet_ANet_Release/files_/acnet_annot.zip). 

## Prerequisites
- python 3.5
- pytorch 1.4.0
- torchtext
- easydict
- terminaltables

## Training
Use the following commands for training:
```
cd moment_localization && export CUDA_VISIBLE_DEVICES=0
python dense_train.py --verbose --cfg ../experiments/dense_activitynet/acnet.yaml
```
You may get better results than that reported in our paper thanks to the code updates.




## Acknowledgement
Part of our code is based on the previous works https://github.com/microsoft/VideoX/tree/master/2D-TAN and https://github.com/baopj/DenseEventsGrounding.
