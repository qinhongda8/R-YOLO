# R-YOLO: A Robust Object Detector in Adverse Weather

Here, we provide detailed instructions to help replay our experiments. To make it easy to reproduce our research results.

We give an overview of the environment installation and adaptation experiment.

## Environment installation

```shell
# create conda env
conda create -n R_YOLO python=3.8
# activate the enviorment
conda activate R_YOLO
# install 
pip install -r requirements.txt
```

## Dataset download

1. Download Cityscapes dataset and Foggy-Cityscapes dataset   [Cityscape](https://www.cityscapes-dataset.com/)

2. Organize the dataset as following:

```shell
R_YOLO/
└── datasets/
    └── Normal_to_Foggy/
        └── images/
            ├── Normal_and_Normaltrans
            └── Foggy_and_Foggytrans
        └── labels/
            └── Normal_and_Normaltrans
```

## Training

1. QTNet 

- Prepare the data of translation image

```shell
# train the madel of normal_to_advere
python train_QTNet.py -- made normal_to_advere
# train the madel of normal_to_advere
python train_QTNet.py -- made advere_to_normal
# generate translation image of normal
python infer_QTNet.py -- made normal_to_advere
# generate translation image of advere
python infer_QTNet.py -- made advere_to_normal
```
2. R-YOLO

```shell

python train_FCNet.py

```

## Evaluation

```shell

python train_FCNet.py -- weight ./runs/weights/best.py

```

## Models
### Pre-trained Models

YOLOv5:

|  Model   |      data        |  AP   |                                       Model Weights                                        |
| :-----:  | :-------------:  | :---: | :----------------------------------------------------------------------------------------: |
| YOLOv5-M |    pretrain      | 27.84 | [link](https://drive.google.com/file/d/1IJQeRP9wHPU0J27YTea-y3lIW96bMAUu/view?usp=sharing) |
| YOLOv5-M | normal+foggy(un) | 24.16 | [link](https://drive.google.com/file/d/12q-LB4iDvgXGW50Q-bYOahpalUvO3SIa/view?usp=sharing) |
