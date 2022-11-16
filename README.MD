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

## Dataset download and transfer format 

1. Dataset download

* **Cityscape and FoggyCityscape:**  Download Cityscapes dataset and Foggy-Cityscapes dataset [Cityscape](https://www.cityscapes-dataset.com/), see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data), than tranfor format from VOC to YOLO.
* **RTTS:** Please follow the [instruction](https://sites.google.com/site/boyilics/website-builder/reside) to prepare dataset.
* **RainCityscape:** Please follow the [instruction](https://www.photoshopessentials.com/photo-effects/photoshop-weather-effects-rain/) to prepare dataset.

2. Organize the dataset as following:

```shell
# the dateset format follow YOLO data format.(eg. normal to foggy)
R_YOLO/
└── datasets/
    └── Normal_to_Foggy/
        └── images/
            ├── Normal_train
            ├── Foggy_train
            └── Foggy_test
        └── labels/
            ├── Normal_val
            └── Foggy_val
```

## Training

### Step 1: Training the QTNet and Preparing the data of translation image


```shell
# train the madel of normal_to_advere
python train_QTNet.py -- mode normal_to_advere
# train the madel of normal_to_advere
python train_QTNet.py -- mode advere_to_normal
# generate translation image of normal
python infer_QTNet.py -- mode normal_to_advere
# generate translation image of advere
python infer_QTNet.py -- mode advere_to_normal
```


### Step 2: Training the FCNet


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
