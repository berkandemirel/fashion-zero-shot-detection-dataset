This repository contains Fashion-ZSD dataset of the [Zero-Shot Object Detection by Hybrid Region Embedding](https://arxiv.org/pdf/1805.06157.pdf).

### Citing:

If you find this dataset useful in your research, please consider citing:

    @article{demirel2018zero,
    title={Zero-Shot Object Detection by Hybrid Region Embedding},
    author={Demirel, Berkan and Cinbis, Ramazan Gokberk and Ikizler-Cinbis, Nazli},
    journal={arXiv preprint arXiv:1805.06157},
    year={2018}
    }


### Details:

Fashion-ZSD is a toy dataset that we generate for evaluation of ZSD methods, based on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Fashion-MNIST originally consists of Zalando’s article images with associated labels. This dataset contains 70,000 grayscale images of size 28x28, and 10 classes. For Zero-shot detection task, we split the dataset into two disjoint sets; seven classes are used in training and three classes are used as the unseen test classes.

The dataset consists of images from four different scenarios. From left-to-right in image, (a)full objects only, (b)partial occlusions, (c)clutter regions included, and (d)a scene withboth partial occlusions and clutter regions. Ground truth object regions are shown with green and noise regions are shown in red boxes.

<p align="center">
<img src="data.png" align="center" width="500px" height="150px"/>
</p>


### Content:
This repo contains:
* Training, validation and test part of Fashion-ZSD dataset.
* Word embedding vectors of training and test classes.
* Evaluation script.

### Classes:

Training Classes:

    - tshirt
    - trouser
    - coat
    - sandal
    - shirt
    - sneaker
    - bag

Test Classes:

    - ankle_boot
    - dress
    - pullover
