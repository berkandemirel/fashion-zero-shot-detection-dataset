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

Fashion-ZSD is a toy dataset that we generate for evaluation of ZSD methods, based on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Fashion-MNIST originally consists of Zalando’s article images with associated labels. This dataset contains 70,000 grayscale images of size 28x28, and 10 classes. For Zero-shot detection task, we split the dataset into two disjoint sets; seven classes are used in training and three classes are used as the unseen test classes. We generate multi-object images such that there are three different objects in each image. Randomly cropped objects are utilized to create clutter regions. As shown in Figure, we consider four scenarios: no noise or occlusion, scenes with partial occlusions, those with clutter, and, finally scenes with both partial occlusions and clutter regions. 8000 images of the resulting 16333 training images are held out for validation purposes. As a result, we obtain the Fashion-ZSD dataset with 8333 training, 8000 validation and 6999 test images. In the image below, ground truth object regions are shown with green and noise regions are shown in red boxes.

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
