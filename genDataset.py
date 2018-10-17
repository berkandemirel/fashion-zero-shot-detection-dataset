#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Explanations will be provided later.
# Written by berkan
# Contact: demirelberkan@gmail.com
# --------------------------------------------------------

from __future__ import print_function
import  argparse
import os
import gzip
import numpy as np
import scipy.misc

#class names
labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

# input image dimensions
imgRows, imgCols = 28, 28

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", type=str, help="path to folder containing images")
parser.add_argument("--output_dir", type=str, help="path to folder for result images")
parser.add_argument("--img_comp", type=int, default=1, help="maximum number of classes that can be located in a image")
parser.add_argument("--image_scale", type=int, default=4, help="scale coefficient of the output image")
parser.add_argument("--train_classes", type=str, help="list of training classes")
parser.add_argument("--test_classes", type=str, help="list of test classes")
parser.add_argument("--predicate_matrix", type=str, help="class-attribute association table")

params = parser.parse_args()

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def readClassList(path):

    f = open(path, "r")
    results = []

    for line in f:
        results.append(int(line.split('\n')[0]))

    return results

def loadFashionDataset(path, kind='train'):
    #Adopted from: https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

def run():

    datasetComplexity = params.img_comp
    imageScale = params.image_scale


    print('Fashion dataset is loaded.')
    X_train, y_train = loadFashionDataset(params.input_dir, kind='train')
    X_test, y_test = loadFashionDataset(params.input_dir, kind='t10k')

    images = np.concatenate((X_train, X_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)
    print('Fashion dataset is combined.')
    print('Number of images:', len(images))

    trainingClasses = readClassList(params.train_classes)
    testClasses = readClassList(params.test_classes)
    print('Number of training classes:', len(trainingClasses))
    print('Number of test classes:', len(testClasses))


    trainIndices = np.nonzero(np.in1d(labels, trainingClasses)+0)
    testIndices = np.nonzero(np.in1d(labels, testClasses)+0)

    xData = images.reshape(images.shape[0], imgRows, imgCols)
    xData = xData.astype('float32')
    trainData = xData[trainIndices,:,:][0]
    testData = xData[testIndices,:,:][0]

    templateImage = np.zeros((imgRows*2, imgCols*imageScale))

    margin = 12 #margin value between objects

    currObj = 0
    counter = 0
    trainFile = []
    for i in range(len(trainData)):

        if i%datasetComplexity == 0:
            if i != 0:
                scipy.misc.imsave('fashionDataset/train/train_' + str(counter) + '.jpg', currImage)
                trainFile.close()
                counter = counter +1
            trainFile = open("fashionDataset/train_labels/train_" + str(counter) + ".txt", "w")
            currImage = templateImage
            currImage[margin:imgCols+margin, margin:imgRows+margin] = trainData[i,:,:]

            b = (float(margin), float(imgCols+margin), float(margin), float(imgRows+margin))
            bb = convert((imgRows*imageScale, imgCols*2), b)
            trainFile.write(str(labels[trainIndices[0][i]]) + " " + " ".join([str(a) for a in bb]) + '\n')
            currObj = 1
        else:
            currImage[margin:imgCols+margin,
                    margin + imgRows * currObj:margin + imgRows * (currObj + 1)] = trainData[i,:,:]
            b = (float(margin + imgRows * currObj), float(margin + imgRows * (currObj + 1)),
                            float(margin), float(imgCols+margin))
            bb = convert((imgRows*imageScale, imgCols*2), b)
            trainFile.write(str(labels[trainIndices[0][i]]) + " " + " ".join([str(a) for a in bb]) + '\n')
            currObj = currObj + 1

    currObj = 0
    counter = 0
    testFile = []
    for i in range(len(testData)):

        if i%datasetComplexity == 0:
            if i != 0:
                scipy.misc.imsave('fashionDataset/test/test_' + str(counter) + '.jpg', currImage)
                testFile.close()
                counter = counter +1
            testFile = open("fashionDataset/test_labels/test_" + str(counter) + ".txt", "w")
            currImage = templateImage
            currImage[margin:imgCols+margin, margin:imgRows+margin] = testData[i,:,:]

            b = (float(margin), float(imgCols+margin), float(margin), float(imgRows+margin))
            bb = convert((imgRows*imageScale, imgCols*2), b)
            testFile.write(str(labels[testIndices[0][i]]) + " " + " ".join([str(a) for a in bb]) + '\n')
            currObj = 1
        else:
            currImage[margin:imgCols+margin,
                    margin + imgRows * currObj:margin + imgRows * (currObj + 1)] = testData[i,:,:]
            b = (float(margin + imgRows * currObj), float(margin + imgRows * (currObj + 1)),
                            float(margin), float(imgCols+margin))
            bb = convert((imgRows*imageScale, imgCols*2), b)
            testFile.write(str(labels[testIndices[0][i]]) + " " + " ".join([str(a) for a in bb]) + '\n')
            currObj = currObj + 1

run()
