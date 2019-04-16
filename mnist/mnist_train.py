#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : mnist_train.py
# @Date  : 2019/4/16 0016
# @Contact : 1329778364@qq.com
# @Author: DeepMan

"""
原先安装的是 tensorflow==1.12.0
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras import *


class MyModel(Model):  # 构建模型
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        out = self.d2(x)
        return out


mymodel = MyModel()

loss_object = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam()

train_loss = metrics.Mean(name="train_loss")
train_acc = metrics.SparseCategoricalAccuracy(name="train_acc")

test_loss = metrics.Mean(name="test_loss")
test_acc = metrics.SparseCategoricalAccuracy(name="test_acc")


@tf.function
def train_step(image, label):
    with tf.GradientTape() as tape:
        prediction = mymodel(image)
        loss = loss_object(label, prediction)
    gradients = tape.gradient(loss, mymodel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mymodel.trainable_variables))

    train_loss(loss)
    train_acc(label, prediction)

@tf.function
def test_step(image, label):
    prediction = mymodel(image)
    t_loss = loss_object(label, prediction)

    test_acc(label, prediction)
    test_loss(t_loss)

def convert_types(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label


if __name__ == '__main__':

    dataset, info = tfds.load(name="mnist", with_info=True,as_supervised=True)
    mnist_train, mnist_test = dataset['train'], dataset['test']

    mnist_train = mnist_train.map(convert_types).shuffle(100).batch(10)
    mnist_test = mnist_test.map(convert_types).batch(10)

    EPOCHs = 5
    for epoch in range(EPOCHs):
        for image, label in mnist_train:
            train_step(image, label)

        for test_image, test_label in mnist_test:
            test_step(test_image, test_label)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print()


