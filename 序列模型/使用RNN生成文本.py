#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 使用RNN生成文本.py
# @Date  : 2019/4/16 0016
# @Contact : 1329778364@qq.com
# @Author: DeepMan
import time
import os
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import tensorflow

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(MyModel, self).__init__()
        self.units = units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, self.embedding_dim)

        if tf.test.is_gpu_available():
            self.GRU = tf.keras.layers.CuDNNGRU(self.units, return_sequences=True, recurrent_initializer="glorot_uniform",
                                                stateful=True)
        else:
            self.GRU = tf.keras.layers.GRU(self.units,
                                           return_sequences=True,
                                           recurrent_activation="sigmoid",
                                           recurrent_initializer="glorot_uniform",
                                           stateful=True)
        self.fc = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x):
        embedding = self.embedding(x)
        output = self.GRU(embedding)
        prediction = self.fc(output)
        return prediction


if __name__ == '__main__':

    # 下载数据集
    path_yo_file = tf.keras.utils.get_file(
        'shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(path_yo_file).read()
    # print('Length of text: {} characters'.format(len(text)))  # 文字长度：1115394个字符
    # print(text[:1000])
    vocab = sorted(set(text))  # 表示的单个字母 总共有65个独特的字符
    print('{} unique characters'.format(len(vocab)))

    # 向量化文本
    # 在训练之前，我们需要将字符串映射到数字表示值创建两个对照表：
    # 一个用于将字符映射到数字，另一个用于将数字映射到字符。
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])
    for char, _ in zip(char2idx, range(20)):
        print('{:6s} ---> {:4d}'.format(repr(char), char2idx[char]))
    print(
        '{} ---- characters mapped to int ---- > {}'.format(text[:13], text_as_int[:13]))

    seq_length = 100
    chunk = tf.data.Dataset.from_tensor_slices(
        text_as_int).batch(seq_length + 1, drop_remainder=True)

    for item in chunk.take(5):
        print(repr("".join(idx2char[item.numpy()])))

    # 接下来，利用此文本块创建输入文本和目标文本：
    def spilit_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    dataset = chunk.map(spilit_input_target)  # 利用map 函数来对数据集进行切分

    # 对于时间步0的输入，我们收到了映射到数字18的字符，并尝试预测映射到数字47的字符。
    # 在时间步1，执行相同的操作，但除了当前字符外，还要考虑上一步的信息。

    # for input_example, target_example in dataset.take(1):
    #     print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    #     print('Target data:', repr(''.join(idx2char[target_example.numpy()])))
    #
    # for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    #     print("Step {:4d}".format(i))
    #     print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    #     print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

    # 我们使用tf.data将文本分成块。但在将这些数据馈送到模型中之前，我们需要对数据进行重排，并将其打包成批。
    BATCH_SIZE = 10
    BUFFER_SIZE = 1000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE, drop_remainder=True)

    vocab_size = len(vocab)
    embedding_dim = 256
    units = 1024
    model = MyModel(vocab_size, embedding_dim, units)

    optimizer = tf.train.AdadeltaOptimizer()

    def loss_function(real, predict):
        return tf.losses.sparse_softmax_cross_entropy(
            labels=real, logits=predict)

    model.build(tf.TensorShape([BATCH_SIZE, seq_length]))
    model.summary()

    # 存储模型的地址
    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    EPOCHS = 5
    for epoch in range(EPOCHS):
        start = time.time()
        hidden = model.reset_states()

        for (batch, (inp, target)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predict = model(inp)
                loss = loss_function(target, predict)
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            if batch % 100 == 0:
                print("Epoch {} batch {} Loss {:.4f}".format(epoch+1, batch,loss))
        if (epoch+1) % 5 == 0:
            model.save_weights(checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


