#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:33:53 2018

@author: guy.hacohen
"""
import keras
import numpy as np
import keras.backend as K
import time


def _set_optimizer_learning_rate(optimizer, learning_rate):
    """Set optimizer lr across Keras / tf.keras API versions."""
    lr_attr = None
    if hasattr(optimizer, "learning_rate"):
        lr_attr = optimizer.learning_rate
    elif hasattr(optimizer, "lr"):
        lr_attr = optimizer.lr
    else:
        raise AttributeError("Optimizer has neither 'learning_rate' nor 'lr' attribute.")

    if hasattr(lr_attr, "assign"):
        lr_attr.assign(learning_rate)
        return

    # Older APIs exposed backend helpers for Variable-like objects.
    if hasattr(K, "set_value"):
        K.set_value(lr_attr, learning_rate)
        return

    # Fall back to direct attribute assignment for plain numeric values.
    if hasattr(optimizer, "learning_rate"):
        optimizer.learning_rate = learning_rate
    elif hasattr(optimizer, "lr"):
        optimizer.lr = learning_rate

def compile_model(model, initial_lr=1e-3, loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'], momentum=0.0):
    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(initial_lr, beta_1=0.9, beta_2=0.999,
                                          epsilon=None, decay=0.0,
                                          amsgrad=False)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(initial_lr, momentum=momentum)
    else:
        print("optimizer not supported")
        raise ValueError
    
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)


def basic_data_function(x_train, y_train, batch, history, model):#返回全量训练集，vanilla training
    return x_train, y_train

def basic_lr_scheduler(initial_lr, batch, history):#返回初始学习率，vanilla training
    return initial_lr


def generate_random_batch(x, y, batch_size):
    size_data = x.shape[0]
    cur_batch_idxs = np.random.choice(size_data, batch_size, replace=False)
    return x[cur_batch_idxs, :, :, :], y[cur_batch_idxs,:]


def train_model_batches(model, dataset, num_batches, batch_size=100,
                        test_each=50, batch_generator=generate_random_batch, initial_lr=1e-3,
                        lr_scheduler=basic_lr_scheduler, loss='categorical_crossentropy',
                        data_function=basic_data_function,#决定当前可用数据子集（课程学习就在这里生效）
                        verbose=False):
    #训练模型的主函数，核心是每个 batch 进行一次训练，并且每隔 test_each 个 batch 就在测试集上评估一次模型的性能。训练过程中会记录每个 batch 的训练损失和准确率，以及每次评估的测试损失和准确率，最后返回一个包含这些信息的 history 字典。
    
    x_train = dataset.x_train
    y_train = dataset.y_train_labels
    x_test = dataset.x_test
    y_test = dataset.y_test_labels

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "batch_num": [], "data_size": []}
    start_time = time.time()
    for batch in range(num_batches):
        cur_x, cur_y = data_function(x_train, y_train, batch, history, model)
        cur_lr = lr_scheduler(initial_lr, batch, history)
        _set_optimizer_learning_rate(model.optimizer, cur_lr)
        batch_x, batch_y = batch_generator(cur_x, cur_y, batch_size)
        cur_loss, cur_accuracy = model.train_on_batch(batch_x, batch_y)
        history["loss"].append(cur_loss)
        history["acc"].append(cur_accuracy)
        history["data_size"].append(cur_x.shape[0])
        if test_each is not None and (batch+1) % test_each == 0:#每隔 test_each 个 batch 就在测试集上评估一次模型的性能
            cur_val_loss, cur_val_acc = model.evaluate(x_test, y_test, verbose=0)
            history["val_loss"].append(cur_val_loss)
            history["val_acc"].append(cur_val_acc)
            history["batch_num"].append(batch)
            if verbose:
                print("val accuracy:", cur_val_acc)
        if verbose and (batch+1) % 5 == 0:#每隔 5 个 batch 就打印一次当前的训练状态，包括当前的 batch 数、学习率、当前使用的数据量、当前的训练损失，以及距离上次打印的时间间隔。
            print("batch: " + str(batch+1) + r"/" + str(num_batches))
            print("last lr used: " + str(cur_lr))
            print("data_size: " + str(cur_x.shape[0]))
            print("loss: " + str(cur_loss))
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

    return history
