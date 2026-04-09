# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

# Functions and classes for loading and using the Inception model.
import models.inception
#import stl10
from models.inception import transfer_values_cache

from sklearn import svm
import numpy as np
import pickle
import classic_nets_imagenet


def _is_valid_feature_matrix(values):
    if values is None:
        return False
    arr = np.asarray(values)
    if arr.ndim != 2:
        return False
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return False
    return np.isfinite(arr).all()


# download the models / datasets
def get_transfer_values_inception(dataset):
    data_dir = r'./data/'
    models.inception.data_dir = os.path.join(data_dir, 'inception/')
    dataset.data_dir = os.path.join(data_dir, dataset.name + r'/')
    if not os.path.exists(dataset.data_dir):
        os.mkdir(dataset.data_dir)
    models.inception.maybe_download()
#    dataset.maybe_download()
    
    #load the inception model
    model = models.inception.Inception()
    
    #load the dataset data
#    images_train, cls_train, labels_train = dataset.load_training_data()
#    images_test, cls_test, labels_test = dataset.load_test_data()
    
    images_train = dataset.x_train
#    cls_train = dataset.y_train
#    labels_train = dataset.y_train_labels

    images_test = dataset.x_test
#    cls_test = dataset.y_test
#    labels_test = dataset.y_test_labels
    
    # path to save the cache values
    file_path_cache_train = os.path.join(dataset.data_dir, 'inception_' + dataset.name + '_train.pkl')
    file_path_cache_test = os.path.join(dataset.data_dir, 'inception_' + dataset.name + '_test.pkl')
    
    # stl10 and inception both need pixels between 0 to 255.
    # however, when using other datasets, preprocessing might 
    # be required.

    # images_scaled = images_train * 255.0

    print("Transfering training set")
    
    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                                  images=images_train,
                                                  model=model)
    if not _is_valid_feature_matrix(transfer_values_train):
        print("invalid cached inception train transfer values, recomputing cache")
        if os.path.exists(file_path_cache_train):
            os.remove(file_path_cache_train)
        transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                                      images=images_train,
                                                      model=model)
    
    print("Transfering test set")
    
    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                                  images=images_test,
                                                  model=model)
    if not _is_valid_feature_matrix(transfer_values_test):
        print("invalid cached inception test transfer values, recomputing cache")
        if os.path.exists(file_path_cache_test):
            os.remove(file_path_cache_test)
        transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                                      images=images_test,
                                                      model=model)
    return transfer_values_train, transfer_values_test


def get_transfer_values_classic_networks(dataset, network_name):

    # path to save the cache values
    file_path_cache_train = os.path.join(dataset.data_dir, network_name + '_' + dataset.name + '_train.pkl')
    file_path_cache_test = os.path.join(dataset.data_dir, network_name + '_' + dataset.name + '_test.pkl')


    #
    # if output_path is not None:
    #     history_output = output_path + "_nets" + str(args.num_models) + "_history"
    #     print('saving trained model to:', history_output)
    #     with open(history_output, 'wb') as file_pi:
    #         pickle.dump(history, file_pi)

    print("Transfering training set")

    if os.path.exists(file_path_cache_train):
        print("training set already exist on disk")
        with open(file_path_cache_train, "rb") as pick_file:
            transfer_values_train = pickle.load(pick_file)
        if not _is_valid_feature_matrix(transfer_values_train):
            print("invalid cached train transfer values, recomputing cache")
            transfer_values_train = classic_nets_imagenet.classify_img(dataset.x_train, network_name)
            with open(file_path_cache_train, "wb") as pick_file:
                pickle.dump(transfer_values_train, pick_file)
    else:
        transfer_values_train = classic_nets_imagenet.classify_img(dataset.x_train, network_name)
        with open(file_path_cache_train, "wb") as pick_file:
            pickle.dump(transfer_values_train, pick_file)

    print("Transfering test set")

    if os.path.exists(file_path_cache_test):
        print("test set already exist on disk")
        with open(file_path_cache_test, "rb") as pick_file:
            transfer_values_test = pickle.load(pick_file)
        if not _is_valid_feature_matrix(transfer_values_test):
            print("invalid cached test transfer values, recomputing cache")
            transfer_values_test = classic_nets_imagenet.classify_img(dataset.x_test, network_name)
            with open(file_path_cache_test, "wb") as pick_file:
                pickle.dump(transfer_values_test, pick_file)
    else:
        transfer_values_test = classic_nets_imagenet.classify_img(dataset.x_test, network_name)
        with open(file_path_cache_test, "wb") as pick_file:
            pickle.dump(transfer_values_test, pick_file)

    return transfer_values_train, transfer_values_test


def transfer_values_svm_scores(train_x, train_y, test_x, test_y):
    if not _is_valid_feature_matrix(train_x):
        raise ValueError(
            "SVM train features are invalid. Expected a finite 2D feature matrix, "
            "but got {}.".format(np.asarray(train_x))
        )
    if len(test_x) != 0 and not _is_valid_feature_matrix(test_x):
        raise ValueError(
            "SVM test features are invalid. Expected a finite 2D feature matrix, "
            "but got {}.".format(np.asarray(test_x))
        )
    clf = svm.SVC(probability=True)#Step 1：训练 SVM
    print("fitting svm")
    clf.fit(train_x, train_y)
    if len(test_x) != 0:
        print("evaluating svm")
        test_scores = clf.predict_proba(test_x)#Step 2：测试集预测
        print('accuracy for svm = ', str(np.mean(np.argmax(test_scores, axis=1) == test_y)))#打印测试准确率
    else:
        test_scores = []
    train_scores = clf.predict_proba(train_x)#Step 3：训练集预测（关键）
    return train_scores, test_scores#用 SVM 计算：每个样本属于各个类别的概率，这个概率值可以用来衡量样本的 difficulty

def svm_scores_exists(dataset, network_name="inception",
                      alternative_data_dir="."):
    if dataset is None:
        data_dir = alternative_data_dir
    else:
        data_dir = dataset.data_dir
    
    svm_train_path = os.path.join(data_dir, network_name + 'svm_train_values.pkl')
    svm_test_path = os.path.join(data_dir, network_name + 'svm_test_values.pkl')
    return os.path.exists(svm_train_path) and os.path.exists(svm_test_path)


def _cached_scores_match_dataset(train_scores, y_train, test_scores, y_test):
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    test_size_match = len(test_scores) == 0 or test_scores.shape[0] == y_test.shape[0]
    return (
        train_scores.shape[0] == y_train.shape[0] and
        test_size_match
    )


def get_svm_scores(transfer_values_train, y_train, transfer_values_test,
                   y_test, dataset, network_name="inception",
                   alternative_data_dir="."):
    
    if dataset is None:
        data_dir = alternative_data_dir
    else:
        data_dir = dataset.data_dir
    
    svm_train_path = os.path.join(data_dir, network_name + 'svm_train_values.pkl')
    svm_test_path = os.path.join(data_dir, network_name + 'svm_test_values.pkl')
    should_compute_scores = (
        not os.path.exists(svm_train_path) or
        not os.path.exists(svm_test_path)
    )

    if not should_compute_scores:
        with open(svm_train_path, 'rb') as file_pi:
            train_scores = pickle.load(file_pi)

        with open(svm_test_path, 'rb') as file_pi:
            test_scores = pickle.load(file_pi)

        should_compute_scores = not _cached_scores_match_dataset(
            train_scores=train_scores,
            y_train=y_train,
            test_scores=test_scores,
            y_test=y_test
        )

    if should_compute_scores:
        if not _is_valid_feature_matrix(transfer_values_train):
            if dataset is None:
                raise ValueError(
                    "Missing/invalid transfer values for SVM training and no dataset was "
                    "provided to recompute them."
                )
            print("transfer values missing or invalid, recomputing transfer features")
            if network_name == "inception":
                transfer_values_train, transfer_values_test = get_transfer_values_inception(dataset)
            else:
                transfer_values_train, transfer_values_test = get_transfer_values_classic_networks(
                    dataset, network_name
                )
        train_scores, test_scores = transfer_values_svm_scores(transfer_values_train, y_train, transfer_values_test, y_test)
        with open(svm_train_path, 'wb') as file_pi:
            pickle.dump(train_scores, file_pi)

        with open(svm_test_path, 'wb') as file_pi:
            pickle.dump(test_scores, file_pi)
    return train_scores, test_scores


def rank_data_according_to_score(train_scores, y_train, reverse=False, random=False):#train_scores:[N,C] 是 SVM 预测的每个样本属于各个类别的概率值，y_train 是样本的真实标签
    y_train = np.asarray(y_train).reshape(-1)
    train_size, _ = train_scores.shape#train_size 是样本数量 N，_ 是类别数量
    if train_size != y_train.shape[0]:
        raise ValueError(
            "train_scores and y_train must have the same number of samples, "
            "got {} and {}".format(train_size, y_train.shape[0])
        )
    hardness_score = train_scores[list(range(train_size)), y_train]#Step 1：取“正确类别概率”（关键！！！）,difficulty ≈ 该样本被正确分类的概率
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))#难度降序排序，score高 → easy;score低 → hard
    if reverse:#anti-curriculum（先学难的）
        res = np.flip(res, 0)
    if random:#random baseline 随机排序
        np.random.shuffle(res)
    return res
