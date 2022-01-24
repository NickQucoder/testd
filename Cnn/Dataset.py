# -*- coding:utf-8 -*-
# Author:Nicky

import struct

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

dataset_path = Path('F:\\杂文件\\Python数据集\\Hand Writing')

train_img_path = dataset_path/'train-images.idx3-ubyte'
train_label_path = dataset_path/'train-labels.idx1-ubyte'
test_img_path = dataset_path/'t10k-images.idx3-ubyte'
test_label_path = dataset_path/'t10k-labels.idx1-ubyte'

# train_f = open(train_img_path,'rb')
# print(struct.unpack('>4i',train_f.read(16)))
# print(np.fromfile(train_f,dtype=np.uint8).reshape(-1,4*4))
train_num = 50000
valid_num = 10000
test_num = 10000

with open(train_img_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    tep_img = np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)
    train_img = tep_img[:train_num]
    valid_img = tep_img[train_num:]

with open(test_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)

with open(train_label_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    tep_label = np.fromfile(f, dtype=np.uint8)
    train_label = tep_label[:train_num]
    valid_label = tep_label[train_num:]

with open(test_label_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    test_label = np.fromfile(f,dtype=np.uint8)
    # test_label = np.fromfile(f, dtype=np.uint8)

# train_label_f = open(train_label_path,'rb')
# struct.unpack('>2i',train_label_f.read(8))


def show_train(index):
    plt.imshow(train_img[index].reshape(28,28),cmap='gray')
    print('label:{}'.format(train_label[index]))
    plt.show()


def show_valid(index):
    plt.imshow(valid_img[index].reshape(28,28),cmap='gray')
    print('label:{}'.format(valid_label[index]))
    plt.show()


def show_test(index):
    plt.imshow(test_img[index].reshape(28,28),cmap='gray')
    print('label:{}'.format(test_label[index]))
    plt.show()


# show_valid(np.random.randint(10000))

# img = train_img[np.random.randint(28*28)].reshape(28,28)
# plt.imshow(img,cmap='gray')
# plt.show()
