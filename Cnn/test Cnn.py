# -*- coding:utf-8 -*-
# Author:Nicky
import copy
import numpy as np
import math
import struct
import matplotlib.pyplot as plt
from pathlib import Path


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()


dimensions = [28*28,10]
activation = [tanh,softmax]
distribution = [
    {'b':[0,1]},
    {'b':[0,1] , 'w':[-math.sqrt(6/dimensions[0]+dimensions[1]),math.sqrt(6/dimensions[0]+dimensions[1])]}
]  #列表


def init_parameters_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]


def init_parameters_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer-1],dimensions[layer])*(dist[1]-dist[0])+dist[0]


def init_parameters():
    parameter = []
    for i in range(len(distribution)):
        layer_parameter = {}
        for j in distribution[i].keys():
            if j == 'b' :
                layer_parameter['b'] = init_parameters_b(i)
                continue
            if j == 'w':
                layer_parameter['w'] = init_parameters_w(i)
                continue
        parameter.append(layer_parameter)
    return parameter


parameters = init_parameters()


def predict(img,parameters):
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
    l1_out = activation[1](l1_in)  # l0 = A1(data + b0)    output = A2(l0*W1 + b1)
    return l1_out


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

with open(train_img_path,'rb') as f:  # 用二进制的形式打开文件进行只读
    struct.unpack('>4i',f.read(16))  # 前16个二进制不是数据，去除
    tep_img = np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)/255  # 最大值是255
    train_img = tep_img[:train_num]
    valid_img = tep_img[train_num:]

with open(test_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)/255  # 一列 28*28个

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

# print(train_img)


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


def d_softmax(data):
    sm = softmax(data)
    return np.diag(sm)-np.outer(sm,sm)  # 矩阵


def d_tanh(data):
    return 1/(np.cosh(data))**2


onehot = np.identity(dimensions[-1])   # 输出规格的单位矩阵的数组
differential = {softmax:d_softmax,tanh:d_tanh}


def sqr_loss(img,lab,parameters):
    y_pred = predict(img,parameters)
    y = onehot[lab]
    diff = y - y_pred
    return np.dot(diff,diff)


# print(sqr_loss(train_img[0],train_label[0],parameters))
# print(d_softmax(np.array([1,2,3,4])))
# show_test(3)

# print(predict(np.random.rand(28*28),parameters))


def grad_parameters(img,lab,parameters):
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']  # 字典
    l1_out = activation[1](l1_in)

    diff = onehot[lab]-l1_out

    act1 = np.dot(differential[activation[1]](l1_in),diff)

    grad_b1 = -2*act1
    grad_w1 = -2*np.outer(l0_out,act1)  # 求外积
    grad_b0 = -2*differential[activation[0]](l0_in)*np.dot(parameters[1]['w'],act1)

    return {'w1':grad_w1,'b1':grad_b1,'b0':grad_b0}

# print(grad_parameters(train_img[1],train_label[1],parameters))
#

# b1
h = 0.0001
for i in range(10):
    img_i = np.random.randint(train_num)
    test_parameters = init_parameters()
    derivative = grad_parameters(train_img[img_i],train_label[img_i],test_parameters)['b1']
    value1 = sqr_loss(train_img[img_i],train_label[img_i],test_parameters)
    test_parameters[1]['b'][i]+=h
    value2 = sqr_loss(train_img[img_i],train_label[img_i],test_parameters)  # 验证公式
    # print(derivative[i]-(value2-value1)/h)


def valid_loss(parameters):
    loss_accu = 0
    for img_i in range(valid_num):
        loss_accu += sqr_loss(valid_img[img_i],valid_label[img_i],parameters)  # 精度误差（平方差相加）
    return loss_accu


def valid_accuracy(parameters):
    corrcect = [predict(valid_img[img_i],parameters).argmax() == valid_label[img_i] for img_i in range(valid_num)]
    print('validation accuracy : {}'.format(corrcect.count(True)/len(corrcect)))


# print(valid_loss(parameters))
# print(valid_accuracy(init_parameters()))

batch_size = 100  # 组大小为100张图片


def train_batch(current_batch,parameters):
    grad_accu = grad_parameters(train_img[current_batch*batch_size+0],train_label[current_batch*batch_size+0],parameters)  # 累加梯度
    for img_i in range(1,batch_size):
        grad_tmp = grad_parameters(train_img[current_batch*batch_size+img_i],train_label[current_batch*batch_size+img_i],parameters)
        for key in grad_accu.keys():
            grad_accu[key] += grad_tmp[key]
    for key in grad_accu.keys():
        grad_accu[key] /= batch_size  # 做平均
    return grad_accu

# print(train_batch(0,parameters))


def combine_parameters(parameters,grad,learn_rate):
    parameters_tmp = copy.deepcopy(parameters)
    parameters_tmp[0]['b'] -= learn_rate*grad['b0']
    parameters_tmp[1]['b'] -= learn_rate*grad['b1']
    parameters_tmp[1]['w'] -= learn_rate*grad['w1']
    return parameters_tmp

# print(combine_parameters(parameters,train_batch(0,parameters),1))


parameters = init_parameters()
print(valid_accuracy(parameters))
learn_rate = 1

for i in range(train_num // batch_size):
    if i % 100 == 99:
        print('running batch {}/{}'.format(i + 1, train_num // batch_size))
    grad_tmp = train_batch(i, parameters)
    parameters = combine_parameters(parameters, grad_tmp, learn_rate)
print(valid_accuracy(parameters))

# for i in range(train_num//batch_size):
#     if i%100 == 99:
#         print('running batch {}/{}'.format(i+1,train_num//batch_size))
#     grad_tmp = train_batch(i,parameters)
#     parameters = combine_parameters(parameters,grad_tmp,1)


