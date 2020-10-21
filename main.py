### Article information
### Journal: Frontiers in Microbiology
### Title: Helix matrix transformation combined with convolutional neural network algorithm for matrix-assisted laser desorption ionization-time of flight mass spectrometry-based bacterial identification
### Authors: Jin Ling, Gaomin Li, Hong Shao, Hong Wang, Hongrui Yin, Hu zhou, Yufei Song, Gang Chen
### doi:

# encoding = utf-8
# Code for MALDI-TOF MS-based bacterial identification
# Python 3.7.x
# CentOS work path: /data/nn
# Windows system work path: E:\\v2

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2 as cv
import os

# 选择显卡，0或1
# Choose graphics card, 0 or 1
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# 内螺旋矩阵变换
# Helix matrix transformation function
def HelixMatrix(matrix, x_cur=0, y_cur=0, number=1, n=50):
    if n == 0:
        #print(matrix)
        return 0
    if n == 1:
        matrix[x_cur][y_cur] = number
        #print(matrix)
        return 0
    # 上
    # Up
    for i in range(n):
        matrix[x_cur][y_cur + i] = number
        number += 1
    # 右
    # Right
    for i in range(n - 1):
        matrix[x_cur + i + 1][y_cur + n - 1] = number
        number += 1
    # 下
    # Down
    for i in range(n - 1):
        matrix[x_cur + n - 1][y_cur + n - 2 - i] = number
        number += 1
    # 左
    # Left
    for i in range(n - 2):
        matrix[x_cur + n - 2 - i][y_cur] = number
        number += 1
    HelixMatrix(matrix, x_cur + 1, y_cur + 1, number, n - 2)

# Obtain data from a text file
## data: y values of MALDI-TOF MS spectra
## nn: label numbers of species
def ObtainData(source="D:\\v2\\data_3000.txt"):
    with open(source,'r') as f:
        lines = f.readlines()
        data = []
        nn = []
        for line in lines:
            line = line.strip('\n')
            dx = line.split(';')[0].split(', ')
            data.append([np.array(dx[0:2500]).astype('float32')])
            nn.append(int(line.split(';')[1]))
    return data,nn

# 数据的内螺旋矩阵变换
# Transform data into helix matrix form
def Data2Helix(data):
    matrix = np.zeros((50, 50))
    HelixMatrix(matrix)
    np_data = np.array(data)
    dl = []
    for nd in np_data:
        matrix1 = matrix.copy()
        for i in range(0, matrix1.shape[0]):
            for j in range(0, matrix1.shape[1]):
                matrix1[i][j] = nd[0][int(matrix1[i][j]) - 1]
        dl.append(matrix1)
    np_d = np.array(dl).astype('uint8')
    # 重要，增加一个维度以适应2D卷积
    # Add an extra-dimension for fitting the input of 2D convolution
    np_d = np.expand_dims(np_d,-1)
    return np_d

# Transform type of labels
def GetLabel(nn):
    np_l = np.array(nn).astype('uint8')
    return np_l

# Show picture
def ShowPic(np_dn):
    plt.clf()
    img = np_dn
    plt.imshow(img, cmap=plt.cm.gray_r)
    plt.show()

# 二值化
# Function of binarization
def Threshold(img):
    # The threshold for binarization can be set as an integer depending on your MS quality
    # Here, the threshold for binarization is set as 16
    ret, th1 = cv.threshold(img, 16, 255, cv.THRESH_BINARY)
    return th1

# 数据每条做二值化变换后返回
# Prepare data with binarization and resize
def GetDataSet(np_d, np_l):
    dts = []
    for d in np_d:
        #二值化
        dt = Threshold(d)
        #缩放到25*25
        dtr = [(cv.resize(Threshold(dt), (25, 25), interpolation=cv.INTER_CUBIC))/255]
        dts.append(dtr)
    x_train, x_test, y_train, y_test = train_test_split(np.array(dts).transpose(0,2,3,1), np_l, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test

# 测试数据每条做二值化变换后返回
# Prepare test data with binarization and resize
def GetTestDataSet(np_td, np_tl):
    dts = []
    for d in np_td:
        #二值化
        dt = Threshold(d)
        #缩放到25*25
        dtr = [(cv.resize(Threshold(dt), (25, 25), interpolation=cv.INTER_CUBIC))/255]
        dts.append(dtr)
    x_test = np.array(dts).transpose(0,2,3,1)
    y_test = np_tl
    return x_test, y_test

# 训练数据切片
# Prepare data slices for training
def TrainDataSlic(x_train,y_train):
    y_train = np.float32(tf.keras.utils.to_categorical(y_train))
    batch_size = 30
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(10)
    return train_data

# 测试数据切片
# Prepare data slices for test
def TestDataSlic(x_test, y_test):
    y_test = np.float32(tf.keras.utils.to_categorical(y_test))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.batch(len(x_test))
    return test_data

# 构建CNN
# Build a CNN model
def CNN(train_data, test_data):
    metrics = [tf.keras.metrics.TruePositives(name='tp'), tf.keras.metrics.FalsePositives(name='fp'), tf.keras.metrics.TrueNegatives(name='tn'), tf.keras.metrics.FalseNegatives(name='fn'), tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    # Shape of input: 25,25,1
    input_xs = tf.keras.Input(shape=(25, 25, 1))
    # The first convolution layer, filter size is 4, kernel size is 3
    conv = tf.keras.layers.Conv2D(4, 3, padding="SAME", activation=tf.nn.relu)(input_xs)
    # The second convolution layer, filter size is 8, kernel size is 3
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(8, 3, padding="SAME", activation=tf.nn.relu)(conv)
    conv = tf.keras.layers.MaxPool2D()(conv)
    # The third convolution layer, filter size is 16, kernel size is 3
    conv = tf.keras.layers.Conv2D(16, 3, padding="SAME", activation=tf.nn.relu)(conv)
    flat = tf.keras.layers.Flatten()(conv)
    # The fully connected layer
    fc = tf.keras.layers.Dense(64, activation=tf.nn.relu)(flat)
    # The output layer
    logits = tf.keras.layers.Dense(14, activation=tf.nn.softmax)(fc)
    model = tf.keras.Model(inputs=input_xs, outputs=logits)
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss=tf.losses.categorical_crossentropy, metrics=metrics)
    history = model.fit(train_data, epochs=1, validation_data=test_data)
    score = model.evaluate(test_data)
    print('last score:', score)
    return model, history

# Calculate score of accuracy
def GetYtureScores(y_test,ps):
    predictions = []
    scores = []
    y_ture = []
    # 以最大的概率值选择预测结果（数字编号）
    for p in ps:
        cl = np.argmax(p)
        predictions.append(int(cl))
        score = max(p)
        scores.append(score)
    # 根据预测结果和标签是否相等获取y_ture，调取理论正确值的概率作为该点的得分
    for i,ele in enumerate(y_test):
        if predictions[i] == y_test[i]:
            y_ture.append(1)
        else:
            y_ture.append(0)
    return y_ture,scores, predictions

if __name__ == '__main__':
    ## TRAINING ##
    # 数据格式为：每行一条，分号分隔ms_data和label,ms_data每个点以逗号空格分隔
    # Data format (line in text): y value,y value, ... y value,y value;label
    data, nn = ObtainData("/data/nn/data_sample.txt")
    #data, nn = ObtainData("E:\\v2\\data_sample.txt")
    # Transform data and label into Numpy array type
    np_d = Data2Helix(data)
    np_l = GetLabel(nn)
    x_train, x_test, y_train, y_test = GetDataSet(np_d, np_l)
    train_data = TrainDataSlic(x_train,y_train)
    test_data = TestDataSlic(x_test, y_test)
    # Model and history of training process
    model, history = CNN(train_data, test_data)
    print(model)
    print(history)
    model.save("/data/nn/model_sample.h5")
    #model.save("E:\\v2\\model_sample.h5")

    ## PREDICTION ##
    test_data, test_nn = ObtainData("/data/nn/data_sample.txt")
    #test_data, test_nn = ObtainData("E:\\v2\\data_sample.txt")
    np_td = Data2Helix(test_data)
    np_tl = GetLabel(test_nn)
    tx_test, ty_test = GetTestDataSet(np_td, np_tl)
    # 读取模型并预测
    model_file = "/data/nn/model_sample.h5"
    #model_file = "E:\\v2\\model_sample.h5"
    model = tf.keras.models.load_model(model_file)
    ps = model.predict(tx_test)
    y_ture, scores, predictions = GetYtureScores(ty_test, ps)
    # In y_ture array, number one means correct prediction and number zero means wrong prediction,
    print(y_ture)
    # Max probability value for each prediction of classification
    print(scores)
    # Predicted label for each test sample
    print(predictions)

