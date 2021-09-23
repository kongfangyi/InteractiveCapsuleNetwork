#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/3 20:52
# @Author  : q_y_jun
# @email   : q_y_jun@163.com
# @File    : bi_lstm.py

import os
import pickle
import numpy as np
import pandas as pd

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Lambda, GRU, Bidirectional, Input, Flatten, Convolution1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from metrics import f1,returnMacroF1
from Attention_layer import AttentionM
from capsule_net import Capsule
from sklearn.metrics import f1_score

np.random.seed(100)

batch_size = 64
nb_epoch = 300
hidden_dim = 120
nb_filter = 60
kernel_size = 3


# test = pd.read_csv('./data/test_binary.csv', sep='\t')

def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x


def make_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, X_dev, y_train, y_dev, y_test= [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)
            y_test.append(y)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    y_train = np_utils.to_categorical(np.array(y_train))
    y_dev = np_utils.to_categorical(np.array(y_dev))
    y_test = np.array(y_test)

    return [X_train, X_test, X_dev, y_train, y_dev, y_test]

def dense(maxlen, max_features, num_features, W, hidden_dim=160, drop_out=0.46):

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    #embedded = Dropout(drop_out)(embedded)
    # bi-lstm
    #embedded = LSTM(hidden_dim, recurrent_dropout=drop_out, return_sequences=True)(embedded)
    #enc = GRU(hidden_dim, recurrent_dropout=drop_out)(embedded)
    flatten_embedded = Flatten()(embedded)
    dense = Dense(128, activation="relu")(flatten_embedded)

    output = Dense(6, activation='softmax')(dense)
    model = Model(inputs=sequence, outputs=output)

    return model

#  best epoch:42    dropout:0.28   hidden dimension: 120
def gru(maxlen, max_features, num_features, W, hidden_dim=160,dropout_rate= 0.46):

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)    # bi-lstm
    embedded = GRU(hidden_dim, recurrent_dropout=0.28, return_sequences=False)(embedded)
    output = Dense(6, activation='softmax')(embedded)
    model = Model(inputs=sequence, outputs=output)

    return model

#  best epoch:42    dropout:0.28   hidden dimension: 120
def bi_gru(maxlen, max_features, num_features, W, hidden_dim=160,dropout_rate= 0.46):

    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=0.46))(embedded)
    output = Dense(6, activation='softmax')(enc)
    model = Model(inputs=sequence, outputs=output)

    return model



def capsule(maxlen, max_features, num_features, W, hidden_dim=160,dropout_rate= 0.46):

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    embedded = Dropout(dropout_rate)(embedded)
    # bi-lstm
    #embedded = GRU(hidden_dim, recurrent_dropout=0.28, return_sequences=True)(embedded)
    enc = GRU(hidden_dim, recurrent_dropout=0.28)(embedded)
    capusleVec = Capsule(num_capsule=6,routings=3,kernel_size=(3,1))(enc)
    dense = Dense(128, activation="relu")(capusleVec)

    output = Dense(6, activation='softmax')(dense)
    model = Model(inputs=sequence, outputs=output)

    return model


def cnn_model(maxlen, max_features, num_features, W):

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen,
                         weights=[W], trainable=False)(sequence)
    embedded = Dropout(0.25)(embedded)
    conv = Convolution1D(filters=nb_filter, kernel_size=kernel_size, padding='valid',
                         activation='relu', strides=1)(embedded)
    conv = Convolution1D(filters=nb_filter, kernel_size=kernel_size, padding='valid',
                         activation='relu', strides=1)(conv)
    # fc1_dropout = Dropout(0.3)(conv)
    maxpooling = MaxPooling1D(pool_size=2)(conv)
    dense = Dense(128, activation="relu")(maxpooling)
    flatten = Flatten()(dense)
    output = Dense(6, activation='softmax')(flatten)
    model = Model(inputs=sequence, outputs=output)
    return model


def bigru_and_attention(maxlen, max_features, num_features, W, dropout = 0.0):
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    embedded = Dropout(dropout)(embedded)
    bigru = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout, return_sequences=True))(embedded)
    att = AttentionM()(bigru)
    output = Dense(6, activation='softmax')(att)
    model = Model(inputs=sequence, outputs=output)

    return model

def gru_and_attention(maxlen, max_features, num_features, W, dropout = 0.0):
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    embedded = Dropout(dropout)(embedded)
    gru = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout, return_sequences=True))(embedded)
    att = AttentionM()(gru)
    output = Dense(6, activation='softmax')(att)
    model = Model(inputs=sequence, outputs=output)
    return model



def gru_and_capsule_net(maxlen, max_features, num_features, W, hidden_dim):
    dropout = 0.46
    Routings = 3
    Num_capsule = 128
    Dim_capsule = 6
    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen,
                         weights=[W], trainable=False)(sequence)
    embedded = Dropout(dropout)(embedded)
    # bi-lstm
    embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout, return_sequences=True))(embedded)
    enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout, return_sequences=True))(embedded)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True, kernel_size=(3, 1))(enc)
    output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Dropout(dropout)(output_capsule)
    output = Dense(6, activation='softmax')(capsule)
    model = Model(inputs=sequence, outputs=output)
    return model



if __name__ == '__main__':
    # dropout_rate = [0.32, 0.34, 0.36, 0.38]
    #dropout_rate = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32,
    #               0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]
    dropout_rate = [0.24]

    # hidden_dim_arr = [120, 140, 160, 180]

    pickle_file = '.\\pickle\\wassa_single_periodical.pickle3'

    #revs为评论数据；W为gensim的权重矩阵;word_idx_map是单词Index索引；vocab是词典，每个单词出现的次数；maxlen是每句话中的单词数量
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    #将每个句子都扩展为maxlen大小这里是164
    X_train, X_test, X_dev, y_train, y_dev, y_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    #训练集的大小
    n_train_sample = X_train.shape[0]
    #测试集的大小
    n_test_sample = X_test.shape[0]
    #句子长度padding的长度
    len_sentence = X_train.shape[1]  # 200

    max_features = W.shape[0]  #the size of vocabulary

    num_features = W.shape[1]  # 400
    for item in dropout_rate:
        print('=================================== lstm result is {0} =================================='.format(item))
        #maxlen为句子Padding后的长度，max_features是字典大小,num_features是词向量长度，W为词嵌入矩阵
        #model = cnn_model(maxlen, max_features, num_features, W)
        #model = gru(maxlen, max_features, num_features, W)
        #model = gru(maxlen, max_features, num_features, W)
        #model = bi_gru(maxlen, max_features, num_features, W)
        model = bigru_and_attention(maxlen, max_features, num_features, W, dropout=item)
        #model = gru_and_capsule_net(maxlen, max_features, num_features, W,hidden_dim=160)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
        callbackList = [EarlyStopping(monitor='val_acc', patience=3),
                        ModelCheckpoint(filepath=".\\checkPoint\\modelCheckpoint.h5",
                                        monitor="val_acc", save_best_only=True)]
        model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                  callbacks=callbackList)
        y_pred = model.predict(X_test, batch_size=batch_size)
        y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={'test_sentiment': y_test, "sentiment": y_pred})
    result_path = "single_bert.csv"
    result_output.to_csv("./result/"+result_path, index=False, quoting=3)
    print("macro f1 score is :", f1_score(y_test,y_pred, average="macro"))