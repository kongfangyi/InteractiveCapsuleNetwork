#!/usr/bin/env python

# author:q_y_jun
# Email: q_y_jun@163.com
# datetime:2020/4/16
# software: PyCharm

import os
import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Add,Multiply,BatchNormalization, Dense, Dropout, Embedding, LSTM, Cropping1D, GRU, \
    Bidirectional, Input, Flatten, Convolution1D, MaxPooling1D,Dot
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import concatenate, Lambda
from keras.utils import np_utils
from metrics import f1,returnMacroF1
from Attention_layer import AttentionM, AttentionMM
from capsule_net import Capsule
import tensorflow as tf
from keras.layers import Reshape, Concatenate, Softmax

from keras import backend as K
from interActiveLayer import interActivate,Tanh,TransMatrix, ScaleMaxPool, My2DCapsuleLayer

batch_size = 384
nb_epoch = 200
hidden_dim = 160

kernel_size = 3

nb_filter = 60

y = []

np.random.seed(7)
tf.set_random_seed(7)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# best parameter epoch: 54   dropout:0.36  hidden_dim=140
def gru(left_pickle, right_pickle, dropout=0.36, hidden_dim = 140):
    left_maxlen, left_max_features, left_num_features, left_W, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, y_test = get_feature(
        left_pickle)
    print(len(left_y_train))
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)

    left_sequence = Input(shape=(left_maxlen,), dtype='int32')
    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen,
                              weights=[left_W], trainable=False)(left_sequence)
    left_enc = GRU(hidden_dim, recurrent_dropout=dropout)(left_embedded)

    right_sequence = Input(shape=(right_maxlen,), dtype='int32')
    right_embedded = Embedding(input_dim=right_max_features, output_dim=right_num_features, input_length=right_maxlen,
                               weights=[right_W], trainable=False)(right_sequence)
    right_enc = GRU(hidden_dim, recurrent_dropout=dropout)(right_embedded)

    x = concatenate([left_enc, right_enc])
    output = Dense(6, activation='softmax')(x)
    model = Model(inputs=[left_sequence, right_sequence], outputs=output)

    return model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test


# best parameter  epoch:41   dropout:0.36    hidden dimension: 160
def bi_gru(left_pickle, right_pickle, dropoout=0.36, hidden_dim = 160):
    print("this is bi_gru model")
    left_maxlen, left_max_features, left_num_features, left_W, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, y_test = get_feature(
        left_pickle)
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)

    left_sequence = Input(shape=(left_maxlen,), dtype='int32')
    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen,
                              weights=[left_W], trainable=False)(left_sequence)
    left_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropoout))(left_embedded)

    print(np.shape(left_enc))

    right_sequence = Input(shape=(right_maxlen,), dtype='int32')
    right_embedded = Embedding(input_dim=right_max_features, output_dim=right_num_features, input_length=right_maxlen,
                               weights=[right_W], trainable=False)(right_sequence)
    right_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropoout))(right_embedded)

    print(np.shape(right_enc))

    x = concatenate([left_enc, right_enc])
    output = Dense(6, activation='softmax')(x)
    model = Model(inputs=[left_sequence, right_sequence], outputs=output)

    return model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test



# best parameter  epoch:41   dropout:0.36    hidden dimension: 160
def bi_gru_attention(left_pickle, right_pickle, dropoout=0.36, hidden_dim = 160):
    print("this is bi_gru_attention model")
    left_maxlen, left_max_features, left_num_features, left_W, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, y_test = get_feature(
        left_pickle)
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)

    left_sequence = Input(shape=(left_maxlen,), dtype='int32')
    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen,
                              weights=[left_W], trainable=False)(left_sequence)
    left_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropoout,return_sequences=True))(left_embedded)
    left_att = AttentionM()(left_enc)
    print(np.shape(left_enc))

    right_sequence = Input(shape=(right_maxlen,), dtype='int32')
    right_embedded = Embedding(input_dim=right_max_features, output_dim=right_num_features, input_length=right_maxlen,
                               weights=[right_W], trainable=False)(right_sequence)
    right_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropoout, return_sequences=True))(right_embedded)
    right_att = AttentionM()(right_enc)
    print(np.shape(right_enc))

    comb = Concatenate()([left_att,right_att])

    output = Dense(6, activation='softmax')(comb)
    model = Model(inputs=[left_sequence, right_sequence], outputs=output)

    return model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test


def cnn_model(left_pickle, right_pickle):
    left_maxlen, left_max_features, left_num_features, left_W, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, y_test = get_feature(
        left_pickle)
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)


    left_sequence = Input(shape=(left_maxlen,), dtype='int32')
    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen,
                         weights=[left_W], trainable=False)(left_sequence)
    left_embedded = Dropout(0.25)(left_embedded)
    left_conv = Convolution1D(filters=nb_filter, kernel_size=kernel_size, padding='valid',
                         activation='relu', strides=1)(left_embedded)
    left_conv = Convolution1D(filters=nb_filter, kernel_size=kernel_size, padding='valid',
                         activation='relu', strides=1)(left_conv)
    left_maxpooling = MaxPooling1D(pool_size=2)(left_conv)

    left_flatten = Flatten()(left_maxpooling)


    right_sequence = Input(shape=(right_maxlen,), dtype='int32')
    right_embedded = Embedding(input_dim=right_max_features, output_dim=right_num_features, input_length=right_maxlen,
                              weights=[right_W], trainable=False)(right_sequence)
    right_embedded = Dropout(0.25)(right_embedded)

    right_conv = Convolution1D(filters=nb_filter, kernel_size=kernel_size, padding='valid',
                              activation='relu', strides=1)(right_embedded)
    right_conv = Convolution1D(filters=nb_filter, kernel_size=kernel_size, padding='valid',
                              activation='relu', strides=1)(right_conv)
    right_maxpooling = MaxPooling1D(pool_size=2)(right_conv)
    right_dense = Dense(128, activation="relu")(right_maxpooling)

    right_flatten = Flatten()(right_dense)

    x = concatenate([left_flatten, right_flatten])

    output = Dense(6, activation='softmax')(x)
    model = Model(inputs=[left_sequence, right_sequence], outputs=output)
    return model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test

#将数字填充到一样的大小
def pad_to_givenSize(ori_matrix):
    #input: 3d matrix, (dim1,dim2,dim3)
    #purpose: pad the 2ed dim to given dim_size
    #output: 3d matrix, (dim1, dim_size, dim3)
    temp_matrix_1 = tf.zeros_like(ori_matrix[:,0:21,:])
    temp_matrix = tf.concat([ori_matrix,temp_matrix_1], axis=1)

    return temp_matrix


def gru_and_capsule_net(maxlen, max_features, num_features, W, dropout = 0.0):
    Routings = 5
    Num_capsule = 10
    Dim_capsule = 32
    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen,
                         weights=[W], trainable=False)(sequence)
    enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout, return_sequences=True))(embedded)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True, kernel_size=(3, 1))(enc)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout)(capsule)

    output = Dense(6, activation='softmax')(capsule)
    model = Model(inputs=sequence, outputs=output)
    return model


def context_gru_and_capsule_net(left_pickle, right_pickle,dropout_rate=0.46):
    Routings = 5
    Num_capsule = 10
    Dim_capsule = 32

    left_maxlen, left_max_features, left_num_features, left_W, left_X_train, left_y_train, left_X_dev, left_y_dev,left_test,y_test = get_feature(
        left_pickle)
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)

    left_sequence = Input(shape=(left_maxlen,), dtype='int32')
    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen,
                              weights=[left_W], trainable=False)(left_sequence)
    left_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(left_embedded)
    left_capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(left_enc)
    left_output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(left_capsule)
    #left_capsule = Flatten()(left_capsule)

    right_sequence = Input(shape=(right_maxlen,), dtype='int32')
    right_embedded = Embedding(input_dim=right_max_features, output_dim=right_num_features, input_length=right_maxlen,
                               weights=[right_W], trainable=False)(right_sequence)
    right_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(right_embedded)
    right_capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                           share_weights=True)(right_enc)
    right_output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(right_capsule)
    #right_capsule = Flatten()(output_capsule)

    x = Concatenate()([left_output_capsule, right_output_capsule])
    capsule = Dense(128)(x)

    output = Dense(6, activation='softmax')(capsule)
    model = Model(inputs=[left_sequence, right_sequence], outputs=output)

    return model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, \
        right_y_dev, right_test, y_test

def bigru_and_capsule_net(left_pickle, right_pickle,dropout_rate):
    Routings = 5
    Num_capsule = 10
    Dim_capsule = 32

    left_maxlen, left_max_features, left_num_features, left_W, left_X_train, left_y_train, left_X_dev, left_y_dev,left_test,y_test = get_feature(
        left_pickle)
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)

    left_sequence = Input(shape=(left_maxlen,), dtype='int32')
    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen,
                              weights=[left_W], trainable=False)(left_sequence)
    left_embedded = Dropout(dropout_rate)(left_embedded)
    # bi-lstm
    #left_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(left_embedded)
    left_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(left_embedded)

    #left_capsule = Flatten()(left_capsule)

    right_sequence = Input(shape=(right_maxlen,), dtype='int32')
    right_embedded = Embedding(input_dim=right_max_features, output_dim=right_num_features, input_length=right_maxlen,
                               weights=[right_W], trainable=False)(right_sequence)
    right_embedded = Dropout(dropout_rate)(right_embedded)
    #right_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(right_embedded)
    right_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(right_embedded)

    #right_capsule = Flatten()(output_capsule)

    x = Concatenate()([left_enc, right_enc])
    x = Reshape((-1,2*hidden_dim))(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                            share_weights=True)(x)
    output_capsule = Flatten()(capsule)
    capsule = Dense(128)(output_capsule)

    output = Dense(6, activation='softmax')(capsule)
    model = Model(inputs=[left_sequence, right_sequence], outputs=output)

    return model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, \
        right_y_dev, right_test, y_test


# best parameter epoch: 54   dropout:0.36  hidden_dim=140
def bert_sentenceCoding_lstm(left_pickle, right_pickle, dropout=0.36, hidden_dim=140):
    bert_filePath = "./pickle/bert_embeding_chnises.pickle"
    bert_file = open(bert_filePath, "rb")
    bert_left_embedding, bert_right_embedding = pickle.load(bert_file)
    bert_file.close()
    print(bert_left_embedding.shape)
    left_pickle_file = 'pickle/big_left_chinese_periodical.pickle3'
    left_pickle_file = open(left_pickle_file, 'rb')
    left_revs, left_W, left_word_idx_map, left_vocab, left_maxlen = pickle.load(left_pickle_file)
    left_pickle_file.close()
    bert_filePath = "./pickle/bert_embeding_chnises.pickle"
    bert_file = open(bert_filePath, "rb")
    bert_left_embedding, bert_right_embedding = pickle.load(bert_file)
    bert_file.close()
    print(bert_left_embedding.shape)
    left_train_len = len([i for i in left_revs if i["split"] == 1])
    left_dev_len = len([i for i in left_revs if i["split"] == 0])

    left_maxlen = len(bert_right_embedding[0])
    left_X_train = bert_left_embedding[:left_train_len].tolist()
    left_y_train = np_utils.to_categorical([i["y"] for i in left_revs[:left_train_len]])
    left_X_dev = bert_left_embedding[left_train_len:left_train_len + left_dev_len].tolist()
    left_y_dev = np_utils.to_categorical([i["y"] for i in left_revs[left_train_len:left_train_len + left_dev_len]])
    left_x_test = bert_left_embedding[left_train_len + left_dev_len:].tolist()
    y_test = [i["y"] for i in left_revs[left_train_len + left_dev_len:]]

    right_maxlen = len(bert_right_embedding[0])
    right_X_train = bert_right_embedding[:left_train_len].tolist()
    right_X_dev = bert_right_embedding[left_train_len:left_train_len + left_dev_len].tolist()
    right_x_test = bert_right_embedding[left_train_len + left_dev_len:].tolist()

    right_y_train = left_y_train
    right_y_dev = left_y_dev
    right_y_test = y_test

    left_sequence = Input(shape=(left_maxlen,), dtype='float')
    right_sequence = Input(shape=(right_maxlen,), dtype='float')
    x = Concatenate(axis=1)([left_sequence, right_sequence])
    dense1 = Dense(128, activation='relu')(x)
    dense2 = Dense(64, activation='relu')(dense1)
    dense3 = Dense(32, activation='relu')(dense2)
    output = Dense(6, activation='softmax')(dense3)
    model = Model(inputs=[left_sequence, right_sequence], outputs=output)

    return model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_x_test, right_X_train, right_y_train, right_X_dev, right_y_dev, right_x_test, right_y_test


#____________________________________________________________________________________________________________________

def interActiveCapsule(left_pickle, right_pickle,hidden_dim=160,dropout_rate=0.46, capsule_dim=32, input_kernel_size=12):
    Routings = 3  #更改
    Num_capsule = 6
    Dim_capsule = capsule_dim

    left_maxlen, left_max_features, left_num_features, left_W, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, y_test = get_feature(
        left_pickle)
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)

    left_sequence = Input(shape=(left_maxlen,), dtype='int32')
    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen,
                              weights=[left_W], trainable=False)(left_sequence)
    left_embedded = Dropout(dropout_rate)(left_embedded)
    # bi-lstm
    left_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(left_embedded)
    left_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(left_embedded)

    # left_capsule = Flatten()(left_capsule)

    right_sequence = Input(shape=(right_maxlen,), dtype='int32')
    right_embedded = Embedding(input_dim=right_max_features, output_dim=right_num_features, input_length=right_maxlen,
    weights = [right_W], trainable = False)(right_sequence)
    right_embedded = Dropout(dropout_rate)(right_embedded)
    right_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(right_embedded)
    right_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate,return_sequences=True))(right_embedded)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    # right_capsule = Flatten()(right_capsule)

    #comboVec = Concatenate(axis=1)([left_enc, right_enc])

    interActivateVec = interActivate(hidden_dims= hidden_dim)([left_enc, right_enc])
    print("input_size",interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)


    scaledPool_inter_left = MaxPooling1D(pool_size = 165)(tanh_inter_left)
    scaledPool_inter_left = Reshape((165,))(scaledPool_inter_left)

    print("scaledPool_inter_left ",scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size = 165)(tanh_inter_right)
    scaledPool_inter_right = Reshape((165,))(scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_left = Dropout(dropout_rate)(softmax_inter_left)
    softmax_inter_left_1 = Dense(165,activation="softmax")(softmax_inter_left)
    softmax_inter_left_1 = Dropout(dropout_rate)(softmax_inter_left_1)
    softmax_inter_right = Softmax()(scaledPool_inter_right)
    softmax_inter_right = Dropout(dropout_rate)(softmax_inter_right)
    softmax_inter_right_1 = Dense(165,activation="softmax")(softmax_inter_right)
    softmax_inter_right_1 = Dropout(dropout_rate)(softmax_inter_right_1)

    softmax_inter_left = Dot(axes=1)([left_enc,softmax_inter_left_1])
    print("softmax_inter_left", softmax_inter_left, left_enc)
    softmax_inter_right = Dot(axes=1)([right_enc,softmax_inter_right_1])
    print("softmax_inter_right", softmax_inter_right, right_enc)


    comboVec = Concatenate(axis=1)([softmax_inter_left, softmax_inter_right])
    comboVec = Reshape((-1,2*hidden_dim))(comboVec)
    comboVec_dropout = Dropout(dropout_rate)(comboVec)
    #print("comboVect: ", comboVec)
    #combo_gru = Bidirectional(GRU(hidden_dim,dropout=0.08,return_sequences=True))(comboVec)
    #combo_gru = Bidirectional(GRU(24, dropout=0.08))(combo_gru)
    #combo_gru = Flatten(combo_gru)

    '''
    output1 = Dense(128, activation="relu")(comboVec)
    output1 = Dropout(0.34)(output1)
    output2 = Dense(64, activation="relu")(output1)
    output2 = Dropout(0.25)(output2)
    output3 = Dense(32, activation="relu")(output2)
    output3 = Dropout(0.12)(output3)
    '''

    my2dCapsule = Capsule(routings=Routings,num_capsule=Num_capsule,dim_capsule=Dim_capsule,
                          kernel_size=input_kernel_size)(comboVec_dropout)
    my2dCapsule_dropout = Dropout(dropout_rate)(my2dCapsule)
    print("capsule output: ", my2dCapsule)
    #bilstm_capsule = Bidirectional(LSTM(hidden_dim,recurrent_dropout=0.34,return_sequences=True))(my2dCapsule)
    #bilstm_capsule = Bidirectional(LSTM(hidden_dim,recurrent_dropout=0.34, return_sequences=True))(bilstm_capsule)
    #attentioned_capsule = AttentionM()(bilstm_capsule)
    output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(my2dCapsule_dropout)
    #my2dCapsule = Flatten()(my2dCapsule)
    output = Dense(6, activation="softmax")(output_capsule )

    print("output: ", output)

    model = Model(inputs=[left_sequence, right_sequence], outputs=output)

    return model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, \
           right_y_dev, right_test, y_test

def interActive_bigru(left_pickle, right_pickle,hidden_dim,dropout_rate, capsule_dim):
    Routings = 3  #更改
    Num_capsule = 6
    Dim_capsule = capsule_dim

    left_maxlen, left_max_features, left_num_features, left_W, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, y_test = get_feature(
        left_pickle)
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)

    left_sequence = Input(shape=(left_maxlen,), dtype='int32')
    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen,
                              weights=[left_W], trainable=False)(left_sequence)
    left_embedded = Dropout(dropout_rate)(left_embedded)
    # bi-lstm
    left_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(left_embedded)
    left_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(left_embedded)

    # left_capsule = Flatten()(left_capsule)

    right_sequence = Input(shape=(right_maxlen,), dtype='int32')
    right_embedded = Embedding(input_dim=right_max_features, output_dim=right_num_features, input_length=right_maxlen,
    weights = [right_W], trainable = False)(right_sequence)
    right_embedded = Dropout(dropout_rate)(right_embedded)
    right_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(right_embedded)
    right_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate,return_sequences=True))(right_embedded)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    # right_capsule = Flatten()(right_capsule)

    #comboVec = Concatenate(axis=1)([left_enc, right_enc])

    interActivateVec = interActivate(hidden_dims= hidden_dim)([left_enc, right_enc])
    print("input_size",interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size = 165)(tanh_inter_left)
    scaledPool_inter_left = Reshape((165,))(scaledPool_inter_left)
    print("scaledPool_inter_left ",scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size = 165)(tanh_inter_right)
    scaledPool_inter_right = Reshape((165,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    softmax_inter_left = Dot(axes=1)([left_enc,softmax_inter_left])
    print("softmax_inter_left", softmax_inter_left, left_enc)
    softmax_inter_right = Dot(axes=1)([right_enc,softmax_inter_right])
    print("softmax_inter_right", softmax_inter_right, right_enc)


    comboVec = Concatenate(axis=1)([softmax_inter_left, softmax_inter_right])
    comboVec = Reshape((-1,2*hidden_dim))(comboVec)
    comboVec_dropout = Dropout(dropout_rate)(comboVec)
    #print("comboVect: ", comboVec)
    combo_gru = Bidirectional(GRU(hidden_dim,dropout=dropout_rate))(comboVec_dropout)
    #combo_gru_att = AttentionM()(combo_gru)
    #combo_gru = Flatten(combo_gru)

    '''
    output1 = Dense(128, activation="relu")(comboVec)
    output1 = Dropout(0.34)(output1)
    output2 = Dense(64, activation="relu")(output1)
    output2 = Dropout(0.25)(output2)
    output3 = Dense(32, activation="relu")(output2)
    output3 = Dropout(0.12)(output3)
    '''

    #my2dCapsule = Capsule(routings=Routings,num_capsule=Num_capsule,dim_capsule=Dim_capsule,
                          #kernel_size=input_kernel_size)(comboVec_dropout)
    #my2dCapsule_dropout = Dropout(dropout_rate)(comboVec_dropout)
    print("capsule output: ", combo_gru)
    #comboVec_dropout = Flatten()(comboVec_dropout)
    #bilstm_capsule = Bidirectional(LSTM(hidden_dim,recurrent_dropout=0.34,return_sequences=True))(my2dCapsule)
    #bilstm_capsule = Bidirectional(LSTM(hidden_dim,recurrent_dropout=0.34, return_sequences=True))(bilstm_capsule)
    #attentioned_capsule = AttentionM()(bilstm_capsule)
    #output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(my2dCapsule_dropout)
    #my2dCapsule = Flatten()(my2dCapsule)
    output = Dense(6, activation="softmax")(combo_gru)

    print("output: ", output)

    model = Model(inputs=[left_sequence, right_sequence], outputs=output)

    return model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, \
           right_y_dev, right_test, y_test

def interActive_bilstm_attention(left_pickle, right_pickle,hidden_dim,dropout_rate, capsule_dim):
    Routings = 3  #更改
    Num_capsule = 6
    Dim_capsule = capsule_dim

    left_maxlen, left_max_features, left_num_features, left_W, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, y_test = get_feature(
        left_pickle)
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)

    left_sequence = Input(shape=(left_maxlen,), dtype='int32')
    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen,
                              weights=[left_W], trainable=False)(left_sequence)
    left_embedded = Dropout(dropout_rate)(left_embedded)
    # bi-lstm
    left_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(left_embedded)
    left_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(left_embedded)

    # left_capsule = Flatten()(left_capsule)

    right_sequence = Input(shape=(right_maxlen,), dtype='int32')
    right_embedded = Embedding(input_dim=right_max_features, output_dim=right_num_features, input_length=right_maxlen,
    weights = [right_W], trainable = False)(right_sequence)
    right_embedded = Dropout(dropout_rate)(right_embedded)
    right_embedded = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate, return_sequences=True))(right_embedded)
    right_enc = Bidirectional(GRU(hidden_dim, recurrent_dropout=dropout_rate,return_sequences=True))(right_embedded)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    # right_capsule = Flatten()(right_capsule)

    #comboVec = Concatenate(axis=1)([left_enc, right_enc])

    interActivateVec = interActivate(hidden_dims= hidden_dim)([left_enc, right_enc])
    print("input_size",interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size = 165)(tanh_inter_left)
    scaledPool_inter_left = Reshape((165,))(scaledPool_inter_left)
    print("scaledPool_inter_left ",scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size = 165)(tanh_inter_right)
    scaledPool_inter_right = Reshape((165,))(scaledPool_inter_right)
    print("scaledPool_inter_right ", scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    softmax_inter_right = Softmax()(scaledPool_inter_right)

    softmax_inter_left = Dot(axes=1)([left_enc,softmax_inter_left])
    print("softmax_inter_left", softmax_inter_left, left_enc)
    softmax_inter_right = Dot(axes=1)([right_enc,softmax_inter_right])
    print("softmax_inter_right", softmax_inter_right, right_enc)


    comboVec = Concatenate(axis=1)([softmax_inter_left, softmax_inter_right])
    comboVec = Reshape((-1,2*hidden_dim))(comboVec)
    comboVec_dropout = Dropout(dropout_rate)(comboVec)
    #print("comboVect: ", comboVec)
    combo_gru = Bidirectional(LSTM(hidden_dim,dropout=dropout_rate,return_sequences=True))(comboVec_dropout)
    combo_gru_att = AttentionM()(combo_gru)
    #combo_gru = Flatten(combo_gru)

    '''
    output1 = Dense(128, activation="relu")(comboVec)
    output1 = Dropout(0.34)(output1)
    output2 = Dense(64, activation="relu")(output1)
    output2 = Dropout(0.25)(output2)
    output3 = Dense(32, activation="relu")(output2)
    output3 = Dropout(0.12)(output3)
    '''

    #my2dCapsule = Capsule(routings=Routings,num_capsule=Num_capsule,dim_capsule=Dim_capsule,
                          #kernel_size=input_kernel_size)(comboVec_dropout)
    #my2dCapsule_dropout = Dropout(dropout_rate)(comboVec_dropout)
    print("capsule output: ", combo_gru_att)
    #comboVec_dropout = Flatten()(comboVec_dropout)
    #bilstm_capsule = Bidirectional(LSTM(hidden_dim,recurrent_dropout=0.34,return_sequences=True))(my2dCapsule)
    #bilstm_capsule = Bidirectional(LSTM(hidden_dim,recurrent_dropout=0.34, return_sequences=True))(bilstm_capsule)
    #attentioned_capsule = AttentionM()(bilstm_capsule)
    #output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(my2dCapsule_dropout)
    #my2dCapsule = Flatten()(my2dCapsule)
    output = Dense(6, activation="softmax")(combo_gru_att )
    print("output: ", output)
    model = Model(inputs=[left_sequence, right_sequence], outputs=output)

    return model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, \
           right_y_dev, right_test, y_test

def get_feature(pickle_path):
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_path, 'rb'))
    maxlen = 165
    X_train, X_test, X_dev, y_train, y_dev, y_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]

    n_test_sample = X_test.shape[0]

    len_sentence = X_train.shape[1]  # 200

    max_features = W.shape[0]

    num_features = W.shape[1]  # 400

    return maxlen, max_features, num_features, W, X_train, y_train, X_dev, y_dev, X_test, y_test

def get_attention(sent_model, sequences):

    cnt_reviews_1 = sequences[0].shape[0]
    sent_att_w_1 = sent_model.layers[22].get_weights()
    sent_all_att_1 = []
    sent_before_att_1 = K.function([sent_model.layers[0].input,sent_model.layers[1].input, K.learning_phase()],
                                   [sent_model.layers[22].input])
    # print(sent_att_w[0].shape, sent_att_w[1].shape, sent_att_w[2].shape)

    for i in range(cnt_reviews_1):
        sent_each_att_1 = sent_before_att_1([sequences[0][i].reshape((1, sequences[0].shape[-1])),
                                              sequences[1][i].reshape((1, sequences[1].shape[-1])),0])
        mask_1 = [0 if w == 0 else 1 for w in sequences[0][i]]

        sent_each_att_1 = cal_att_weights(sent_each_att_1, sent_att_w_1, mask_1)
        sent_each_att_1 = sent_each_att_1.ravel()
        sent_all_att_1.append(sent_each_att_1)

    sent_before_att_2 = K.function([sent_model.layers[0].input,sent_model.layers[1].input,K.learning_phase()],
                                 [sent_model.layers[23].input])
    cnt_reviews_2 = sequences[1].shape[0]
    sent_att_w_2 = sent_model.layers[23].get_weights()
    sent_all_att_2 = []

    # print(sent_att_w[0].shape, sent_att_w[1].shape, sent_att_w[2].shape)

    for i in range(cnt_reviews_2):
        sent_each_att_2 = sent_before_att_2([sequences[0][i].reshape((1, sequences[0].shape[-1])),
                                              sequences[1][i].reshape((1, sequences[1].shape[-1])),0])
        mask = [0 if w == 0 else 1 for w in sequences[1][i]]

        sent_each_att_2 = cal_att_weights(sent_each_att_2, sent_att_w_2, mask)
        sent_each_att_2 = sent_each_att_2.ravel()
        sent_all_att_2.append(sent_each_att_2)

    return [sent_all_att_1, sent_all_att_2]

def cal_att_weights(output, att_w, mask):
    eij = np.tanh(np.dot(output[0], att_w[0]) + att_w[1])
    eij = np.squeeze(eij*att_w[1])
    # eij = eij.reshape((eij.shape[0], eij.shape[1]))
    ai = np.exp(eij)
    weights = ai / np.sum(ai)
    weights = weights * mask
    return weights


if __name__ == '__main__':
    #dropout_rate_arr = [ 0.02, 0.04]
    #实验室runcode从0.06开始
    #dropout_rate_arr = [0.06, 0.08, 0.1, 0.16]
    #实验室runcode1从0.2开始
    dropout_rate_arr = [0.46]
    hidden_dim_arr = [160]
    capsule_dim_arr = [6]

    #left_pickle_file = './pickle/small_corpus/left_chinese_periodical.pickle3'
    left_pickle_file = './pickle/wassa/left_english_periodical.pickle3'

    #right_pickle_file = './pickle/small_corpus/right_chinese_periodical.pickle3'
    right_pickle_file = './pickle/wassa/right_english_periodical.pickle3'
    result_dic = dict()
    result_dic['dropout_rate'] = []
    result_dic['hidden_dim_loop'] = []
    result_dic['capsule_dim'] = []
    result_dic['macro_f1'] = []
    for hidden_dim_loop in hidden_dim_arr:
        for dropout_rate in dropout_rate_arr:
            for capsule_dim in capsule_dim_arr:
                    print('the parameters is: ||||||dropout rate', dropout_rate,"|||||hiddendim",hidden_dim_loop,
                          "\|||||||capsule dim:",capsule_dim )
                    # for item in hidden_dim_arr:
                    #     print('===============1457\11\47/=================== context gru didden dim is {0} =================================='.format(item))
                    model, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, \
                        right_y_dev, right_test, y_test = interActiveCapsule(left_pickle_file, right_pickle_file)

                    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
                    callbackList = [EarlyStopping(monitor='val_acc', patience=3), ModelCheckpoint(filepath = ".\\checkPoint\\modelCheckpoint.h5",
                                                                                                  monitor = "val_acc", save_best_only = True)]
                    # model.fit(X_train, y_train, validation_data=[..X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2,
                    #           callbacks=[early_stopping])
                    model.fit([left_X_train, right_X_train], left_y_train, validation_data=[[left_X_dev, right_X_dev], left_y_dev],
                              batch_size = batch_size, epochs = nb_epoch, verbose= 2,
                              callbacks = callbackList)
                    y_pred = model.predict([left_test, right_test], batch_size=batch_size)
                    y_pred = np.argmax(y_pred, axis=1)

                    '''
                    #生成情感可视化数据
                    y_pred_att_1, y_pred_att_2  = get_attention(model, [left_test, right_test])
                    # pickle_file = os.path.join('pickle', 'sst_binary_attention.pickle3')
                    pickle_file = os.path.join('pickle', '1.8.sst_contextCapsule_chinese.pickle3')
                    pickle.dump([y_pred_att_1, y_pred_att_2 , y_pred], open(pickle_file, 'wb'))
                    '''

                    result_output = pd.DataFrame(data={'test_sentiment': y_test, "sentiment": y_pred})
                    # # Use pandas to write the comma-separated output file
                    result_save_path = "./result/interActive_capsule.csv"
                    result_output.to_csv(result_save_path, index=False, quoting=3)
                    result_outputStr = "interActive_capsule_net:" + ",the macro f1 score is: " + str(returnMacroF1(result_save_path))
                    print(result_outputStr)


                    result_dic['dropout_rate'].append(dropout_rate)
                    result_dic['hidden_dim_loop'].append(hidden_dim_loop)
                    result_dic['capsule_dim'].append(capsule_dim)
                    result_dic['macro_f1'].append(returnMacroF1(result_save_path))
        result_df= pd.DataFrame(result_dic)
        result_df.to_csv("./result.csv")


    #dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('gating_1').output)
    # 以这个model的预测值作为输出
    #dense1_output = dense1_layer_model.predict([left_test, right_test])

    #file = './pickle/sigmoid_gru.pickle3'
    #pickle.dump([dense1_output], open(file, 'wb'))
