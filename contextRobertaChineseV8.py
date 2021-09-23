# -*- coding: utf-8 -*-
#使用tensor-hub
from __future__ import division, print_function

import logging

logging.basicConfig(level=logging.ERROR)

import numpy as np
from capsuleNetv2 import Capsule
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import sequence

from keras.utils import np_utils
from transformers import *

import pandas as pd
from metrics import f1, returnMacroF1
from tensorflow.keras.layers import Reshape, Concatenate, Softmax
from tensorflow.keras.layers import Concatenate, Lambda

from tensorflow.keras import backend as K
from interActiveLayer import interActivate, Tanh, TransMatrix
from tensorflow.keras.layers import Add, Multiply, BatchNormalization, Dense, Dropout, Embedding, LSTM, Cropping1D, GRU, \
    Bidirectional, Input, Flatten, Convolution1D, MaxPooling1D, Dot
from bert4keras.models import build_transformer_model
from tensorflow.keras.models import Model
import pickle

from tokenizers import BertWordPieceTokenizer
import tensorflow_hub as hub

print(tf.__version__)

config_path = "chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json"
checkpoint_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

# ___________________________________________________________________________________________________________

import numpy as np

'''
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import MaxPool3D
'''

# tf2
# '''
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool3D, Layer


# '''


# test harness
# creates a Sequential model out of a single layer and passes the
# input through it to produce output
def test_layer(layer, x):
    layer_config = layer.get_config()
    layer_config["input_shape"] = x.shape
    layer = layer.__class__.from_config(layer_config)
    model = Sequential()
    model.add(layer)
    model.compile("rmsprop", "mse")
    x_ = np.expand_dims(x, axis=0)
    return model.predict(x_)[0]


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    scale_1 = K.square(scale) * x
    scale_2 = (1 + K.square(scale)) * scale
    return scale_1 / scale_2


# custom layer
class interActivate(Layer):

    def __init__(self, activation=None, hidden_dims=240, **kwargs):
        self.activation = activation
        self.hidden_dims = hidden_dims
        super(interActivate, self).__init__(**kwargs)

    # 输入的上下文长度需一致
    def build(self, input_shape):
        self.shape = input_shape
        self.w = self.add_weight(name="w", shape=(2 * self.hidden_dims, 2 * self.hidden_dims),
                                 initializer="normal", trainable=True)
        super(interActivate, self).build(input_shape)

    def call(self, x):
        # 先进行相乘 front_slice*w*transpos(back_slice)
        front_slice = x[0]
        back_slice = x[1]
        temp_prod = K.dot(front_slice, self.w)
        print("temp_prod.shape", temp_prod.shape)
        temp_x = K.permute_dimensions(back_slice, (0, 2, 1))
        print("temp_x.shape", temp_x.shape)
        end_prod = K.batch_dot(temp_prod, temp_x)
        print("end_prod.shape", end_prod.shape)

        '''
        #call函数的返回值不能是多个否则无法分解开，因此只能返回一个值而不能在该函数内多次对front及back部分进行操作
        #在结果上使用tanh函数
        front_tanh = K.tanh(end_prod)
        trans_back = K.permute_dimensions(end_prod,(0,2,1))
        back_tanh = K.tanh(trans_back)

        #在结果上使用
        '''

        return end_prod

    # 注意：如果输入的形状发生改变，此处一定要标明
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 34, 34)


class TransMatrix(Layer):
    def __init__(self):
        super(TransMatrix, self).__init__()

    def build(self, input_shape):
        self.shape = input_shape

    def call(self, x):
        return K.permute_dimensions(x, (0, 2, 1))

    # 注意：如果输入的形状发生改变，此处一定要标明返回的数据的形状
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[1])


class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def build(self, input_shape):
        self.shape = input_shape

    def call(self, x):
        return K.tanh(x)


# 根据输入dim进行maxPooling并将维度降低,dim = 1为对列进行maxpooling;dim = 2为对行进行maxpooling
class ScaleMaxPool(Layer):
    def __init__(self, trans_dim):
        super(ScaleMaxPool, self).__init__()
        self.trans_dim = trans_dim

    def build(self, input_shape):
        self.shape = input_shape

    def call(self, x):
        if self.trans_dim == 1:
            print("original_matrix", x)
            pooled_matrix = MaxPool3D((1, self.input_shape[1], 1))(x)
            print("poold_matrix", pooled_matrix)
            scaledMatrix = K.reshape(pooled_matrix, (self.input_shape[0], self.input_shape[1]))
            print("scaled_matrix", scaledMatrix)
            return scaledMatrix
        else:
            print("original_matrix", x)
            pooled_matrix = MaxPool3D((1, 1, self.input_shape[1]))(x)
            print("poold_matrix", pooled_matrix)
            scaledMatrix = K.reshape(pooled_matrix, (self.input_shape[0], self.input_shape[1]))
            print("scaled_matrix", scaledMatrix)
            return scaledMatrix

    # 注意：如果输入的形状发生改变，此处一定要标明返回的数据的形状
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.input_shape[1])


# 定义胶囊网络
class My2DCapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(My2DCapsuleLayer, self).__init__(**kwargs)
        self.temp_shape = 12
        self.output_dim = 6
        self.activation = squash

    # 初始化的参数包括U（输入矩阵）和M（中间映射矩阵）中间的权重矩阵w1,从M到输出V(本文中共六个）的矩阵
    def build(self, input_shape):
        super(My2DCapsuleLayer, self).build(input_shape)
        self.shape = input_shape
        # 初始化U、M之间的系数矩阵
        self.w1 = self.add_weight(name="w1", shape=(self.shape[-1], self.temp_shape),
                                  initializer="normal", trainable=True)
        self.w2 = self.add_weight(name="w2", shape=(self.temp_shape, self.output_dim),
                                  initializer="normal", trainable=True)

    def call(self, x):
        # 得到第一个中间隐藏层矩阵U
        self.M1 = x @ self.w1
        # 得到第二个中间隐藏层矩阵M
        self.M2 = self.M1 @ self.w2
        # 挤压得到输出层矩阵V
        self.V = squash(self.M2)
        return self.V

    def compute_output_shape(self, input_shape):
        return (self.input_shape[0], self.output_dim)


# ____________________________________________________________________________________________________________

def _convert_to_transformer_inputs(instance, tokenizer,max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    """默认返回input_ids,token_type_ids,attention_mask"""
    inputs_tokens = tokenizer.encode(instance).tokens

    padding_length = 0
    if len(inputs_tokens) < MAX_SEQUENCE_LENGTH:
        padding_length = max_sequence_length - len(inputs_tokens)
    # 填充
    inputs_tokens = inputs_tokens + ['pad'] * padding_length

    return [inputs_tokens[:MAX_SEQUENCE_LENGTH]]



def compute_input_arrays(train_data_input,  max_sequence_length):
    input_tokens = []
    tokenizer = BertWordPieceTokenizer('chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt')
    for instance in tqdm(train_data_input):
        tokens = _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)
        input_tokens.append(tokens[0])

    return [np.asarray(input_tokens, dtype=np.str)]

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
    X_train, X_test, X_dev, y_train, y_dev, y_test = [], [], [], [], [], []
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


def get_feature(pickle_path):
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_path, 'rb'))
    X_train, X_test, X_dev, y_train, y_dev, y_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]

    n_test_sample = X_test.shape[0]

    len_sentence = X_train.shape[1]  # 200

    max_features = W.shape[0]

    num_features = W.shape[1]  # 400

    return maxlen, max_features, num_features, W, X_train, y_train, X_dev, y_dev, X_test, y_test


pickle_file = './pickle/left_chinese_periodical_168.pickle3'
maxlen, max_features, num_features, W, X_train, y_train, X_dev, y_dev, X_test, y_test = get_feature(pickle_file)


def split_model_v1():
    input_id_left = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask_left = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn_left = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    input_id_right = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask_right = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn_right = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    # config = BertConfig.from_pretrained("./bert_base_chinese/bert-base-chinese-config.json", output_hidden_states=True)
    config = BertConfig.from_pretrained("./bert_base_chinese/bert-base-chinese-config.json", output_hidden_states=True)
    bert_model = TFBertModel.from_pretrained("./bert_base_chinese/bert-base-chinese-tf_model.h5", config=config)

    # 产生左边的bert模型
    sequence_output_left, pooler_output_left, last_left = bert_model(input_id_left,
                                                                     attention_mask=input_mask_left,
                                                                     token_type_ids=input_atn_left)
    # (bs,140,768)(bs,768)
    x_1 = tf.keras.layers.GlobalAveragePooling1D()(sequence_output_left)
    x_1 = tf.keras.layers.Dropout(0.15)(x_1)
    x_1 = tf.keras.layers.Dense(32, activation='softmax')(x_1)

    # 产生右边的bert模型
    sequence_output_right, pooler_output_right, last_right = bert_model(input_id_right,
                                                                        attention_mask=input_mask_right,
                                                                        token_type_ids=input_atn_right)
    # (bs,140,768)(bs,768)
    x_2 = tf.keras.layers.GlobalAveragePooling1D()(sequence_output_right)
    x_2 = tf.keras.layers.Dropout(0.15)(x_2)
    x_2 = tf.keras.layers.Dense(32, activation='softmax')(x_2)

    x = Concatenate(axis=1)([x_1, x_2])

    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(6, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[input_id_left, input_mask_left, input_atn_left,
                                          input_id_right, input_mask_right, input_atn_right
                                          ], outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=8e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])
    return model


def split_model():
    hiden_dim = 256
    bert_coding_len = 768
    input_left = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH), dtype=tf.string)

    input_right = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH), dtype=tf.string)
    bert = hub.KerasLayer(
        './chinese_roberta_wwm_ext_L-12_H-768_A-12/zh-roberta-wwm-L12',
        output_key='pooled_output',
        trainable=True)
    pooled_bert_seq_left = bert(input_left)
    '''
    print("input_left",bert_seq_left)
    #bert_seq_left = tf.keras.layers.Masking()(bert_seq_left)
    print("input_left_masking", bert_seq_left)
    bert_seq_right = bert(input_right)
    #bert_seq_right = tf.keras.layers.Masking()(bert_seq_right)
    # (bs,140,768)(bs,768)
    x_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(120, return_sequences=True))(bert_seq_left)

    x_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(120, return_sequences=True))(bert_seq_right)



    
    interActivateVec = interActivate(hidden_dims=120)([x_1, x_2])
    print("input_size", interActivateVec)

    tanh_inter_left = Tanh()(interActivateVec)
    inter_trans = TransMatrix()(interActivateVec)
    tanh_inter_right = Tanh()(inter_trans)

    scaledPool_inter_left = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH)(tanh_inter_left)
    scaledPool_inter_left = Reshape((MAX_SEQUENCE_LENGTH,))(scaledPool_inter_left)

    print("scaledPool_inter_left ", scaledPool_inter_left)
    scaledPool_inter_right = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH)(tanh_inter_right)
    scaledPool_inter_right = Reshape((MAX_SEQUENCE_LENGTH,))(scaledPool_inter_right)

    softmax_inter_left = Softmax()(scaledPool_inter_left)
    #    softmax_inter_left = Dropout(dropout_rate)(softmax_inter_left)
    softmax_inter_left_1 = Dense(MAX_SEQUENCE_LENGTH, activation="softmax")(softmax_inter_left)
    #    softmax_inter_left_1 = Dropout(dropout_rate)(softmax_inter_left_1)
    softmax_inter_right = Softmax()(scaledPool_inter_right)
    #    softmax_inter_right = Dropout(dropout_rate)(softmax_inter_right)
    softmax_inter_right_1 = Dense(MAX_SEQUENCE_LENGTH, activation="softmax")(softmax_inter_right)
    #    softmax_inter_right_1 = Dropout(dropout_rate)(softmax_inter_right_1)

    bert_seq_left = Reshape((50, -1))(bert_seq_left)
    bert_seq_right = Reshape((50, -1))(bert_seq_right)
    softmax_inter_left = Dot(axes=1)([bert_seq_left, softmax_inter_left_1])
    print("softmax_inter_left", softmax_inter_left, input_left)
    softmax_inter_right = Dot(axes=1)([bert_seq_right, softmax_inter_right_1])
    print("softmax_inter_right", softmax_inter_right, input_left)

    comboVec = Concatenate(axis=1)([softmax_inter_left, softmax_inter_right])
    print("comboVec",comboVec)
    comboVec = Reshape((-1, 768))(comboVec)
    #    comboVec_dropout = Dropout(dropout_rate)(comboVec)
    # print("comboVect: ", comboVec)
    #combo_gru = Bidirectional(GRU(512, dropout=0.08, return_sequences=True))(comboVec)
    # combo_gru = Bidirectional(GRU(24, dropout=0.08))(combo_gru)
    # combo_gru = Flatten(combo_gru)


    print("combo_gru",comboVec)
    my2dCapsule = Capsule(routings=3, num_capsule=6, dim_capsule=32)(comboVec)
    #my2dCapsule_dropout = Dropout(dropout_rate)(my2dCapsule)
    #print("capsule output: ", my2dCapsule)
    bilstm_capsule = Bidirectional(LSTM(16,recurrent_dropout=0.34,return_sequences=False))(my2dCapsule)
    # bilstm_capsule = Bidirectional(LSTM(hidden_dim,recurrent_dropout=0.34, return_sequences=True))(bilstm_capsule)
    # attentioned_capsule = AttentionM()(bilstm_capsule)
    output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(my2dCapsule)
    #my2dCapsule = Flatten()(my2dCapsule)
    '''
    output = Dense(6, activation="softmax")(pooled_bert_seq_left)

    print("output: ", output)

    model = tf.keras.models.Model(inputs=[input_left,
                                          input_right], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])
    return model


def pad_sequence(train_input_data, dev_input_data, test_input_data, maxlen):
    x_train_f = [sequence.pad_sequences(i_list, maxlen=maxlen) for i_list in train_input_data]
    x_dev_f = [sequence.pad_sequences(i_list, maxlen=maxlen) for i_list in dev_input_data]
    x_test_f = [sequence.pad_sequences(i_list, maxlen=maxlen) for i_list in test_input_data]
    return x_train_f, x_dev_f, x_test_f


def get_context(left_pickle, right_pickle, dropout=0.36, hidden_dim=140):
    left_maxlen, left_max_features, left_num_features, left_W, left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, y_test = get_feature(
        left_pickle)
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)
    right_maxlen, right_max_features, right_num_features, right_W, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test = get_feature(
        right_pickle)

    return left_X_train, left_y_train, left_X_dev, left_y_dev, left_test, right_X_train, right_y_train, right_X_dev, right_y_dev, right_test, y_test


import pickle

if __name__ == '__main__':
    batch_size = 1
    nb_epoch = 10
    # load train/dev/test test
    train_path = './data/context/train.txt'
    dev_path = './data/context/dev.txt'
    test_path = './data/context/test.txt'

    train = pd.read_csv(train_path, sep='\t',
                        error_bad_lines=False)
    dev = pd.read_csv(dev_path, sep='\t', error_bad_lines=False)
    test = pd.read_csv(test_path, sep='\t',
                       error_bad_lines=False)

    # left_x
    left_x_train = train["left"]
    left_x_test = test["left"]
    left_x_dev = dev["left"]

    # right_x
    right_x_train = train["right"]
    right_x_test = test["right"]
    right_x_dev = dev["right"]

    # the length of different sets
    lens_train = [len(i.split()) for i in left_x_train]
    max_len = max(lens_train)
    MAX_SEQUENCE_LENGTH = max_len
    MAX_SEQUENCE_LENGTH = 50
    print("max_len is:", max_len)


    #'''
    # 生成bert词向量
    # left bertEmbedding

    left_train_inputs = compute_input_arrays(left_x_train,  MAX_SEQUENCE_LENGTH)
    file = open('.\\pickle/chineseContext_bert_left_train.pickle', 'wb')
    pickle.dump(left_train_inputs, file, protocol = 4)
    file.close()

    left_dev_inputs = compute_input_arrays(left_x_dev,  MAX_SEQUENCE_LENGTH)
    file = open('.\\pickle/chineseContext_bert_left_dev.pickle', 'wb')
    pickle.dump(left_dev_inputs, file, protocol = 4)
    file.close()

    left_test_inputs = compute_input_arrays(left_x_test,  MAX_SEQUENCE_LENGTH)
    file = open('.\\pickle/chineseContext_bert_left_test.pickle', 'wb')
    pickle.dump(left_test_inputs, file, protocol = 4)
    file.close()

    # right bertEmbedding
    right_train_inputs = compute_input_arrays(right_x_train,  MAX_SEQUENCE_LENGTH)
    file = open('.\\pickle/chineseContext_bert_right_train.pickle', 'wb')
    pickle.dump(right_train_inputs, file, protocol = 4)
    file.close()

    right_dev_inputs = compute_input_arrays(right_x_dev,  MAX_SEQUENCE_LENGTH)
    file = open('.\\pickle/chineseContext_bert_right_dev.pickle', 'wb')
    pickle.dump(right_dev_inputs, file, protocol = 4)
    file.close()


    right_test_inputs = compute_input_arrays(right_x_test, MAX_SEQUENCE_LENGTH)
    file = open('.\\pickle/chineseContext_bert_right_test.pickle', 'wb')
    pickle.dump(right_test_inputs, file, protocol = 4)
    file.close()
    #'''


    print("已经处理完成")


    # 读取
    file_content = open('.\\pickle/chineseContext_bert_left_train.pickle', 'rb')
    left_train_inputs = pickle.load(file_content)
    file_content.close()

    file_content = open('.\\pickle/chineseContext_bert_left_dev.pickle', 'rb')
    left_dev_inputs = pickle.load(file_content)
    file_content.close()

    file_content = open('.\\pickle/chineseContext_bert_left_test.pickle', 'rb')
    left_test_inputs = pickle.load(file_content)
    file_content.close()

    # right
    file_content = open('.\\pickle/chineseContext_bert_right_train.pickle', 'rb')
    right_train_inputs = pickle.load(file_content)
    file_content.close()

    file_content = open('.\\pickle/chineseContext_bert_right_dev.pickle', 'rb')
    right_dev_inputs = pickle.load(file_content)
    file_content.close()

    file_content = open('.\\pickle/chineseContext_bert_right_test.pickle', 'rb')
    right_test_inputs = pickle.load(file_content)
    file_content.close()

    #'''

    print("已经处理完成")

    # dropout_rate = [0.32, 0.34, 0.36, 0.38]
    # dropout_rate = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32,
    #               0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]
    dropout_rate = [0.36]

    # hidden_dim_arr = [120, 140, 160, 180]

    pickle_file = '.\\pickle/chineseContext_bert.pickle'

    # revs为评论数据；W为gensim的权重矩阵;word_idx_map是单词Index索引；vocab是词典，每个单词出现的次数；maxlen是每句话中的单词数量
    # revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))

    train_inputs = [left_train_inputs[0]] + [right_train_inputs[0]]
    dev_inputs = [left_dev_inputs[0]] + [right_dev_inputs[0]]
    test_inputs = [left_test_inputs[0]] + [right_test_inputs[0]]
    print("开始训练")
    callbackList = [EarlyStopping(monitor='val_acc', patience=2), ModelCheckpoint(
        filepath=".\\checkPoint\\modelCheckpoint.h5",
        monitor="val_acc", save_best_only=True)]
    model = split_model()
    model.fit(train_inputs, y_train,
              validation_data=[dev_inputs, y_dev],
              batch_size=batch_size, epochs=nb_epoch, verbose=2)
    y_pred = model.predict(test_inputs, batch_size=batch_size)

    y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={'test_sentiment': y_test, "sentiment": y_pred})
    # # Use pandas to write the comma-separated output file
    result_save_path = "./result/wassa_context_bert_wordEmbedding.csv"
    result_output.to_csv(result_save_path, index=False, quoting=3)
    result_outputStr = "context_and_capsule_net:" + ",the macro f1 score is: " + str(returnMacroF1(result_save_path))
    print(result_outputStr)