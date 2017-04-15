from __future__ import print_function, division
from collections import Counter

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, concatenate, Input, LSTM, GRU, Conv1D, add, Flatten
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau

from .preprocess import *
from .eval import *


def dense_model(shape, output, h_units = [256, 128, 64]):  
    model = Sequential()
    
    model.add(Dense(h_units[0], input_shape=shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(.3))
    
    for num in h_units[1:]:
        model.add(Dense(num))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(.3))
        
    model.add(Dense(output))
    model.add(PReLU())
    
    model.compile(optimizer="nadam", loss="mse")
    
    return model


def dense_pos_model(shape, output, h_units):
    pep_in = Input(shape)
    len_in = Input((1,))
    
    merged = []
    merged.append(concatenate([pep_in, len_in]))
    
    pep_br = Dense(h_units[0])(merged[-1])
    pep_br = BatchNormalization()(pep_br)
    pep_br = PReLU()(pep_br)
    pep_br = Dropout(.3)(pep_br)
    
    for num in h_units[1:]:
        merged.append(concatenate([pep_br, len_in]))
        pep_br = Dense(num)(merged[-1])
        pep_br = BatchNormalization()(pep_br)
        pep_br = PReLU()(pep_br)
        pep_br = Dropout(.3)(pep_br)
    
    merged.append(concatenate([pep_br, len_in]))
    pep_br = Dense(output)(merged[-1])
    pred = PReLU()(pep_br)
    
    model = Model(inputs=[pep_in, len_in], outputs=pred)
    
    model.compile(optimizer="nadam", loss="mse")
    
    return model


def dense_poslen_model(shape, output, h_units):
    pep_in = Input(shape)
    pos_in = Input((1,))
    len_in = Input((1,))
    
    merged = []
    merged.append(concatenate([pep_in, pos_in, len_in]))
    
    pep_br = Dense(h_units[0])(merged[-1])
    pep_br = BatchNormalization()(pep_br)
    pep_br = PReLU()(pep_br)
    pep_br = Dropout(.3)(pep_br)
    
    for num in h_units[1:]:
        merged.append(concatenate([pep_br, pos_in, len_in]))
        pep_br = Dense(num)(merged[-1])
        pep_br = BatchNormalization()(pep_br)
        pep_br = PReLU()(pep_br)
        pep_br = Dropout(.3)(pep_br)
    
    merged.append(concatenate([pep_br, pos_in, len_in]))
    pep_br = Dense(output)(merged[-1])
    pred = PReLU()(pep_br)
    
    model = Model(inputs=[pep_in, pos_in, len_in], outputs=pred)
    
    model.compile(optimizer="nadam", loss="mse")
    
    return model


def rnn_model(shape, output, h_units, rnn_type):
    # h_units[0] - number of units in RNN
    # rnn_type = ["lstm", "gru", "bilstm", "bigru"]
    model = Sequential()
    
    if rnn_type.find("gru") != -1:  # TODO: change this to something more sane
        rnn_layer =  GRU(h_units[0], kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal", dropout=.2, recurrent_dropout=.2,
                       unroll=True, input_shape=shape)
    elif rnn_type.find("lstm") != -1:
        rnn_layer = LSTM(h_units[0], kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal", dropout=.2, recurrent_dropout=.2,
                       unroll=True, input_shape=shape)
    else:
        print("Can't find neither GRU not LSTM")
        return 0
    
    model.add(rnn_layer)
    model.add(BatchNormalization())
    model.add(PReLU())
    
    for num in h_units[1:]:
        model.add(Dense(num))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(.3))
        
    model.add(Dense(output))
    model.add(PReLU())
    
    # model.compile(optimizer="nadam", loss=make_mean_sample_error(max_pos))
    model.compile(optimizer="nadam", loss="mse")
    
    return model


def cnn_pos_model(shape, output, h_units):
    def res_block(prev_layer, shape):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        branch = Conv1D(h_units[1], 1, kernel_initializer="he_normal")(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        
        return add([prev_layer, branch])
    
    # merged = []
    # merged.append(concatenate([pep_in, len_in]))
    
    # pep_br = Dense(h_units[0])(merged[-1])
    # pep_br = BatchNormalization()(pep_br)
    # pep_br = PReLU()(pep_br)
    # pep_br = Dropout(.3)(pep_br)
    
    # for num in h_units[1:]:
    #     merged.append(concatenate([pep_br, len_in]))
    #     pep_br = Dense(num)(merged[-1])
    #     pep_br = BatchNormalization()(pep_br)
    #     pep_br = PReLU()(pep_br)
    #     pep_br = Dropout(.3)(pep_br)
    
    pep_in = Input(shape)
    pos_in = Input((1,))
    
    pep_branch = res_block(pep_in, shape)
    for ind in range(1, h_units[0]):
        pep_branch = res_block(pep_branch, shape)
        
    pep_branch = Flatten()(pep_branch)
    
    merged = concatenate([pep_branch, pos_in])
    
    merged = Dense(64)(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    pred = Dense(output)(merged)
    pred = PReLU()(pred)
    
    model = Model(inputs=[pep_in, pos_in], outputs=pred)
    
    model.compile(optimizer="nadam", loss="mse")
    
    return model


def delta_model(shape, output, h_units):
    def _block(prev_layer, shape):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        branch = Conv1D(192, 1, kernel_initializer="he_normal")(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        
        return add([prev_layer, branch])
    
    inp_forw = Input(shape = shape)
    inp_back = Input(shape = shape)
    inp_pos = Input(shape = (1,))
    # inp_len = Input((1,)) # one hot encoding for length
    
    shared_model = Sequential()
    shared_model.add(GRU(h_units[0][0], kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal", dropout=.2, recurrent_dropout=.2, 
                       unroll=True, input_shape = shape))
    
    for num in h_units[0][1:]:
        shared_model.add(Dense(num))
        shared_model.add(BatchNormalization())
        shared_model.add(PReLU())
        shared_model.add(Dropout(.3))
    
    diff_forw = shared_model(inp_forw)
    diff_forw = Dense(1)(diff_forw)
    pred_forw = PReLU(name="pred_forw")(diff_forw)
    
    diff_back = shared_model(inp_back)
    diff_back = Dense(1)(diff_back)
    pred_back = PReLU(name="pred_back")(diff_back)
    
    merged = concatenate([pred_forw, pred_back])
    
    for num in h_units[1]:
        merged = concatenate([merged, inp_pos])
        merged = Dense(num)(merged)
        merged = BatchNormalization()(merged)
        merged = PReLU()(merged)
        merged = Dropout(.3)(merged)
    
    merged = Dense(1)(merged)
    pred_coord = PReLU(name="pred_final")(merged)
    
    model = Model(inputs=[inp_forw, inp_back, inp_pos], outputs=[pred_forw, pred_back, pred_coord])
    
    model.compile(optimizer="nadam", loss="mse")
    
    return model