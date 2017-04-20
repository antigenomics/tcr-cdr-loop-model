from __future__ import print_function

import os
import sys
os.environ["THEANO_FLAGS"] = "lib.cnmem=0.3"
import theano

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, minmax_scale, maxabs_scale
from sklearn.metrics import mean_squared_error
from sklearn.cluster import MiniBatchKMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam

import seaborn as sns
sns.set_style("whitegrid") 

import wisp
from wisp import model as wmodel
from wisp import eval as weval
from wisp import train as wtrain


##########################
##########################

SEQ_LEN       = int(sys.argv[1])
N_EPOCHS      = 700
STARTING_FROM = 30
SMOOTH_WIND   = 3
SMOOTH_STEP   = 1
BATCH_SIZE    = 16
VERBOSE       = 0

##########################
##########################

(X_train, y_train), in_shape, data_name = wtrain.load_data("data/can." + str(SEQ_LEN) + ".csv.gz", "x", 4, 4, add_pos=True, for_rnn=True)
(X_test, y_test), _, _ = wtrain.load_data("data/cdr." + str(SEQ_LEN) + ".csv.gz", "x", 4, 4, add_pos=True, for_rnn=True)

beholder = wtrain.Beholder(folderpath="can_cdr", seq_len=SEQ_LEN, 
                           train_data=(X_train, y_train), test_data=(X_test, y_test), 
                           input_shape=in_shape, data_name=data_name)
beholder.add_model("cnn_pos", [2,64])
beholder.add_model("cnn_pos", [3,64])
beholder.add_model("cnn_pos", [4,64])
beholder.add_model("cnn_pos", [2,128])
beholder.add_model("cnn_pos", [3,128])
beholder.add_model("cnn_pos", [4,128])
beholder.train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
beholder.save(starting_from=STARTING_FROM, smooth_window=SMOOTH_WIND, smooth_step=SMOOTH_STEP)