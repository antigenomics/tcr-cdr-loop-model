from __future__ import print_function

import os
import sys
os.environ["THEANO_FLAGS"] = "lib.cnmem=0.4"
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
N_EPOCHS      = 500
STARTING_FROM = 1
SMOOTH_WIND   = 1
SMOOTH_STEP   = 1
BATCH_SIZE    = 16
VERBOSE       = 0

##########################
##########################

(X_train, y_train), in_shape, data_name = wtrain.load_data("data/can." + str(SEQ_LEN) + ".csv.gz", "x", 4, 4, add_pos=True, for_rnn=True)
(X_test, y_test), _, _ = wtrain.load_data("data/cdr." + str(SEQ_LEN) + ".csv.gz", "x", 4, 4, add_pos=True, for_rnn=True)

beholder = wtrain.Beholder(SEQ_LEN, (X_train, y_train), (X_test, y_test), in_shape, data_name)
beholder.add_model("cnn_pos", [2,64])
beholder.add_model("cnn_pos", [3,64])
beholder.add_model("cnn_pos", [4,64])
beholder.add_model("cnn_pos", [2,128])
beholder.add_model("cnn_pos", [3,128])
beholder.add_model("cnn_pos", [4,128])
beholder.train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

# best_models = [(4,4), (6,6)]
# best_layers = [[128, 64]]

# best_hist = {}
# best_models_list = {}
# best_models_loss = pd.DataFrame()

# best_models = [(6,6)]
# for l,r in best_models:
#     _, best_models_loss = wtrain.train_model(MAX_POS, 0, "x", [2, 64], l, r, N_EPOCHS, 
#                                            best_hist, best_models_list, best_models_loss, 
#                                            features = "omega", model_type="cnn_pos", 
#                                              batch_size=BATCH_SIZE, verbose=VERBOSE)
#     _, best_models_loss = wtrain.train_model(MAX_POS, 0, "x", [3, 64], l, r, N_EPOCHS, 
#                                            best_hist, best_models_list, best_models_loss, 
#                                            features = "omega", model_type="cnn_pos", 
#                                              batch_size=BATCH_SIZE, verbose=VERBOSE)
#     _, best_models_loss = wtrain.train_model(MAX_POS, 0, "x", [4, 64], l, r, N_EPOCHS, 
#                                              best_hist, best_models_list, best_models_loss, 
#                                              features = "omega", model_type="cnn_pos", 
#                                              batch_size=BATCH_SIZE, verbose=VERBOSE)
    
#     _, best_models_loss = wtrain.train_model(MAX_POS, 0, "x", [2, 128], l, r, N_EPOCHS, 
#                                            best_hist, best_models_list, best_models_loss, 
#                                            features = "omega", model_type="cnn_pos", 
#                                              batch_size=BATCH_SIZE, verbose=VERBOSE)
#     _, best_models_loss = wtrain.train_model(MAX_POS, 0, "x", [3, 128], l, r, N_EPOCHS, 
#                                            best_hist, best_models_list, best_models_loss, 
#                                            features = "omega", model_type="cnn_pos", 
#                                              batch_size=BATCH_SIZE, verbose=VERBOSE)
#     _, best_models_loss = wtrain.train_model(MAX_POS, 0, "x", [4, 128], l, r, N_EPOCHS, 
#                                            best_hist, best_models_list, best_models_loss, 
#                                            features = "omega", model_type="cnn_pos", 
#                                              batch_size=BATCH_SIZE, verbose=VERBOSE)
    
#     _, best_models_loss = wtrain.train_model(MAX_POS, 0, "x", [2, 192], l, r, N_EPOCHS, 
#                                            best_hist, best_models_list, best_models_loss, 
#                                            features = "omega", model_type="cnn_pos", 
#                                              batch_size=BATCH_SIZE, verbose=VERBOSE)
#     _, best_models_loss = wtrain.train_model(MAX_POS, 0, "x", [3, 192], l, r, N_EPOCHS, 
#                                            best_hist, best_models_list, best_models_loss, 
#                                            features = "omega", model_type="cnn_pos", 
#                                              batch_size=BATCH_SIZE, verbose=VERBOSE)
#     _, best_models_loss = wtrain.train_model(MAX_POS, 0, "x", [4, 192], l, r, N_EPOCHS, 
#                                            best_hist, best_models_list, best_models_loss, 
#                                            features = "omega", model_type="cnn_pos", 
#                                              batch_size=BATCH_SIZE, verbose=VERBOSE)

##########################
##########################

fig = plt.figure()
fig.set_figwidth(18)
fig.set_figheight(10)

gs = plt.GridSpec(2,4) # 2 rows, 3 columns
ax1 = fig.add_subplot(gs[0,:2]) # First row, first column
ax2 = fig.add_subplot(gs[0,2:4]) # First row, second column
ax3 = fig.add_subplot(gs[1,2:4]) # First row, third column

plt.gcf().subplots_adjust(bottom=0.4)

def smooth(vec):
    res = []
    window = SMOOTH_WIND
    step = SMOOTH_STEP
    for i in range(window, len(vec)-window, step):
        res.append(np.mean(vec[i-window:i+window+1]))
    return res

# cur_hist = best_hist
# best_models = sorted([(h, np.mean(cur_hist[h].history["val_loss"][-5:])) for h in cur_hist], key=lambda x: x[1])[:10]

cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, len(beholder.models))]

for i, key in enumerate(sorted(beholder.models.keys(), key=lambda x:x[0])):
    ax1.plot(np.log2(smooth(beholder.history[key].history["loss"][STARTING_FROM:])), label=key, c=colors[i])
    ax2.plot(np.log2(smooth(beholder.history[key].history["val_loss"][STARTING_FROM:])), label=key, c=colors[i])
    
best_models_loss = beholder.test_df.sort_values("model")
ax3 = sns.boxplot(x = "val_loss", y = "model", data = best_models_loss, palette="rainbow")
# ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
    
ax1.set_title("loss")
ax2.set_title("val")
ax3.set_title("boostrapped loss")
ax1.legend(prop={'size':4})
ax1.legend(bbox_to_anchor=(.9, -1.1), loc='lower right', ncol = 1)

fig.tight_layout()

plt.savefig("loss/last.png")