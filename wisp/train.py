from __future__ import print_function, division
from collections import Counter

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, concatenate, Input, LSTM, GRU, Conv1D, add, Flatten
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau

from .preprocess import *
from .eval import *
from .model import *

def train_model(max_pos, n_clust, coord, layers, 
                left_window, right_window, 
                n_epochs, 
                hist, model_list, df_loss,
                how = "no", scale = "no", 
                features = "onehot", model_type = "dense", fading = False):
    model_name = model_type + ".l" + str(left_window) + "_r" + str(right_window) + "." + "-".join(map(str, layers)) + "." + how + "_" + scale + "." + features

    if n_clust > 0:
        if fading:
            model_name += ".clust.fade_" + str(n_clust)
        else:
            model_name += ".clust_" + str(n_clust)
    
    if model_name not in hist:
        print(model_name, end = "\t")
        
        #
        # Load the data
        #
        df_cdr = pd.read_csv("data/cdr_coord_" + coord + ".csv.gz")
        df_can = pd.read_csv("data/can_coord_" + coord + ".csv.gz")
        
        #
        # Scale the data
        #
        if how in ["col", "abs"]:
            scale_data(df_cdr, how, scale)
            scale_data(df_can, how, scale)

        #
        # Extract feactures
        #
        input_shape = (0,)
        if features == "onehot":
            coord_fun = onehot
        elif features == "omega":
            coord_fun = onehot_omega
        elif features == "twohot":
            coord_fun = twohot_omega
        else:
            print("Unknown parameter", features)
            return 0
        
        for_rnn = False
        if model_type in ["gru", "lstm", "cnn_pos"]:
            for_rnn = True
        
        X_can, y_can = coord_fun(df_can, left_window, right_window, max_pos, for_rnn)
        X_cdr, y_cdr = coord_fun(df_cdr, left_window, right_window, max_pos, for_rnn)

        #
        # Prepare to build the model
        #
        if model_type == "dense":
            model_fun = dense_model
            input_shape = (len(CHARS)*(right_window+left_window+1),)
            
        elif model_type == "dense_pos":
            model_fun = dense_pos_model
            input_shape = (len(CHARS)*(right_window+left_window+1),)
            
            # add positions
            X_can = [X_can, np.array([float((x % max_pos) + 1) / max_pos for x in range(X_can.shape[0])])]
            X_cdr = [X_cdr, np.array([float((x % max_pos) + 1) / max_pos for x in range(X_cdr.shape[0])])]
            
        elif model_type == "dense_poslen":
            model_fun = dense_poslen_model
            input_shape = (len(CHARS)*(right_window+left_window+1),)
            
            # add positions and lengths
            X_can = [X_can, np.array([float((x % max_pos) + 1) / max_pos for x in range(X_can.shape[0])]), np.full((X_can.shape[0],1), max_pos)]
            X_cdr = [X_cdr, np.array([float((x % max_pos) + 1) / max_pos for x in range(X_cdr.shape[0])]), np.full((X_cdr.shape[0],1), max_pos)]

        elif model_type in ["gru", "lstm"]:
            model_fun = rnn_model
            input_shape = (right_window+left_window+1, len(CHARS))
        elif model_type in ["cnn_pos"]:
            X_can = [X_can, np.array([float((x % max_pos) + 1) / max_pos for x in range(X_can.shape[0])])]
            X_cdr = [X_cdr, np.array([float((x % max_pos) + 1) / max_pos for x in range(X_cdr.shape[0])])]
            
            model_fun = cnn_pos_model
            input_shape = (right_window+left_window+1, len(CHARS))
        else:
            print("Unknown parameter", coord_fun)
            return 0
        
        if features == "twohot":
            input_shape = (right_window+left_window, 2*len(CHARS))
        
        #
        # Build the model
        #
        if model_type in ["dense", "dense_pos", "dense_poslen"]:
            model = model_fun(input_shape, 1, layers)
        elif model_type in ["gru", "lstm", "cnn_pos"]:
            model = model_fun(input_shape, 1, layers)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, cooldown=1, min_lr=0.001)
        
        if n_clust == 0:
            hist_obj = model.fit(X_can, y_can, batch_size=64, epochs=n_epochs, verbose=0, validation_data=(X_cdr, y_cdr), callbacks=[reduce_lr])
        else:
            labels, labels_cnt, min_cluster, min_cluster_size = [], -1, -1, -1
            if model_type == "dense":
                labels, labels_cnt, min_cluster, min_cluster_size = cluster(y_can.reshape((-1, max_pos)), n_clust)
            labels_new = []
            for x in labels:
                labels_new.extend([x] * max_pos)
            labels = labels_new
            
            if fading:
                weight_vec = np.array([np.log(min_cluster_size) / np.log(labels_cnt[x]) for x in labels])
                weight_vec = np.exp(np.log(weight_vec) / (200 ** .5))

                hist_obj = model.fit(X_can, y_can, sample_weight=weight_vec, batch_size=64, epochs=n_epochs, verbose=0, validation_data=(X_cdr, y_cdr), callbacks=[reduce_lr])
            else:
                weight_vec = np.array([np.log(min_cluster_size) / np.log(labels_cnt[x]) for x in labels])

                hist_obj = model.fit(X_can, y_can, sample_weight=weight_vec, batch_size=64, epochs=n_epochs, verbose=0, validation_data=(X_cdr, y_cdr), callbacks=[reduce_lr])
        
        hist[model_name] = hist_obj
        model_list[model_name] = model
        
        print(hist_obj.history["val_loss"][-1], end="\t")
        
        boot_loss_vec = bootstrap_cdr(model_list[model_name], X_cdr, y_cdr, max_pos)
        df_new = pd.DataFrame({"val_loss": boot_loss_vec, "model": model_name})
        
        print("(", np.mean(boot_loss_vec), ")")
        
        return model, pd.concat([df_loss, df_new])
    else:
        return model_list[model_name], None