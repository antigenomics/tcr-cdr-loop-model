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


def load_data(filepath, coord, left_window, right_window, add_pos=False, add_len=False, for_rnn=False, len_num=0, delta=False):
    df = pd.read_csv(filepath, index_col=0)
    seq_len = len(df["seq"][0])
        
    if coord == "x":
        df = df.ix[:, :(seq_len+1)]
    elif coord == "y":
        df = df.ix[:, [0] + list(range(seq_len + 1, seq_len*2 + 2))]
    elif coord == "z":
        df = df.ix[:, [0] + list(range(seq_len*2 + 2, seq_len*3 + 3))]
    else:
        print("Unknown coordinate")
        return None
    
    name = "l" + str(left_window) + "_r" + str(right_window)
    
    X = []
    y = []
    shape = 0
    if not delta:
        X, y = onehot_omega(df, left_window, right_window, seq_len, for_rnn)
        shape = (X.shape[1], len(CHARS))
    else:
        print("Not implemented yet")
        sys.exit()
    
    if add_pos:
        X = [X, np.array([float((x % seq_len) + 1) / seq_len for x in range(X.shape[0])])]
    
    if add_len:
        print("Not implemented yet")
        return None
    
    return (X, y), shape, name


class Beholder:
    """
    Operates on the specific train / test data, for multiple models.
    For different types of data use multiple Beholders.
    """
    
    def __init__(self, seq_len, train_data, test_data, input_shape, data_name):
        self.models = {}
        self.history = {}
        self.test_df = pd.DataFrame()
        self._model_fun = {"dense": dense_model, 
                           "dense_pos": dense_pos_model, 
                           "dense_poslen": dense_poslen_model, 
                           "gru": gru_model, 
                           "cnn_pos": cnn_pos_model, 
                           "delta": delta_model}
        self._inp_shape = input_shape
        self._data_name = data_name
        self._output_size = 1
        self._X_train = train_data[0]
        self._y_train = train_data[1]
        self._X_test = test_data[0]
        self._y_test = test_data[1]
        self._seq_len = seq_len
        
    
    def add_model(self, model_type, layers, n_clust=0, fading=False):
        model_name = model_type + "." + self._data_name + "." + "-".join(map(str, layers))
        
        if n_clust > 0:
            if fading: 
                model_name += ".clust.fade_" + str(n_clust)
            else:      
                model_name += ".clust_" + str(n_clust)
        
        print("[Beholder] Adding the model", model_name, end="...\t")
        self.models[model_name] = self._model_fun[model_type](self._inp_shape, self._output_size, layers)
        print("Done.")
    
    
    def train(self, n_epochs, batch_size=16, verbose=0):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, cooldown=1, min_lr=0.0005)
        print("[Beholder] Training...")
        for model_name in self.models:
            print(model_name, end="\t")
            self.history[model_name] = self.models[model_name].fit(self._X_train, self._y_train, 
                                                                   batch_size=batch_size, verbose=verbose, epochs=n_epochs, 
                                                                   validation_split = .2, callbacks=[reduce_lr])
            print(self.history[model_name].history["val_loss"][-1], end="\t")
            print("(", self.test(model_name), ")")
    
    
    def test(self, model_name):
        boot_loss_vec = bootstrap_cdr(self.models[model_name], self._X_test, self._y_test, self._seq_len)
        df_new = pd.DataFrame({"val_loss": boot_loss_vec, "model": model_name})
        self.test_df = pd.concat([self.test_df, df_new])
        return np.mean(boot_loss_vec)
    
    
    def save(self, filepath):
        for name in self.models:
            self.save_model(name)
        
    
    def save_model(self, name):
        pass
        
    
    def load(self, filepaths):
        pass
    

def train_model(seq_len, n_clust, coord, layers, 
                left_window, right_window, 
                n_epochs, 
                hist, model_list, df_loss,
                how = "no", scale = "no", 
                features = "onehot", model_type = "dense", fading = False,
                batch_size = 16, verbose = 2):
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
        df_cdr = pd.read_csv("data/can." + str(seq_len) + ".csv.gz")
        df_can = pd.read_csv("data/put." + str(seq_len) + ".csv.gz")
        
        if coord == "x":
            df_cdr = df_cdr.ix[:, 1:(seq_len+2)]
            df_can.reset_index(inplace=True, drop=True)
            df_can = df_can.ix[:, 1:(seq_len+2)]
            df_can.reset_index(inplace=True, drop=True)        
        else:
            print("Only X is allowed")
            return None
        
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
        
        X_can, y_can = coord_fun(df_can, left_window, right_window, seq_len, for_rnn)
        X_cdr, y_cdr = coord_fun(df_cdr, left_window, right_window, seq_len, for_rnn)

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
            X_can = [X_can, np.array([float((x % seq_len) + 1) / seq_len for x in range(X_can.shape[0])])]
            X_cdr = [X_cdr, np.array([float((x % seq_len) + 1) / seq_len for x in range(X_cdr.shape[0])])]
            
        elif model_type == "dense_poslen":
            model_fun = dense_poslen_model
            input_shape = (len(CHARS)*(right_window+left_window+1),)
            
            # add positions and lengths
            X_can = [X_can, np.array([float((x % seq_len) + 1) / seq_len for x in range(X_can.shape[0])]), np.full((X_can.shape[0],1), seq_len)]
            X_cdr = [X_cdr, np.array([float((x % seq_len) + 1) / seq_len for x in range(X_cdr.shape[0])]), np.full((X_cdr.shape[0],1), seq_len)]

        elif model_type in ["gru", "lstm"]:
            model_fun = rnn_model
            input_shape = (right_window+left_window+1, len(CHARS))
        elif model_type in ["cnn_pos"]:
            X_can = [X_can, np.array([float((x % seq_len) + 1) / seq_len for x in range(X_can.shape[0])])]
            X_cdr = [X_cdr, np.array([float((x % seq_len) + 1) / seq_len for x in range(X_cdr.shape[0])])]
            
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

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, cooldown=1, min_lr=0.0005)
        
        if n_clust == 0:
            hist_obj = model.fit(X_can, y_can, batch_size=batch_size, epochs=n_epochs, verbose=verbose, validation_data=(X_cdr, y_cdr), callbacks=[reduce_lr])
        else:
            labels, labels_cnt, min_cluster, min_cluster_size = [], -1, -1, -1
            if model_type == "dense":
                labels, labels_cnt, min_cluster, min_cluster_size = cluster(y_can.reshape((-1, seq_len)), n_clust)
            labels_new = []
            for x in labels:
                labels_new.extend([x] * seq_len)
            labels = labels_new
            
            if fading:
                weight_vec = np.array([np.log(min_cluster_size) / np.log(labels_cnt[x]) for x in labels])
                weight_vec = np.exp(np.log(weight_vec) / (200 ** .5))

                hist_obj = model.fit(X_can, y_can, sample_weight=weight_vec, batch_size=batch_size, epochs=n_epochs, verbose=verbose, validation_data=(X_cdr, y_cdr), callbacks=[reduce_lr])
            else:
                weight_vec = np.array([np.log(min_cluster_size) / np.log(labels_cnt[x]) for x in labels])

                hist_obj = model.fit(X_can, y_can, sample_weight=weight_vec, batch_size=batch_size, epochs=n_epochs, verbose=verbose, validation_data=(X_cdr, y_cdr), callbacks=[reduce_lr])
        
        hist[model_name] = hist_obj
        model_list[model_name] = model
        
        print(hist_obj.history["val_loss"][-1], end="\t")
        
        boot_loss_vec = bootstrap_cdr(model_list[model_name], X_cdr, y_cdr, seq_len)
        df_new = pd.DataFrame({"val_loss": boot_loss_vec, "model": model_name})
        
        print("(", np.mean(boot_loss_vec), ")")
        
        return model, pd.concat([df_loss, df_new])
    else:
        return model_list[model_name], None