from __future__ import print_function, division

from collections import Counter
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid") 

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, BatchNormalization, concatenate, Input, LSTM, GRU, Conv1D, add, Flatten
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger

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
    
    def __init__(self, folderpath, seq_len, train_data, test_data, input_shape, data_name):
        self._folderpath = "models/" + folderpath
        if not os.path.exists(self._folderpath):
            os.makedirs(self._folderpath)
        print("[Beholder] Working folder:", self._folderpath)
        
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
        
        self._best_model = ""
        self._best_model_err = 1000
        
        print("[Beholder] Train data samples:", self._y_train.shape[0])
        print("[Beholder] Test data samples:", self._y_test.shape[0])
        
    
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
    
    
    def train(self, n_epochs, batch_size=16, verbose=0, lr=0):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, cooldown=0, min_lr=0.00002)
        # early_stop = EarlyStopping(patience=5)
        print("[Beholder] Training...")
        for model_name in self.models:
            print(" --", model_name, end="\t")
            
            if lr:
                self.models[model_name].optimizer.lr.set_value(lr)
            
            self.history[model_name] = self.models[model_name].fit(self._X_train, self._y_train, 
                                                                   batch_size=batch_size, verbose=verbose, epochs=n_epochs, 
                                                                   validation_data=(self._X_test, self._y_test), callbacks=[reduce_lr, CSVLogger(self._folderpath + "/" + model_name + ".log")])
            print(self.history[model_name].history["val_loss"][-1], end="\t")
            print("(", self.test(model_name), ")")
            
            if self.history[model_name].history["val_loss"][-1] < self._best_model_err:
                self._best_model_err = self.history[model_name].history["val_loss"][-1]
                self._best_model = model_name
            
            self.save_model(model_name)
    
    
    def test(self, model_name):
        boot_loss_vec = bootstrap_cdr(self.models[model_name], self._X_test, self._y_test, self._seq_len)
        df_new = pd.DataFrame({"val_loss": boot_loss_vec, "model": model_name})
        self.test_df = pd.concat([self.test_df, df_new])
        return np.mean(boot_loss_vec)
    
    
    def plot_loss(self, starting_from=50, smooth_window=3, smooth_step=1):
        def smooth(vec):
            res = []
            window = smooth_window
            step = smooth_step
            for i in range(window, len(vec)-window, step):
                res.append(np.mean(vec[i-window:i+window+1]))
            return res
        
        fig = plt.figure()
        fig.set_figwidth(18)
        fig.set_figheight(10)

        gs = plt.GridSpec(2,4) # 2 rows, 3 columns
        ax1 = fig.add_subplot(gs[0,:2]) # First row, first column
        ax2 = fig.add_subplot(gs[0,2:4]) # First row, second column
        ax3 = fig.add_subplot(gs[1,2:4]) # First row, third column

        plt.gcf().subplots_adjust(bottom=0.4)

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(self.models))]

        for i, key in enumerate(sorted(self.models.keys(), key=lambda x:x[0])):
            # ax1.plot(np.log2(smooth(self.history[key].history["loss"][starting_from:])), label=key, c=colors[i])
            # ax2.plot(np.log2(smooth(self.history[key].history["val_loss"][starting_from:])), label=key, c=colors[i])
            ax1.plot(smooth(self.history[key].history["loss"][starting_from:]), label=key, c=colors[i])
            ax2.plot(smooth(self.history[key].history["val_loss"][starting_from:]), label=key, c=colors[i])

        best_models_loss = self.test_df.sort_values("model")
        ax3 = sns.boxplot(x = "val_loss", y = "model", data = best_models_loss, palette="rainbow")

        ax1.set_title("loss")
        ax2.set_title("val")
        ax3.set_title("boostrapped loss")
        ax1.legend(prop={'size':4})
        ax1.legend(bbox_to_anchor=(.9, -1.1), loc='lower right', ncol = 1)

        fig.tight_layout()

        plt.savefig(self._folderpath + "/loss.png")
        
        
    def plot_pred(self, model_name):
        def plot(y, pred, ax, title, colors=["black", "red"]):
            y_true = y.reshape((-1, self._seq_len))
            y_pred = pred.reshape(y_true.shape)
            for i in range(len(y_true)):
                ax.plot(range(self._seq_len), y_true[i,:], c=colors[0], alpha=.5, label="real")
                ax.plot(range(self._seq_len), y_pred[i,:], c=colors[1], linestyle="dotted", alpha=.8, label="pred")
            ax.set_title(title)

        fig, ax = plt.subplots(nrows=1, ncols=2)
        fig.set_figwidth(16)

        plot(self._y_test, self.models[model_name].predict(self._X_test), ax[0], "Test data")

        plot(self._y_train, self.models[model_name].predict(self._X_train), ax[1], "Train data")

        plt.savefig(self._folderpath + "/pred.png")
        
        
    def save(self, starting_from=50, smooth_window=3, smooth_step=1):
        # for name in self.models:
        #     self.save_model(name)
        self.plot_loss(starting_from, smooth_window, smooth_step)
        self.plot_pred(self._best_model)
        
    
    def save_model(self, name):
        self.models[name].save(self._folderpath + "/" + name + ".h5")
        self.models[name].save_weights(self._folderpath + "/" + name + ".weights.h5")
        np.savetxt(self._folderpath + "/" + name + ".train.pred.txt", self.models[name].predict(self._X_train)) 
        np.savetxt(self._folderpath + "/" + name + ".test.pred.txt", self.models[name].predict(self._X_test)) 
        
    
    def load(self, folder):
        for filename in os.listdir(folder):
            if filename.endswith(".h5") and filename.find("weight") == -1:
                model_name = filename[:filename.find(".h5")]
                self.models[model_name] = load_model(folder + "/" + filename)
                print("[Beholder] Load", model_name, "from", folder + "/" + filename)
    

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