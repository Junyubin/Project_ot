import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0' ## 44서버 GPU ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import *
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, Input, GaussianNoise, LSTM, RepeatVector, TimeDistributed, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Input, Reshape
from tensorflow.keras.models import Model, model_from_json, load_model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import categorical_accuracy, RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from utils import *

class corr_model:
    def __init__(self, config=None, mode=None, name=None):
        self.mode = mode
        self.name = name
        self.save_path_model = pwd + '/{}/{}'.format(config["common"]["path"], self.name)
        if not os.path.exists(self.save_path_model):
            os.makedirs(self.save_path_model)
        self.bs = config["batch_size"]
        self.lr = config["common"]["learning_rate"]
        self.optimizer = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        if self.mode=='train':
            self.n_neurons = config["n_neurons"]
            self.epoches = config["epoches"]
            self.x_datashape = config["x_datashape"]
#             self.y_datashape = config["y_datashape"]
        self.__set_neural_network()

    def __create_neural_network(self): 
        
        ae_model = Sequential([
            Dense(self.x_datashape[1], activation = 'relu', input_shape = (self.x_datashape[1],)),
            Dropout(0.3),
            Dense(16, activation = 'relu'),
            Dropout(0.3),
            Dense(32, activation = 'relu'),
            Dropout(0.3),
            Dense(self.x_datashape[1], activation = 'hard_sigmoid')
        ])        

        return ae_model
            
    def __rmse_custom(self, y_true, y_pred, axis=None):
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=axis))
    
    def rmse_custom(self, y_true, y_pred, axis=None):
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=range(axis, len(y_true.shape))))
    
    def __custom_mse(self, y_true, y_pred):
        y_true_masked = tf.boolean_mask(y_true, tf.math.not_equal(y_true, -1))
        y_pred_masked = tf.boolean_mask(y_pred, tf.math.not_equal(y_true, -1))
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred_masked - y_true_masked)))

    def __set_neural_network(self):
        if self.mode=='predict':
            self.neural_network = load_model(self.save_path_model, custom_objects={'__rmse_custom': self.__rmse_custom, 'optimizer': self.optimizer})
        else:
            self.neural_network = self.__create_neural_network()
            self.neural_network.compile(optimizer=self.optimizer, loss=self.__rmse_custom)       
        self.neural_network.summary()

    def optimize_nn(self, X=None, Y=None):
        early_stop = EarlyStopping(monitor = 'loss',  patience=15, verbose=1, min_delta=0.00001)
        ai_history = self.neural_network.fit(X, Y, epochs=self.epoches, batch_size=self.bs, shuffle=True, verbose=1, callbacks=[early_stop])
        self.neural_network.save(self.save_path_model)
        return 'MODEL HAS BEEN SAVED TO {}'.format(self.save_path_model), ai_history
    
    def predict(self, X=None):
        return self.neural_network.predict(X, batch_size=self.bs)
    
class lstm_model:
    def __init__(self, config=None, mode=None, name=None):
        self.mode = mode
        self.name = name
        self.save_path_model = pwd + '/{}/{}'.format(config["common"]["path"], self.name)
        if not os.path.exists(self.save_path_model):
            os.makedirs(self.save_path_model)
            
        self.bs = config["batch_size"]
        self.lr = config["common"]["learning_rate"]
        if self.mode=='train':
            self.n_neurons = config["n_neurons"]
            self.epoches = config["epoches"]
            self.x_datashape = config["x_datashape"]
            self.y_datashape = config["y_datashape"]
        self.__set_neural_network()

    def __create_neural_network(self):
        model = Sequential([
        LSTM(units=256, input_shape=(self.x_datashape[1], self.x_datashape[2])), #time stemp / feature
        Dropout(rate=0.2),
        RepeatVector(n=self.x_datashape[1]),
        LSTM(units=256, return_sequences=True, activation='tanh'),
        Dropout(rate=0.2),
        TimeDistributed(Dense(units=self.x_datashape[2], activation='relu'))
    ])        
        return model

    def __rmse_score(self, true, pred):    
        temp = np.mean(np.power(true - pred, 2), axis=1)
        return np.mean(temp, axis = 1)
    
    def __set_neural_network(self):
        if self.mode=='predict':
            self.neural_network = load_model(self.save_path_model)
        else:
            self.neural_network = self.__create_neural_network()
            self.neural_network.compile(loss='mse', optimizer='rmsprop')
        self.neural_network.summary()

    def optimize_nn(self, X=None, Y=None):
        early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1, min_delta=0.00001)
        ai_history = self.neural_network.fit(X, Y, epochs=self.epoches, batch_size=self.bs, verbose=1, callbacks=[early_stop])
        self.neural_network.save(self.save_path_model)
        return 'MODEL HAS BEEN SAVED TO {}'.format(self.save_path_model), ai_history

    def predict_rmse(self, X=None, Y=None):
        y_pred = self.neural_network.predict(X, batch_size=self.bs)
        return self.__rmse_score(Y, y_pred)