#Hyperas packages (hyperparameter tuning utility)
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

#Loading useful packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import sys
import argparse
from pandas.tools.plotting import parallel_coordinates

#General purpose AI packages
from sklearn.cross_validation import train_test_split,KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, mean_squared_error

#Keras packages
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, ActivityRegularization
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import RMSprop

############## LOSSHISTORY CLASS ###############################################
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

############## PREPARING DATA ##################################################
def data():
    DATAFILE = os.path.join('data','data.csv')
    TARGETFILE = os.path.join('data','target.csv')
    OUTDIR = os.path.join('results')

    train = pd.read_table(DATAFILE,sep=',')
    #parallel_coordinates(train,'G_c')
    #plt.show()
    train = np.asarray(train)

    target = pd.read_table(TARGETFILE,sep=',')
    target = np.asarray(target)

    train_size = 0.7 #70% a training set e 30% a validation set
    X_tr, X_val, Y_tr, Y_val = train_test_split(train, target, train_size=train_size, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)
    return X_tr, Y_tr, X_val, Y_val

############## BUILDING NN  ####################################################
def model(X_tr, Y_tr, X_te, Y_te):
    model = Sequential()
    model.add(Dense(units=20, input_dim=12))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense({{choice([10,20,50])}}))
    model.add(Activation('sigmoid'))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error',optimizer=opt)

    ############## TRAINING NN  ################################################
    filepath = os.path.join('results','weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    mdlcheck = ModelCheckpoint(filepath, verbose=0, save_best_only=True)
    mdlloss = self.LossHistory()

    n_epochs = 100
    n_batch = 16
    history = model.fit(X_tr, Y_tr, validation_data = (X_te, Y_te),  epochs = n_epochs, batch_size = n_batch, callbacks = [mdlcheck,mdlloss])

    score = np.min(mdlloss.val_losses)
    return {'score': score, 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
X_train, Y_train, X_test, Y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)
