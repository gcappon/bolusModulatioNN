
# coding: utf-8

# In[1]:

#Loading useful packages
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os.path
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

#General purpose AI packages
from sklearn.cross_validation import train_test_split,KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process import GaussianProcess

#Keras packages
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, ActivityRegularization
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
from keras import regularizers 

############## LOSSHISTORY CALLBACK CLASS ######################################
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

DATAFILE = os.path.join('data','data.csv')
TARGETFILE = os.path.join('data','target.csv')
OUTDIR = os.path.join('results')

############## PREPARING DATA ##################################################

dataset_trans = pd.read_table(os.path.join('data','dataset_trans.csv'),sep=',')
target = np.asarray(dataset_trans['Y'])
pazienti = np.asarray(dataset_trans['subj'])
del dataset_trans['Y']
del dataset_trans['min_risk']

train = np.asarray(dataset_trans)
train_val_size = 0.8 #80% training+validation set and 20% test set
train_size = 0.7 #70% training set and 30% validation set
X_tr_val, X_te, Y_tr_val, Y_te = train_test_split(train, target, train_size=train_val_size, random_state=1)
X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr_val, Y_tr_val, train_size=train_size, random_state=1)

paz_tr_val = X_tr_val[:,0]
paz_tr = X_tr[:,0]
paz_val = X_val[:,0]
paz_te = X_te[:,0]
X_tr_val = X_tr_val[:,1:14]
X_tr = X_tr[:,1:14]
X_val = X_val[:,1:14]
X_te = X_te[:,1:14]
scaler = StandardScaler().fit(X_tr_val)
X_train = scaler.transform(X_tr_val)
np.save('X_te',X_te)
X_te = scaler.transform(X_te)

def train_nn(X_tr_val,Y_tr_val,X_te,Y_te):
    
    verbose = 1
    
    #Model callbacks
    filepath = os.path.join('results','weights.best.hdf5')
    mdlcheck = ModelCheckpoint(filepath, verbose=0, save_best_only=True)
    mdllosses = LossHistory()
    mdlstop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

    #Model fit
    n_epochs = 5000
    n_batch = 68
    performance_cv = []
    models = []
    
    model = Sequential()
    model.add(Dense(units=436, input_dim=np.shape(X_tr)[1], activity_regularizer=regularizers.l2(0)))
    model.add(Activation('relu'))
    model.add(Dropout(0.0881357))
    model.add(Dense(units=969,activity_regularizer=regularizers.l2(0)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.0315569))
    model.add(Dense(units=373,activity_regularizer=regularizers.l2(0)))
    model.add(Activation('sigmoid'))
    model.add(Dense(units=1))

    opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse',optimizer=opt)

    history = model.fit(X_train, Y_tr_val, validation_data = (X_te, Y_te),  epochs = n_epochs, batch_size = n_batch, callbacks = [mdlstop,mdlcheck,mdllosses],verbose = verbose)
        
    #Recalling best weights 
    model.load_weights(filepath)

    performance = min(mdllosses.val_losses)
    
    print('Obtained loss: ', performance)
    
    return model, performance

############## TRAIN MODEL  #############################################
model, score = train_nn(X_tr_val,Y_tr_val,X_te,Y_te)

############## EVALUATING RESULTS  #############################################
Y_te = np.squeeze(Y_te)
Y_NN = np.squeeze(model.predict(X_te))

#MSE
print('\n Score NN: ',mean_squared_error(Y_NN,Y_te))

"""
#Plot train and validation losses
#plt.plot(loss.losses)
#plt.plot(loss.val_losses)
#plt.show()

#Boxplot of the difference between actual values and estimates
data_to_plot = [Y_te-Y_NN]
plt.boxplot(data_to_plot)
plt.show()

#Histogram of the difference between actual values and estimates
plt.hist(data_to_plot,bins=40)
plt.show()

#Plot of the actual values and estimates
plt.plot(Y_te, marker='^')
plt.plot(Y_NN, marker='o')
plt.show()
"""
np.save('Y_NN',Y_NN)
np.save('Y_te',Y_te)
np.save('paz_te',paz_te)


