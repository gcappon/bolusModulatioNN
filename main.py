#Loading useful packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import sys
import argparse

#General purpose AI packages
from sklearn.cross_validation import train_test_split,KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, mean_squared_error

#Keras packages
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, ActivityRegularization
from keras.callbacks import EarlyStopping

#XGBoost packages
import xgboost as xgb

############## PARSING INPUT ###################################################
class myArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(myArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg

parser = myArgumentParser(description='Running bolusModulatioNN...',fromfile_prefix_chars='@')
parser.add_argument('DATAFILE', type=str, help='Training datafile.')
parser.add_argument('TARGETFILE', type=str, help='Target values.')
parser.add_argument('OUTDIR', type=str, help='Output directory.')
parser.add_argument('--verbose', action='store_true', help='Run and print progress info.')

# Check on the number of input parameters
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

# Read input parameters
args = parser.parse_args()
DATAFILE = args.DATAFILE
TARGETFILE = args.TARGETFILE
OUTDIR = args.OUTDIR
verbose = args.verbose
############## PARSING INPUT ###################################################

############## PREPARING DATA ##################################################
train = pd.read_table(DATAFILE,sep=',')
train = np.asarray(train)

target = pd.read_table(TARGETFILE,sep=',')
target = np.asarray(target)

train_size = 0.7 #70% a training set e 30% a validation set
X_tr, X_val, Y_tr, Y_val = train_test_split(train, target, train_size=train_size, random_state=0)
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_val = scaler.transform(X_val)
############## PREPARING DATA ##################################################

############## BUILDING NN  ####################################################
model = Sequential()
model.add(Dense(units=20, input_dim=12))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(units=50))
model.add(Activation('sigmoid'))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error',optimizer='rmsprop')
############## BUILDING NN  ####################################################

############## TRAINING NN  ####################################################
n_epochs = 300
n_batch = 32
history = model.fit(X_tr, Y_tr, validation_data = (X_val, Y_val),  epochs = n_epochs, batch_size = n_batch)
############## TRAINING NN  ####################################################

############## PREPARING AND TRAINING XGB ######################################
model_trunc = Sequential()
model_trunc.add(Dense(units=20, input_dim=12,weights=model.layers[0].get_weights()))
model_trunc.add(Activation('relu'))
model_trunc.add(Dropout(0.1))
model_trunc.add(Dense(units=50,weights=model.layers[3].get_weights()))
model_trunc.add(Activation('sigmoid'))

act_tr = model_trunc.predict(X_tr)
act_val = model_trunc.predict(X_val)

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have
#much fun of fighting against overfit
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
params = {'booster': 'dart',
'max_depth': 3,
'min_child_weight':10,
'learning_rate':0.3,
'subsample':0.5,
'colsample_bytree':0.6,
'obj':'reg:logistic',
'n_estimators':1000,
'eta':0.3}
booster = xgb.train(params,xgb.DMatrix(act_tr,Y_tr))
############## PREPARING AND TRAINING XGB ######################################

############## EVALUATING RESULTS  #############################################
Y_val = np.squeeze(Y_val)
Y_NN = np.squeeze(model.predict(X_val))
Y_BOO = booster.predict(xgb.DMatrix(act_val))

#MSE of the two methods
print('\n Score NN: ',mean_squared_error(Y_NN,Y_val))
print('\n Score Booster: ',mean_squared_error(Y_BOO,Y_val))

#Boxplot of the difference between actual values and estimates
data_to_plot = [Y_val-Y_NN,Y_val-Y_BOO]
plt.boxplot(data_to_plot)
plt.show()

#Plot of the actual values and estimates
plt.plot(Y_val, marker='^')
plt.plot(Y_NN, marker='o')
plt.plot(Y_BOO, marker = '*')
plt.show()

#
############## EVALUATING RESULTS  #############################################

#TBD: Regularization, RFE, GridSearchCV, best model selection
