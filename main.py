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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop

############## BEGIN PARSING INPUT #############################################
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
############## END PARSING INPUT ###############################################

############## BEGIN PREPARING DATA ############################################
train = pd.read_table(DATAFILE,sep=',')
train = np.asarray(train)

target = pd.read_table(TARGETFILE,sep=',')
target = np.asarray(target)

train_size = 0.7 #70% a training set e 30% a validation set
X_tr, X_val, Y_tr, Y_val = train_test_split(train, target, train_size=train_size, random_state=0)
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_val = scaler.transform(X_val)
############## END PREPARING DATA ##############################################

############## BEGIN BUILDING NN  ##############################################
model = Sequential()
model.add(Dense(units=20, input_dim=12))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(units=50))
model.add(Activation('sigmoid'))
model.add(Dense(units=1))
model.add(Activation('linear'))
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error',optimizer=opt)
############## END BUILDING NN  ################################################

############## BEGIN TRAINING NN  ##############################################
filepath = os.path.join('results','weights.{epoch:02d}-{val_loss:.2f}.hdf5')
mdlcheck = ModelCheckpoint(filepath, verbose=1, save_best_only=True)


n_epochs = 500
n_batch = 16
history = model.fit(X_tr, Y_tr, validation_data = (X_val, Y_val),  epochs = n_epochs, batch_size = n_batch, callbacks = [mdlcheck])
############## END TRAINING NN  ################################################

############## BEGIN EVALUATING RESULTS  #######################################
Y_val = np.squeeze(Y_val)
Y_NN = np.squeeze(model.predict(X_val))

#MSE of the two methods
print('\n Score NN: ',mean_squared_error(Y_NN,Y_val))

#Boxplot of the difference between actual values and estimates
data_to_plot = [Y_val-Y_NN]
plt.boxplot(data_to_plot)
plt.show()

#Plot of the actual values and estimates
plt.plot(Y_val, marker='^')
plt.plot(Y_NN, marker='o')
plt.show()

plt.plot(history.losses)
plt.show()
############## END EVALUATING RESULTS  #########################################

#TBD: Regularization, RFE, GridSearchCV, best model selection

#Hyperparameters:
#   * First layer: # nodes, activation function, regularization, dropout %
#   * Second layer: # nodes, activation function, regularization
#   * Output layer: regularization
#   * Compile: loss function
#   * optimizer: lr, rho, epsilon, decay
#   * Fit: batch_size, epochs, learning rate
