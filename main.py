#Loading useful packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os.path
import sys
import argparse

#General purpose AI packages
from sklearn.cross_validation import train_test_split,KFold

#Keras packages
from keras.models import Sequential
from keras.layers import Dense, Activation

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
print(np.shape(X_tr))
print(np.shape(Y_tr))
############## PREPARING DATA ##################################################

############## BUILDING NN  ####################################################

model = Sequential()
model.add(Dense(units=20, input_dim=12))
model.add(Activation('relu'))
model.add(Dense(units=50))
model.add(Activation('sigmoid'))
model.add(Dense(units=50))
model.add(Activation('sigmoid'))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error',optimizer='adam')

############## BUILDING NN  ####################################################

############## TRAINING NN  ####################################################

n_epochs = 1000
model.fit(X_tr, Y_tr, epochs = n_epochs, batch_size = 100)

############## TRAINING NN  ####################################################

############## EVALUATING NN  ##################################################
score = model.evaluate(X_val,Y_val,batch_size=100)
Y_hat = model.predict(X_val)
print('\n Score: ',score)

plt.plot(Y_val[0:49], marker='^')
plt.plot(Y_hat[0:49], marker='o')
plt.show()
