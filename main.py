#Loading useful packages
import numpy as np
import pandas as pd

import os.path
import sys
import argparse

#from keras.models import Sequential
#from keras.layers import Dense, Activation

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

############## READING FILES ###################################################

train = pd.read_table(DATAFILE,sep=',')
print(train.describe())

target = pd.read_table(TARGETFILE,sep=',')
print('Y',target.describe())
