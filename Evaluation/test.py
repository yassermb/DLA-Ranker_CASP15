#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:09:56 2020

@author: yasser
"""

import logging
import os
import sys
import gc
from os import path, mkdir, getenv, listdir, remove, system, stat
import pandas as pd
import numpy as np
import glob

import seaborn as sns
from math import exp
from subprocess import CalledProcessError, check_call
import traceback
from random import shuffle, random, seed, sample
from numpy import newaxis
import matplotlib.pyplot as plt
import time

from numpy import asarray
from sklearn.preprocessing import OneHotEncoder

import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Dot
from tensorflow.keras.backend import ones, ones_like
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(1, '../lib/')
import tools as tl

print('Your python version: {}'.format(sys.version_info.major))
USE_TENSORFLOW_AS_BACKEND = True
 
if USE_TENSORFLOW_AS_BACKEND:
    os.environ['KERAS_BACKEND'] = 'tensorflow'
else:
    os.environ['KERAS_BACKEND'] = 'theano'
if tl.FORCE_CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
if USE_TENSORFLOW_AS_BACKEND == True:
    import tensorflow as tf
    print('Your tensorflow version: {}'.format(tf.__version__))
    if not tl.FORCE_CPU:
        print("GPU : "+tf.test.gpu_device_name())
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    import theano
    print('Your theano version: {}'.format(theano.__version__))
    
seed(int(np.round(np.random.random()*10)))
#################################################################################################

v_dim = 24

#model = load_model(path.join('../Models', 'Dockground', '0_model'))
model = load_model(path.join('../Models', 'ALL_20_model'))

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
onehot = encoder.fit(np.asarray([['S'], ['C'], ['R']]))

def predict(test_interface, X_test, y_test, reg_type, res_pos, info):
    try:
        print('Prediction for ' + test_interface)
        X_aux = encoder.transform(list(map(lambda x: [x], reg_type)))
        if len(X_test) == 0 or len(X_aux) != len(X_test):
            #raise Exception("Not compatible features!")
            return None, None, None
            
        start = time.time()
        all_scores = model.predict([X_test, X_aux], batch_size=X_test.shape[0])
        end = time.time()
        _ = gc.collect()
        
    except Exception as e:
        #logging.info("Bad target complex!" + '\nError message: ' + str(e) + 
        #             "\nMore information:\n" + traceback.format_exc())
        return None, None, None
    
    return all_scores, start, end