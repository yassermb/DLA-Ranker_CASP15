#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 20:21:06 2022

@author: yasser
"""

import logging
import os
import sys
from os import path, mkdir, getenv, listdir, remove, system, stat
import pandas as pd
import numpy as np
from prody import *
import glob
import shutil
#import matplotlib.pyplot as plt
import seaborn as sns
from math import exp
from subprocess import CalledProcessError, check_call, call
import traceback
from random import shuffle, random, seed, sample
from numpy import newaxis
import matplotlib.pyplot as plt
import time
from prody import *
import collections
import scr
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
import subprocess
import load_data as load
import generate_cubes_reduce_channels_multiproc as reduce_channels
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(filename='manager.log', filemode='w', format='%(levelname)s: %(message)s', level=logging.DEBUG)
mainlog = logging.getLogger('main')
logging.Logger

sys.path.insert(1, '../lib/')
import tools as tl

#sys.path.insert(1, '../Test/')
import test as tst

comp_dir = 'conformations_directory'
target_comp = listdir(comp_dir)

bin_path = "./maps_generator"
v_dim = 24

def mapcomplex(file, pose_class, ch1, ch2, pair, pose):
    try:
        name = pair+'_'+str(pose)
        
        rec = parsePDB(file).select('protein').select('chain ' + ch1)
        rec.setChids('R')
        lig = parsePDB(file).select('protein').select('chain ' + ch2)
        lig.setChids('L')    
        
        writePDB(name+'_r.pdb', rec.toAtomGroup())
        writePDB(name+'_l.pdb', lig.toAtomGroup())
        writePDB(name+'_complex.pdb', rec.toAtomGroup() + lig.toAtomGroup())
        
        scr.get_scr(name+'_r.pdb', name+'_l.pdb', name+'_complex.pdb', name)
        
        rimcoresup = pd.read_csv(name+'_rimcoresup.csv', header=None, sep=' ')
        rec_regions = rimcoresup.loc[rimcoresup[4] == 'receptor']
        rec_regions = pd.Series(rec_regions[5].values, index=rec_regions[2]).to_dict()
        lig_regions = rimcoresup.loc[rimcoresup[4] == 'ligand']
        lig_regions = pd.Series(lig_regions[5].values, index=lig_regions[2]).to_dict()
        
        res_num2name_map_rec = dict(zip(rec.getResnums(),rec.getResnames()))
        res_num2name_map_lig = dict(zip(lig.getResnums(),lig.getResnames()))
        res_num2coord_map_rec = dict(zip(rec.select('ca').getResnums(),rec.select('ca').getCoords()))
        res_num2coord_map_lig = dict(zip(lig.select('ca').getResnums(),lig.select('ca').getCoords()))
        
        L1 = list(set(rec.getResnums()))
        res_ind_map_rec = dict([(x,inx) for inx, x in enumerate(sorted(L1))])
        L1 = list(set(lig.getResnums()))
        res_ind_map_lig = dict([(x,inx+len(res_ind_map_rec)) for inx, x in enumerate(sorted(L1))])
        
        res_inter_rec = [(res_ind_map_rec[x], rec_regions[x], x, 'R', res_num2name_map_rec[x], res_num2coord_map_rec[x]) 
                          for x in sorted(list(rec_regions.keys())) if x in res_ind_map_rec]
        res_inter_lig = [(res_ind_map_lig[x], lig_regions[x], x, 'L', res_num2name_map_lig[x], res_num2coord_map_lig[x])
                          for x in sorted(list(lig_regions.keys())) if x in res_ind_map_lig]
        
        reg_type =  list(map(lambda x: x[1],res_inter_rec))# + list(map(lambda x: x[1],res_inter_lig))
        res_name =  list(map(lambda x: [x[4]],res_inter_rec))# + list(map(lambda x: [x[4]],res_inter_lig))
        res_pos =  list(map(lambda x: x[5],res_inter_rec))# + list(map(lambda x: x[5],res_inter_lig))


        #Merge these two files!
        with open('resinfo','w') as fh_res:
            for x in res_inter_rec:
                fh_res.write(str(x[2])+';'+x[3]+'\n')

        with open('scrinfo','w') as fh_csr:
            for x in res_inter_rec:
                fh_csr.write(str(x[2])+';'+x[3]+';'+x[1]+'\n')
    
        if len(res_inter_rec) < 5 or len(res_inter_lig) < 5:
            raise Exception('There is no interface!')
        
        mapcommand = [bin_path, "--mode", "map", "-i", name+'_complex.pdb', "--native", "-m", str(v_dim), "-t", "167", "-v", "0.8", "-o", name+'_complex.bin']
        call(mapcommand)
        dataset_train = load.read_data_set(name+'_complex.bin')
        
        print(dataset_train.maps.shape)
        data_norm = dataset_train.maps
        
        X = np.reshape(data_norm, (-1,v_dim,v_dim,v_dim,173))
        
        #Reduce channels to 4
        X = reduce_channels.process_map(X)
        
        if X is None:
            remove(name+'_complex.bin')
            raise Exception('Dimensionality reduction failed!')
            
        y = [int(pose_class)]*len(res_inter_rec)
                
        _obj = (X,y,reg_type,res_pos,res_name,res_inter_rec)
        
        remove(name+'_complex.bin')
        remove(name+'_r.pdb')
        remove(name+'_l.pdb')
        remove(name+'_complex.pdb')
        remove(name+'_rimcoresup.csv')
        
    except Exception as e:
        remove(name+'_r.pdb')
        remove(name+'_l.pdb')
        remove(name+'_complex.pdb')
        remove(name+'_rimcoresup.csv')
        logging.info("Bad interface!" + '\nError message: ' + str(e) + 
                      "\nMore information:\n" + traceback.format_exc())
        return None
    
    return _obj


def process_targetcomplex(targetcomplex, comp_dir, report_dict):
    try:
        logging.info('Processing target' + targetcomplex + ' ...')
        
        predictions_file = open('predictions_SCR_' + targetcomplex , 'w')
        
        fold = 'test'
        predictions_file.write('Conf' + '\t' +
                               'Fold' + '\t' +
                               'Scores' + '\t' +
                               'Regions' + '\t' +
                               'Score' + '\t' +
                               'Time' + '\t' +
                               'Class' + '\t' +
                               'RecLig' + '\t' +
                               'ResNumber' + '\n')
        
        good_poses = list(map(lambda x: path.basename(x), glob.glob(path.join(comp_dir, targetcomplex, '*'))))
        print(good_poses)
        for pose in good_poses:
            logging.info('Processing conformation ' + pose + ' ...')
            file = path.join(comp_dir, targetcomplex, pose)
            all_chain_ids = set(parsePDB(file).select('protein').getChids())
            for ch1 in all_chain_ids:
                try:
                    test_interface = path.basename(pose).replace('.pdb', '') + '_' + ch1
                    logging.info('Processing interface ' + test_interface + ' ...')
                    
                    all_chain_ids_tmp = all_chain_ids.copy()
                    all_chain_ids_tmp.remove(ch1)
                    ch2 = ' '.join(list(all_chain_ids_tmp))
                    _obj = mapcomplex(file, '1', ch1, ch2, targetcomplex, test_interface)
                    
                    if _obj == None:
                        raise Exception('No map is generated!')
                    
                    X_test, y_test, reg_type, res_pos,_,info = _obj
                    all_scores, start, end = tst.predict(test_interface, X_test, y_test, reg_type, res_pos, info)
                    
                    if all_scores is None:
                        raise Exception("Prediction faile!")
                    
                    test_preds = all_scores.mean()
                    predictions_file.write(test_interface + '\t' +
                                        str(fold) + '\t' +
                                        ','.join(list(map(lambda x: str(x[0]), all_scores))) + '\t' +
                                        ','.join(reg_type) + '\t' +
                                        str(test_preds) + '\t' +
                                        str(end-start) + '\t' +
                                        str(y_test[0]) + '\t' +
                                        ','.join(list(map(lambda x: x[3], info))) + '\t' +
                                        ','.join(list(map(lambda x: str(x[2]), info))) + '\n')
                
                except:
                    predictions_file.write(test_interface + '\t' +
                                        str(fold) + '\t' +
                                        '0' + '\t' +
                                        'NA' + '\t' +
                                        '0' + '\t' +
                                        'NA' + '\t' +
                                        'NA' + '\t' +
                                        'NA' + '\t' +
                                        'NA' + '\n')
                    
        predictions_file.close()
    
    except Exception as e:
        logging.info("Bad target complex!" + '\nError message: ' + str(e) + 
                      "\nMore information:\n" + traceback.format_exc())

def manage_pair_files(use_multiprocessing):
    tc_cases = []
    for tc in target_comp:
        tc_cases.append((tc, comp_dir))
    report_dict = tl.do_processing(tc_cases, process_targetcomplex, use_multiprocessing)
    return report_dict

report_dict = manage_pair_files(False)