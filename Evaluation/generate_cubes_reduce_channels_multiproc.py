#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:00:07 2022

@author: mohseni
"""

import numpy as np
channels = {'ALA':['C','N','O','CA','CB'], 
            'ARG':['C','N','O','CA','CB','CG','CD','NE','CZ','NH1','NH2'], 
            'ASN':['C','N','O','CA','CB','CG','ND2','OD1'], 
            'ASP':['C','N','O','CA','CB','CG','OD1','OD2'], 
            'CYS':['C','N','O','CA','CB','SG'], 
            'GLN':['C','N','O','CA','CB','CG','CD','NE2','OE1'], 
            'GLU':['C','N','O','CA','CB','CG','CD','OE1','OE2'], 
            'GLY':['C','N','O','CA'], 
            'HIS':['C','N','O','CA','CB','CG','CD2','ND1','CE1','NE2'], 
            'ILE':['C','N','O','CA','CB','CG1','CG2','CD1'], 
            'LEU':['C','N','O','CA','CB','CG','CD1','CD2'], 
            'LYS':['C','N','O','CA','CB','CG','CD','CE','NZ'], 
            'MET':['C','N','O','CA','CB','CG','SD','CE'], 
            'PHE':['C','N','O','CA','CB','CG','CD1','CD2','CE1','CE2','CZ'], 
            'PRO':['C','N','O','CA','CB','CG','CD'], 
            'SER':['C','N','O','CA','CB','OG'], 
            'THR':['C','N','O','CA','CB','CG2','OG1'], 
            'TRP':['C','N','O','CA','CB','CG','CD1','CD2','CE2','CE3','NE1','CZ2','CZ3','CH2'], 
            'TYR':['C','N','O','CA','CB','CG','CD1','CD2','CE1','CE2','CZ','OH'], 
            'VAL':['C','N','O','CA','CB','CG1','CG2']}

v_dim = 24
n_channels = 4 + 4 + 2

all_channels = []
for aa, a_vector in channels.items():
    all_channels += a_vector
    
C_index, O_index, N_index, S_index = [], [], [], []
for i,a in enumerate(all_channels):
    if a[0] == "C":
        C_index.append(i)
    if a[0] == "O":
        O_index.append(i)
    if a[0] == "N":
        N_index.append(i)
    if a[0] == "S":
        S_index.append(i)
        
def process_map(X):
    try:
        X_new = np.zeros(X.shape[:-1] + tuple([n_channels]))
        
        X_new[:,:,:,:,0] = X[:,:,:,:,C_index].sum(axis=4)
        X_new[:,:,:,:,1] = X[:,:,:,:,N_index].sum(axis=4)
        X_new[:,:,:,:,2] = X[:,:,:,:,O_index].sum(axis=4)
        X_new[:,:,:,:,3] = X[:,:,:,:,S_index].sum(axis=4)
    
        for i in range(6):
            X_new[:,:,:,:,i+4] = X[:,:,:,:,167+i]
        
    except:
        return None
    return X_new