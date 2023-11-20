from __future__ import print_function, division

import math, time, os, sys, h5py, random, shutil, torch
from math import sqrt
import numpy as np
from collections import defaultdict
from collections import Counter
from termcolor import colored
from sklearn.metrics import mean_squared_error as MSE

#For "remove" option: set all label values to 0
#  --------------------- Helper functions for simulation --------------------- #  
#shift label, mask, and time before adding measurements
def shift_back(ptvar): 
    ptvar = np.insert(ptvar, 0, 0)
    ptvar = np.delete(ptvar, -1)
    
    return ptvar

#shift label and mask to the original form
def shift_forward(ptvar): 
    ptvar = np.delete(ptvar, 0)
    ptvar = np.insert(ptvar, len(ptvar), 0)

    return ptvar

def get_ptlist(dataset):
    #dataset that will be changed
    ptlist = []
    #get the name of every tensor
    keylist = list(dataset['0'].keys())
    #sort the batch index
    idxl = list(dataset.keys())
    idxl_int = [int(i) for i in idxl]
    idxl_int = sorted(idxl_int)
    idxl_str = [str(i) for i in idxl_int]
    
    for b in idxl_str: #27 batchs
        for i in range(dataset[b]['CatTensor'].shape[0]): #50 pts in every batch
            pt=[]
            for k in keylist:
                pt.append(dataset[b][k][i])
            ptlist.append(pt)
    
    return ptlist, idxl_str


def get_new_time(d, t, loc, sim_type):
    dc = np.where(d==0, 0, t) #dose count
    dn_idx = np.nonzero(d)[0] #index list of all the nonzero values in the dose array
    #new time steps to add
    tadd = []
    
    #function to get the tadd list
    def get_tadd(i,tadd,loc):
        if loc == 'peak':
            if d[i] > 1: #when dose > 1, peak is 3hrs after the dose 
                tadd.append(dc[i] + 3) 
            elif d[i] != 0 and d[i] <= 1:
                tadd.append(dc[i] + 2)
        elif loc == 'trough':
            if i > 0 and d[i] != 0:
                tadd.append(dc[i] - 1) #trough is 1hr before the dose
        elif loc == 'both':
            if i > 0 and d[i] != 0:
                tadd.append(dc[i] - 1)
                
            if d[i] > 1:
                tadd.append(dc[i] + 3) 
            elif d[i] != 0 and d[i] <= 1:
                tadd.append(dc[i] + 2)           
        
    if sim_type == 'add_all':
        for i in range(len(d)):
            get_tadd(i, tadd, loc)
    elif sim_type == 'add_half':
        if len(dn_idx) <= 2: #if there are 1 or 2 doses, only add at the first dose
            get_tadd(dn_idx[0], tadd, loc)
        else: #if this patient has more than 2 doses, add new time points for the early half doses
            for j in range(int(len(dn_idx)/2)):
                get_tadd(dn_idx[j], tadd, loc)
    else:
        raise ValueError("Invalid 'sim_type' parameter. Must be 'add_half' or 'add_all' to get new time steps.")

    tnew = np.sort(np.concatenate((t, np.array(tadd))))
    #index list for the same elements        
    tidx = np.arange(tnew.shape[0])[np.in1d(tnew,t)]
    #index list for different elements 
    tidx_n = np.arange(tnew.shape[0])[~np.in1d(tnew,t)]
    tnew = tnew.astype(np.float32)
    
    return tnew, tidx, tidx_n

## two functions for the "move" option
def move_label_to_peak(cat, cont, dose, time, label):
    new_label = label.copy()
    new_time = time.copy()
    
    new_label = shift_back(new_label)
    for i in range(len(label)):
        if label[i] != 0:
            last_dose_index = np.where(dose[:i+1] != 0)[0][-1]  # Find the index of the last nonzero dose before the label
            time_diff = time[i] - time[last_dose_index] # Time difference between the last dose and the label
            
            if dose[last_dose_index] > 1 and time_diff > 3:
                new_time[last_dose_index + 1] = time[last_dose_index] + 3  # Set the label time to the peak time
                # Set the label value to the predicted value at the peak time
                new_label[last_dose_index + 1] = 0

            elif dose[last_dose_index] <= 1 and time_diff > 2:
                new_time[last_dose_index + 1] = time[last_dose_index] + 2
                new_label[last_dose_index + 1] = 0
            
            #get new cat and cont
            #set a temprary list for to add necessary information to the CatTensor
            #contains codes that occur in every visit for this patient
            cat_templ = [k for k, v in Counter(cat.flatten().tolist()).items() if v==cat.shape[0]] 
            #extend the list the same length as other sequences
            cat_templ.extend(0 for _ in range(cat.shape[1]-len(cat_templ)))
            
            # Move the corresponding elements in Cat and Cont to the peak
            cat = np.delete(cat, i, axis=0) # Delete the previous element
            cat = np.insert(cat, last_dose_index + 1, cat_templ, axis = 0) # Insert the same element at peak
            cont = np.delete(cont, i, axis=0)
            cont = np.insert(cont, last_dose_index + 1, last_dose_index, axis = 0)

            # Set the previous nonzero label to 0
            if last_dose_index + 1 != i:
                new_label[i] = 0
    
    new_label = shift_forward(new_label)
    return cat, cont, new_time, new_label


def move_label_to_trough(cat, cont, dose, time, label):
    new_label = label.copy()
    new_time = time.copy()

    new_label = shift_back(new_label)
    for i in range(len(label)):
        if label[i] != 0:
            if len(np.where(dose[i:] != 0)[0]) > 0: # Make sure the next dose exists
                next_dose_index = np.where(dose[i:] != 0)[0][0]  + i # Find the index of the next nonzero dose after the label
                last_dose_index = np.where(dose[:i+1] != 0)[0][-1]
                time_diff = time[next_dose_index] - time[i]
                
                if time_diff > 1:
                    new_time[next_dose_index - 1] = time[next_dose_index] - 1  # Set the label time to the trough time
                    new_label[next_dose_index - 1] = 0
                    
                    #move Cat and Cont
                    cat_templ = [k for k, v in Counter(cat.flatten().tolist()).items() if v==cat.shape[0]]
                    cat_templ.extend(0 for _ in range(cat.shape[1]-len(cat_templ)))

                    cat = np.delete(cat, i, axis=0) # Delete the previous element
                    cat = np.insert(cat, next_dose_index - 1, cat_templ, axis = 0) # Insert the same element at trough
                    cont = np.delete(cont, i, axis=0)
                    cont = np.insert(cont, next_dose_index - 1, next_dose_index - 2, axis = 0)

                    # Set the previous nonzero label to 0
                    if last_dose_index + 1 != i:
                        new_label[i] = 0

    new_label = shift_forward(new_label)
    return cat, cont, new_time, new_label

def sim_data(ptlist, loc, sim_type, real_data):
    #add measurements (change labels, and also add values to other tensors)
    for pt in range(len(ptlist)):
        #-------------------Step 1: get all the information from the patient-------------------#
        cat = ptlist[pt][0]      #CatTensor
        cont = ptlist[pt][1]     #ContTensor
        d = ptlist[pt][2]        #DoseTensor
        lb = ptlist[pt][3]        #LabelTensor
        m = ptlist[pt][5]        #MaskTensor
        ptid = ptlist[pt][6]     #PtList
        td = 24 * ptlist[pt][7]  #TimeDiffTensor
        #shift leabl and mask tensors one time step back
        lb = shift_back(lb)
        m = shift_back(m)        
        #time and dose count will be used to check which time step to add measurements
        t = np.cumsum(td)
        t = shift_back(t)
    
        #------------------- Step 2: update the data for every patient based on different options -------------------#        
        if loc in ['peak', 'trough', 'both'] and sim_type in ['add_half', 'add_all']:
            tnew, tidx, tidx_n = get_new_time(d, t, loc, sim_type)
            
            #update dose
            dnew = np.zeros(len(tnew))
            for dose in range(len(d)):
                dnew[tidx[dose]]=d[dose]
            
            #update label
            lnew = np.zeros(len(tnew))
            #put the original lb values into the new array
            for l in range(len(lb)):
                lnew[tidx[l]]=lb[l]
            
            #get new cat and cont
            #set a temprary list for to add necessary information to the CatTensor
            #contains codes that occur in every visit for this patient
            templ = [k for k, v in Counter(cat.flatten().tolist()).items() if v==cat.shape[0]] 
            #extend the list the same length as other sequences
            templ.extend(0 for _ in range(cat.shape[1]-len(templ)))
            
            #use this list to check the location to add elements
            idxl=[]
            for tempidx in range(len(tidx)):
                idxl.append(tidx[tempidx]-tempidx)
        
            for i in range(1, len(idxl)):
                if idxl[i]-idxl[i-1]==1: #there is one measurment added
                    cat = np.insert(cat, i, templ, axis = 0)
                    cont = np.insert(cont, i, i-1, axis = 0)
                elif idxl[i]-idxl[i-1]>1: #more than one measurements are added continuously
                    for j in range(1, idxl[i]-idxl[i-1]+1):
                        cat = np.insert(cat, i, templ, axis = 0)
                        cont = np.insert(cont, i, i-1, axis = 0)
            
            #update related arrays
            dnew = dnew.astype(np.float32)
            #shift label and mask tensors back to normal
            mnew = np.where(lnew==0, 0, 1)
            lnew = shift_forward(lnew)
            lnew = lnew.astype(np.float32)
            mnew = shift_forward(mnew)
            
            td_new = np.diff(tnew)
            td_new = np.insert(td_new, len(td_new), 0)
            
            ptlist[pt][0] = cat
            ptlist[pt][1] = cont
            ptlist[pt][2] = dnew
            ptlist[pt][3] = lnew
            ptlist[pt][4] = len(lnew)
            ptlist[pt][5] = mnew
            ptlist[pt][7] = td_new/24
    
    return ptlist


def update_batch(ptlist):
    
    #devide the original list to batches, prepare to replace the old data
    def dev(ptlist):
        bs = 50 #batch size
        pt_batch = [ptlist[i:i + bs] for i in range(0, len(ptlist), bs)]
        return pt_batch
    
    ptb = dev(ptlist)
    
    #get the max visit for every batch
    ptlen = []
    maxtemp = []
    for batch in range(len(ptb)): #27 batches
        for patient in range(len(ptb[batch])):
            ptlen.append(ptb[batch][patient][4])
            
    blen = dev(ptlen)
    for l in blen:
        maxtemp.append(max(l))
    
    #zero padding to remain the same shape
    for i in range(len(ptb)): #27 batches
        for pt in range(len(ptb[i])):        
            ptb[i][pt][0] = np.pad(ptb[i][pt][0], ((0, maxtemp[i]-len(ptb[i][pt][0])), (0, 0)), 'constant')
            ptb[i][pt][1] = np.pad(ptb[i][pt][1], ((0, maxtemp[i]-len(ptb[i][pt][1])), (0, 0)), 'constant')
            ptb[i][pt][2] = np.pad(ptb[i][pt][2], (0, maxtemp[i]-len(ptb[i][pt][2])), 'constant')
            ptb[i][pt][3] = np.pad(ptb[i][pt][3], (0, maxtemp[i]-len(ptb[i][pt][3])), 'constant')
            ptb[i][pt][5] = np.pad(ptb[i][pt][5], (0, maxtemp[i]-len(ptb[i][pt][5])), 'constant')
            ptb[i][pt][7] = np.pad(ptb[i][pt][7], (0, maxtemp[i]-len(ptb[i][pt][7])), 'constant')
    
    #collect the data by different type of variables, not pt (to replace the original tensors)
    catl, contl, dl, ll, ml, tdl = [], [], [], [], [], []
    ptl, vl, vcl = [], [], [] #3 tensors that didn't change
    
    for b in range(len(ptb)): #27 batches
        for p in range(len(ptb[b])):
            catl.append(ptb[b][p][0])
            contl.append(ptb[b][p][1])
            dl.append(ptb[b][p][2])
            ll.append(ptb[b][p][3])
            ml.append(ptb[b][p][5])
            tdl.append(ptb[b][p][7])
            
            ptl.append(ptb[b][p][6])
            vl.append(ptb[b][p][8])
            vcl.append(ptb[b][p][9])
    
    bcat = dev(catl)
    bcont = dev(contl)
    bd = dev(dl)
    bl = dev(ll)
    bm = dev(ml)
    btd = dev(tdl)
    bpt = dev(ptl)
    bv = dev(vl)
    bvc = dev(vcl)
        
    #remain the same format as original tensors
    def toarray(batchlist):
        for i in range(len(batchlist)):
            batchlist[i] = np.array(batchlist[i])
        return batchlist
    
    bcat = toarray(bcat)
    bcont = toarray(bcont)
    bd = toarray(bd)
    bl = toarray(bl)
    bm = toarray(bm)
    btd = toarray(btd)
    blen = toarray(blen)
    bpt = toarray(bpt)
    bv = toarray(bv)
    bvc = toarray(bvc)
    
    return bcat, bcont, bd, bl, bm, btd, blen, bpt, bv, bvc


#  --------------------- Simulation functions --------------------- #
#main simulation functions to get new values
#simulation function for sim_type == 'move'
#when sim_type is "move", we assume real data are removed and loc can't be "both"
def simulate_move(dataset, loc, sim_type = 'move', real_data = 'remove'):
        
    # Iterate over the groups
    for group_name in dataset.keys():
        group = dataset[group_name]
        
        #------------------- Step 1: get all the information -------------------#    
        #the key feature
        td = group['TimeDiffTensor'][:]
        td *= 24
        
        #other features that may change based on td
        cont = group['ContTensor'][:]
        cat = group['CatTensor'][:]    
        lb = group['LabelTensor'][:]
        m = group['MaskTensor'][:]
        d = group['DoseTensor'][:]
        length = group['LengList'][:]
        
        #won't change for any options
        pt_list = group['PtList'][:]
        
        #------------------- Step 2: update the data for every patient based on options -------------------#
        if len(pt_list) == 50:
            for idx in range(len(pt_list)):        
                ptid = pt_list[idx]
                t = np.cumsum(td[idx])
                t = shift_back(t)
                
                #if sim_type == 'move': #the sequence length will remain the same
                if loc == 'peak':
                    cat[idx], cont[idx], time, label = move_label_to_peak(cat[idx], cont[idx], d[idx], t, lb[idx])
                elif loc == 'trough':
                    cat[idx], cont[idx], time, label = move_label_to_trough(cat[idx], cont[idx], d[idx], t, lb[idx])
                else:
                    raise ValueError("Invalid 'loc' parameter. Must be 'peak' or 'trough' to get new time steps for the sim_type 'move'.")
                
                #update related arrays
                td_new = np.diff(time)
                td_new = np.insert(td_new, len(td_new), 0)
                td[idx] = td_new/24
    
                #shift label and mask tensors back to normal
                mnew = np.where(label==0, 0, 1)
                #label = shift_forward(label)
                label = label.astype(np.float32)
                lb[idx] = label
                #mnew = shift_forward(mnew)
                m[idx] = mnew
        
        group['ContTensor'][:] = cont
        group['CatTensor'][:] = cat   
        group['LabelTensor'][:] = lb
        group['MaskTensor'][:] = m
        group['TimeDiffTensor'][:] = td

    return dataset


def simulate_add(dataset, loc, sim_type, real_data):
    
    #get patient data as a list
    ptlist, idxl_str = get_ptlist(dataset)
    #simulation
    new_ptlist = sim_data(ptlist, loc, sim_type, real_data)
    #update the data with new batches
    bcat, bcont, bd, bl, bm, btd, blen, bpt, bv, bvc = update_batch(new_ptlist)
    
    #delete original batch and update them using new data
    for b in idxl_str:
        del dataset[b]['CatTensor']
        dataset[b].create_dataset('CatTensor', data = bcat[int(b)])
        
        del dataset[b]['ContTensor']
        dataset[b].create_dataset('ContTensor', data = bcont[int(b)])
        
        del dataset[b]['DoseTensor']
        dataset[b].create_dataset('DoseTensor', data = bd[int(b)])
        
        del dataset[b]['LabelTensor']
        dataset[b].create_dataset('LabelTensor', data = bl[int(b)])
        
        del dataset[b]['LengList']
        dataset[b].create_dataset('LengList', data = blen[int(b)])
        
        del dataset[b]['MaskTensor']
        dataset[b].create_dataset('MaskTensor', data = bm[int(b)])
        
        del dataset[b]['TimeDiffTensor']
        dataset[b].create_dataset('TimeDiffTensor', data = btd[int(b)])
        
    
        del dataset[b]['PtList']
        dataset[b].create_dataset('PtList', data = bpt[int(b)])
        
        del dataset[b]['VTensor']
        dataset[b].create_dataset('VTensor', data = bv[int(b)])
        
        del dataset[b]['VancoClTensor']
        dataset[b].create_dataset('VancoClTensor', data = bvc[int(b)])
    
    return dataset
    
    