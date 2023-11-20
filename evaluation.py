from __future__ import print_function, division

import sys, h5py, math, torch
import pandas as pd
import numpy as np
from termcolor import colored
from torch import nn
from collections import defaultdict
from math import sqrt
from sklearn.metrics import mean_squared_error as MSE

from train import DataFromH5py


#  --------------------- Helper functions for the evaluation --------------------- #  
def shift_back(ptvar): 
    ptvar = np.insert(ptvar, 0, 0)
    ptvar = np.delete(ptvar, -1)
    
    return ptvar

#get results from the inference PKRNN-1CM model
def get_res_PKRNN_1CM(path, device, testloader): #path to the .pth file from 1CM model
    reout = defaultdict(dict)

    for batch in testloader:
        if batch[0].shape[0] == 50:
            model = torch.load(path, map_location=device)
            k, V, lb, output, d, td, _, _, LengList, PtList, _, _, = model(batch)
        
            k = k.cpu().detach().numpy()
            V = V.cpu().detach().numpy()
            lb = lb.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            d = d.cpu().detach().numpy()
            td = td.cpu().detach().numpy()
            
            for i in range(len(batch[0])):
                reout[PtList[i]] = (k[i], V[i], d[i], td[i], lb[i], output[i])
    
    return reout

#get results from the inference PKRNN_2CM model
def get_res_PKRNN_2CM(path, device, testloader): #path to the .pth file from 2CM model
    reout = defaultdict(dict)

    for batch in testloader:
        model = torch.load(path, map_location=device)
        eta1, eta2, eta3,eta4, K1, K2, V1, V2, lb, output, ccl, d, td, _, LengList, PtList, _= model(batch)
        
        eta1 = eta1.cpu().detach().numpy() #shape:torch.Size([bs, visit])
        eta2 = eta2.cpu().detach().numpy()
        eta3 = eta3.cpu().detach().numpy()
        eta4 = eta4.cpu().detach().numpy()
        ccl = ccl.cpu().detach().numpy()
        lb = lb.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        d = d.cpu().detach().numpy()
        td = td.cpu().detach().numpy()
        
        for i in range(len(batch[0])):
            reout[PtList[i]] = (eta1[i], eta2[i], eta3[i], eta4[i], ccl[i], d[i], td[i], lb[i], output[i])
        
    return reout

#set the non-peak (>3hrs) concentrations as 0
def peak_cnt(y, d):
    ynew = np.zeros(len(y))
    
    for i in range(len(d)-3):
        if d[i] > 1: #when dose > 1, peak is 3hrs after the dose 
            ynew[i+3] = y[i+3] 
        elif d[i] != 0 and d[i] <= 1:
            ynew[i+2] = y[i+2]

    return ynew

#only remain trough concentrations -- the last concentration before the next dose
def trough_cnt(y, d):
    ynew = np.zeros(len(y))
    for i in range(1, len(d)):
        if d[i] != 0 and len(np.where(d[i:] != 0)[0]) > 0: # Make sure the next dose exists
            next_dose_index = np.where(d[i:] != 0)[0][0]  + i # Find the index of the next nonzero dose after the label
            ynew[next_dose_index] = y[next_dose_index]

    return ynew

#set concentrations for both peak and trough
def peak_trough_cnt(y, d):
    ynew = np.zeros(len(y))
    
    for i in range(1, len(d)-3):
        if d[i] > 1 and len(np.where(d[i:] != 0)[0]) > 0: 
            #when there is a dose, set the value to non-zero
            ynew[i+3] = y[i+3]

            next_dose_index = np.where(d[i:] != 0)[0][0]  + i # Find the index of the next nonzero dose after the label
            ynew[next_dose_index] = y[next_dose_index]
        
        elif d[i] != 0 and d[i] <= 1 and len(np.where(d[i:] != 0)[0]) > 0: 
            ynew[i+2] = y[i+2]

            next_dose_index = np.where(d[i:] != 0)[0][0]  + i
            ynew[next_dose_index] = y[next_dose_index]

    return ynew

#calculate concentration by hour
def curve_concn(model_name, reout):
    result = defaultdict(dict)

    if model_name in ['underlying', 'PKRNN_2CM']:
        for pt in reout.keys():
            #-------------------Step 1: get all the information-------------------#
            eta1 = reout[pt][0]
            eta2 = reout[pt][1]
            eta3 = reout[pt][2]
            eta4 = reout[pt][3]
            ccl = reout[pt][4]
            d = reout[pt][5]
            lb = reout[pt][7]            
            output = reout[pt][8]            
            
            td = 24 * reout[pt][6]
            event_time = np.cumsum(td)
            event_time = shift_back(event_time)
            
            lb = shift_back(lb)
            output = shift_back(output)
            
            #-------------------Step 2: calculate concentration for every hour-------------------#
            tlist, clist, dlist = [], [], []
            CList, DList = [0], [0]
            
            for i in range(len(event_time)-1):
                tlist.append(event_time[i])
                dlist.append(d[i])
                clist.append(output[i])

                #PK parameters: k1, v1, v2, R
                v1 = 33.1 * np.exp(eta1[i])
                k1 = 3.96 * ccl[i] / 100 * np.exp(eta2[i])
                r = 1000 / v1
                v2 = 48.3 * np.exp(eta4[i])
                R = - 6.99 / 48.3 * np.exp(eta3[i])
                k2 = R * v2
    
                delta = np.sqrt((k1 - k2 - R * v1) ** 2 + 4 * k1 * R * v1) / v1
                lam1 = (((-k1 + k2 + R * v1) / v1) - delta) / 2
                lam2 = (((-k1 + k2 + R * v1) / v1) + delta) / 2
    
                C1 = r * (k2 / v1) / lam1 / delta
                C2 = -r * (k2 / v1) / lam2 / delta
                C3 = - (lam1 - R) / (k2 / v1)
                C4 = - (lam2 - R) / (k2 / v1)
                
                hrs=int(event_time[i+1]-event_time[i])
                
                for h in range(1, hrs+1):
                    cnt = event_time[i]+h
            
                    if cnt <= event_time[i+1]:
                        tlist.append(cnt)
                        dlist.append(0)
                
                        C = ((CList[-1] + C1) * np.exp(lam1 * d[i]) - C1) * np.exp(lam1 * (h - d[i]))
                        D = ((DList[-1] + C2) * np.exp(lam2 * d[i]) - C2) * np.exp(lam2 * (h - d[i]))
                        CList.append(C)
                        DList.append(D)
                        
                        A = C * C3 + D * C4
                        clist.append(A)
            #-------------------Step 3: save the results to a new dictionary-------------------#          
            result[pt] = (tlist, clist, dlist)
    
    elif model_name == 'PKRNN_1CM':
        for pt in reout.keys():
            #-------------------Step 1: get all the information-------------------#
            k = reout[pt][0]
            v = reout[pt][1]
            d = reout[pt][2]
            lb = reout[pt][4]
            output = reout[pt][5]

            td = 24 * reout[pt][3]
            t = np.cumsum(td)
            t = shift_back(t)
            
            lb = shift_back(lb)
            output = shift_back(output)
            
            #-------------------Step 2: calculate output for every hour-------------------#
            tlist, dlist, clist = [], [], []
            MList = [0]
            
            for i in range(len(t)-1):
                tlist.append(t[i])
                dlist.append(d[i])
                clist.append(output[i])
                
                hrs=int(t[i+1]-t[i])
                
                for h in range(1, hrs+1):
                    cnt = t[i]+h
            
                    if cnt <= t[i+1]:
                        dlist.append(0)
                        tlist.append(cnt)

                        A = np.exp(-k[i]*h)
                        B = 1/k[i]*(-np.exp(-k[i]*d[i]) + 1) * np.exp(-k[i]*(h-d[i]))
                        totalmass = (MList[-1]*A + B)
                        MList.append(totalmass)
                        
                        VancoConcn = (totalmass/(v[i]+1e-6))
                        clist.append(VancoConcn)
            #-------------------Step 3: save the results to a new dictionary-------------------#          
            result[pt] = (tlist, clist, dlist)

    return result

#  --------------------- The main evaluation function --------------------- # 
def evaluation(File, underlying_path, inference_path, device, model_name, eval_loc, eval_type):
    #get testloader from the simulated data file
    dataload = DataFromH5py(File)
    trainloader, validloader, testloader = dataload.get_batch()
    underlying_res = get_res_PKRNN_2CM(underlying_path, device, testloader)
    standard_curve = curve_concn(model_name = 'underlying', reout = underlying_res)
    
    if model_name == 'PKRNN_2CM':
        eval_res = get_res_PKRNN_2CM(inference_path, device, testloader)
        eval_curve = curve_concn(model_name = model_name, reout = eval_res)
    elif model_name == 'PKRNN_1CM':
        eval_res = get_res_PKRNN_1CM(inference_path, device, testloader)
        eval_curve = curve_concn(model_name = model_name, reout = eval_res)
    else:
        raise ValueError("Invalid 'model_name' parameter. Must be 'PKRNN_1CM' or 'PKRNN_2CM' for evaluation.")

    print('Getting the two curves for evaluation...')
    standard_concn = {}
    eval_concn = {}
    
    if eval_loc == 'peak':
        for stpt in standard_curve.keys():
            concn = peak_cnt(standard_curve[stpt][1], standard_curve[stpt][2])
            concn = concn.tolist()
            half_length = int(len(concn)/2)
            if eval_type == 'early':
                standard_concn[stpt] = concn[:half_length] + [0] * (len(concn) - half_length)
            elif eval_type == 'later':
                standard_concn[stpt] = [0] * (len(concn) - half_length) + concn[half_length:]
            else:
                standard_concn[stpt] = concn
        

        for evpt in eval_curve.keys():
            evalconcn = peak_cnt(eval_curve[evpt][1], eval_curve[evpt][2])
            evalconcn = evalconcn.tolist()
            half_length = int(len(evalconcn)/2)
            if eval_type == 'early':
                eval_concn[evpt] = evalconcn[:half_length] + [0] * (len(evalconcn) - half_length)
            elif eval_type == 'later':
                eval_concn[evpt] = [0] * (len(evalconcn) - half_length) + evalconcn[half_length:]
            else:
                eval_concn[evpt] = evalconcn
    
    elif eval_loc == 'trough':
        for stpt in standard_curve.keys():
            concn = trough_cnt(standard_curve[stpt][1], standard_curve[stpt][2])
            concn = concn.tolist()
            half_length = int(len(concn)/2)
            if eval_type == 'early':
                standard_concn[stpt] = concn[:half_length] + [0] * (len(concn) - half_length)
            elif eval_type == 'later':
                standard_concn[stpt] = [0] * (len(concn) - half_length) + concn[half_length:]
            else:
                standard_concn[stpt] = concn
        

        for evpt in eval_curve.keys():
            evalconcn = trough_cnt(eval_curve[evpt][1], eval_curve[evpt][2])
            evalconcn = evalconcn.tolist()
            half_length = int(len(evalconcn)/2)
            if eval_type == 'early':
                eval_concn[evpt] = evalconcn[:half_length] + [0] * (len(evalconcn) - half_length)
            elif eval_type == 'later':
                eval_concn[evpt] = [0] * (len(evalconcn) - half_length) + evalconcn[half_length:]
            else:
                eval_concn[evpt] = evalconcn
    
    elif eval_loc == 'both':
        for stpt in standard_curve.keys():
            concn = peak_trough_cnt(standard_curve[stpt][1], standard_curve[stpt][2])
            concn = concn.tolist()
            half_length = int(len(concn)/2)
            if eval_type == 'early':
                standard_concn[stpt] = concn[:half_length] + [0] * (len(concn) - half_length)
            elif eval_type == 'later':
                standard_concn[stpt] = [0] * (len(concn) - half_length) + concn[half_length:]
            else:
                standard_concn[stpt] = concn
        

        for evpt in eval_curve.keys():
            evalconcn = peak_trough_cnt(eval_curve[evpt][1], eval_curve[evpt][2])
            evalconcn = evalconcn.tolist()
            half_length = int(len(evalconcn)/2)
            if eval_type == 'early':
                eval_concn[evpt] = evalconcn[:half_length] + [0] * (len(evalconcn) - half_length)
            elif eval_type == 'later':
                eval_concn[evpt] = [0] * (len(evalconcn) - half_length) + evalconcn[half_length:]
            else:
                eval_concn[evpt] = evalconcn
    
    else:
        raise ValueError("Invalid 'eval_loc' parameter. Must be 'peak', 'trough', or 'both'.")
    
    print(colored('Evaluation starts...', 'green'))    
    errlist = []
    for st_keys in standard_concn.keys():
        for ev_keys in eval_concn.keys():
            if st_keys == ev_keys:
                err = sqrt(MSE(standard_concn[st_keys], eval_concn[ev_keys]))
                errlist.append(err)
    final_RMSE = sum(errlist)/len(errlist)
    print(colored(f'The RMSE for evaluation type {eval_type} and location {eval_loc} is {final_RMSE}. \n', 'red'))
    
    return final_RMSE
        
        