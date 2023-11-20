from __future__ import print_function, division

import math, time, os, sys, h5py, random, shutil, torch
from math import sqrt
import numpy as np
from collections import defaultdict, Counter
from termcolor import colored
from sklearn.metrics import mean_squared_error as MSE

#  --------------------- Helper functions for the PKRNN-2CM model --------------------- #  
# print to file function
def print2file(buf, outFile, mode = 'a'):
    outfd = open(outFile, mode)
    outfd.write(buf + '\n')
    outfd.close()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def iter_batch2(iterable, samplesize):
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    for _ in range(samplesize):
        results.append(iterator.__next__())
    random.shuffle(results)
    return results

#  --------------------- Major model training utilities --------------------- #
def trainsample(sample, model, optimizer):
    model.zero_grad()
    eta1, eta2, eta3, eta4, k1, k2, v1, v2, lb, pred, ccl, d, td, mask, LengList, PtList, loss = model(sample)  # BP
    
    # FOR DEBUG PURPOSE
    # If there is any wrong in models.py, the loss would be returned as 1. 
    # then this block below will save the model to debug. 

    if loss == 1:
        import h5py
        outFile = 'troubleshooting.h5'
        torch.save(model, 'model1')
        import pickle
        with open(r"patient_list.pickle", "wb") as out:
            pickle.dump(pred, out)
    
    loss.backward()    
    optimizer.step()
    return pred, lb, loss.item()

# train with loaders
def trainbatches(loader, model, optimizer):  
    current_loss = 0
    all_losses = []
    # shuffle the batch        
    loader = iter_batch2(loader, len(loader))

    for i, batch in enumerate(loader):
        output, label_tensor, loss = trainsample(batch, model, optimizer)  # BP
        all_losses.append(loss)
        
    return current_loss, all_losses 


def calculate_Error(model, loader):
    y_real, y_hat = [], []
    PtLists, eta1s, eta2s, eta3s, eta4s, k1s, k2s, v1s, v2s =  [] , [] , [] , [] , [] , [] , [] , [], [] 
    lbs, ccls, ds, tds, masks, LengLists =  [], [], [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):           
            eta1, eta2, eta3, eta4, k1, k2, v1, v2, lb, output,ccl, d, td, mask, LengList, PtList, loss = model(batch)
            
            PtLists.append(PtList)
            eta1s.append(eta1)
            eta2s.append(eta2)
            eta3s.append(eta3)
            eta4s.append(eta4)
            k1s.append(k1)
            k2s.append(k2)
            v1s.append(v1)
            v2s.append(v2)
            lbs.append(lb)
            ccls.append(ccl)
            ds.append(d)
            tds.append(td)
            masks.append(mask)
            LengLists.append(LengList) 

            outLeng = mask.shape[0] * mask.shape[1]
            flatten_mask = mask.view(outLeng)
            flatten_mask = torch.as_tensor(flatten_mask, dtype=torch.uint8)
            output = output.contiguous().view(outLeng)[flatten_mask]
            label_tensor = lb.view(outLeng)[flatten_mask]

            y_hat.extend(output.cpu().data.view(-1).numpy()) 
            y_real.extend(label_tensor.cpu().data.view(-1).numpy())
        
    error = sqrt(MSE(np.asarray(y_real), np.asarray(y_hat)))
    return error, y_real, y_hat, PtLists, eta1s, eta2s, eta3s, eta4s, k1s, k2s, v1s, v2s, lbs, ccls, ds, tds, masks, LengLists


def save_results(dataFile, model_header,stats_data, best_train_result, best_valid_result, best_test_result):    
    if os.path.isfile(dataFile) :
        print('Remove existing file, save results to ', dataFile)
        os.remove(dataFile)      
    
    with h5py.File(dataFile, 'w') as f:
        h = f.create_group('hyperparams')
        h.create_dataset('data', data = model_header)
        
        for name, data in zip(['train', 'valid', 'test'], [best_train_result, best_valid_result, best_test_result]):
            _, _, _, best_PtLists, best_eta1s, best_eta2s, best_eta3s, best_eta4s, best_k1s, best_k2s, best_v1s, best_v2s, best_lbs, best_ccls, best_ds, best_tds, best_masks, best_LengLists = data
            h = f.create_group(name)
            h.create_dataset('y_real', data = data[1])
            h.create_dataset('y_hat', data = data[2])

            for i, (PtList, eta1, eta2, eta3,eta4, k1, k2, v1, v2, lb, ccl, d, td, mask, LengList) in enumerate(zip(best_PtLists, best_eta1s, best_eta2s, best_eta3s, best_eta4s, best_k1s, best_k2s, best_v1s, best_v2s, best_lbs, best_ccls, best_ds, best_tds, best_masks, best_LengLists)):

                g = h.create_group(str(i))
                g.create_dataset('PtList', data = PtList)
                g.create_dataset('eta1', data = eta1.cpu())
                g.create_dataset('eta2', data = eta2.cpu())
                g.create_dataset('eta3', data = eta3.cpu())
                g.create_dataset('eta4', data = eta4.cpu())
                g.create_dataset('k1', data = k1.cpu())
                g.create_dataset('k2', data = k2.cpu())
                g.create_dataset('v1', data = v1.cpu())
                g.create_dataset('v2', data = v2.cpu())
                g.create_dataset('lb', data = lb.cpu())
                g.create_dataset('ccl', data = ccl.cpu())
                g.create_dataset('d', data = d.cpu())
                g.create_dataset('td', data = td.cpu())
                g.create_dataset('mask', data = mask.cpu())
                g.create_dataset('LengList', data = LengList)

    stats_data.to_hdf(dataFile, key = 'stats')

#  --------------------- Main function to train the model --------------------- #

def epochs_run(epochs, train, valid, test, model, stats_data, optimizer, patience=10,
               output_dir='results/', output_header='PKRNN_2CM', model_header='', dataFile = 'datafile'):    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    bestValidError = 1000000000000000.0
    bestTestError = 0.0
    bestValidEpoch = 0
    header = 'Train_Error | Valid_Error | Test_Error | Train_time | Valid_time | Test_Error | Epoch | Average_loss'
    logFile = output_dir + output_header + '.log'
    print2file(model_header, logFile)
    print2file(header, logFile)
    
    for ep in range(epochs):
        # training
        start = time.time()
        model = model.train()
        current_loss, train_loss = trainbatches(loader=train, model=model, optimizer=optimizer)
        avg_loss = np.mean(train_loss)
        train_time = timeSince(start)        
        # evaluation
        model = model.eval()
        train_result = calculate_Error(model=model, loader=train)
        train_Error = train_result[0]
        valid_start = time.time()
        valid_result = calculate_Error(model=model, loader=valid)
        valid_Error = valid_result[0]
        valid_time = timeSince(valid_start)
        test_start = time.time()
        test_result = calculate_Error(model=model, loader=test)
        test_Error = test_result[0]
        test_time = timeSince(test_start)

        pFile = '%s | %s | %s| %s | %s | %s | %s | %s' % (train_Error, valid_Error, test_Error, train_time, valid_time, test_time, ep, avg_loss)  
        print2file(pFile, logFile)
        if valid_Error < bestValidError:
            bestValidError = valid_Error
            bestTrainError = train_Error
            bestValidEpoch = ep
            best_model = model
            bestTestError = test_Error          
            best_test_result = test_result 
            best_valid_result = valid_result           
            best_train_result = train_result              
        if ep - bestValidEpoch > patience:
            break

        print('Epoch %2s | Best Train Error: %4s | Best Valid Error %4s | Best Test Error: %4s' %(ep, bestTrainError, bestValidError, bestTestError), end="\r", flush=True)        
        # save model & parameters
        torch.save(best_model, output_dir + output_header + '.pth')
        torch.save(best_model.state_dict(), output_dir + output_header + '.st')
    # Record in the log file the final best valid Error and train Error
    pFile = 'Best Train Error of %f | Best Valid Error %f | Test Error of %f at epoch %d ' % (bestTrainError, bestValidError, bestTestError, bestValidEpoch)
    print('\n')
    print(colored(pFile, 'green'))
 
    ################# SAVE RESULTS ##################
    dataFile = output_dir + dataFile
    save_results(dataFile, model_header, stats_data, best_train_result, best_valid_result, best_test_result)    
    return bestTrainError, bestValidError, bestTestError, bestValidEpoch


#  --------------------- Helper functions for simulation --------------------- #  
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


def sim_data(ptlist, ptcheck, concn_dict, real_data):
    ptd = {}
    for i in range(len(ptcheck)):
        pt_id = ptcheck[i][6]     #PtList
        td_check = 24 * ptcheck[i][7]  
        ptd[pt_id] = td_check
    
    for pt in range(len(ptlist)):
        lb = ptlist[pt][3]       #LabelTensor
        m = ptlist[pt][5]        #MaskTensor
        ptid = ptlist[pt][6]     #PtList
        td = 24 * ptlist[pt][7]  #TimeDiffTensor
 
        for tdk in ptd.keys():
            if int(tdk) == ptid:
                oldtd = ptd[tdk]
    
        #get the output for this patient
        for outk in concn_dict.keys():
            if int(outk) == ptid:
                output = concn_dict[outk]
     
        t = np.cumsum(td)
        t = np.insert(t, 0, 0)
        t = np.delete(t, -1)

        oldt = np.cumsum(oldtd)
        oldt = np.insert(oldt, 0, 0)
        oldt = np.delete(oldt, -1)
        #index list for different elements 
        tidx = np.arange(t.shape[0])[np.in1d(t,oldt)]
        tidx_n = np.arange(t.shape[0])[~np.in1d(t,oldt)]
    
        if len(tidx_n) > 0:
            for i in tidx_n:
                lb[i] = output[i]
            if real_data == 'remove':
                for j in tidx:
                    lb[j] = 0
        m = np.where(lb==0, 0, 1)
        ptlist[pt][3] = lb
        ptlist[pt][5] = m
        ptlist[pt][7] = td/24
    
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
#duplicate the original input data file
def duplicate_file(file, loc, sim_type, real_data, step):
    # Extract the file name and extension
    file_name_without_ext = file.split('.')[0]
    file_extension = file.split('.')[-1]
    if step == 'step1':
        new_file = f"{file_name_without_ext}_{loc}_{sim_type}_{real_data}_{step}.{file_extension}"
    elif step == 'step2':
        new_name = file_name_without_ext[:-1]
        new_file = f"{new_name}2.{file_extension}"    
    # Create the full source and destination paths
    destination_path = file.replace(file, new_file)
    # Duplicate the file
    shutil.copy2(file, destination_path)    
    print(f"{file} successfully duplicated as {new_file}.")   
    return new_file


def get_concn(model, loader):
    result = {}    
    for group in loader:
        eta1, eta2, eta3, eta4, _, _, _, _, lb, pred_con, ccl, d, td, _, LengList, PtList, _ = model(group)        
        for i in range(len(group[0])):    
            result[PtList[i]] = pred_con[i].cpu().detach().numpy()
    
    return result

def simulate(org_dataset, dataset, concn, sim_type, real_data):
    ptcheck, idxl_check = get_ptlist(org_dataset)   
    if sim_type == 'move':
        ptd = {}
        for ptc in range(len(ptcheck)):
            pt_id = ptcheck[ptc][6]     #PtList
            td_check = 24 * ptcheck[ptc][7]
            ptd[pt_id] = td_check        
        # Iterate over the groups
        for group_name in dataset.keys():
            group = dataset[group_name]            
            td = group['TimeDiffTensor'][:]
            td *= 24
            #other features that may change based on td 
            lb = group['LabelTensor'][:]
            m = group['MaskTensor'][:]
            #won't change for any options
            pt_list = group['PtList'][:]
            if len(pt_list) == 50:
                for idx in range(len(pt_list)):        
                    ptid = pt_list[idx]
                    lbi = lb[idx]
                    mi = m[idx]                  
                    t = np.cumsum(td[idx])
                    t = np.insert(t, 0, 0)
                    t = np.delete(t, -1)

                    for tdk in ptd.keys():
                        if int(tdk) == ptid:
                            oldtd = ptd[tdk]
                    #get the output for this patient
                    for outk in concn.keys():
                        if int(outk) == ptid:
                            output = concn[outk]                    
                    
                    oldt = np.cumsum(oldtd)
                    oldt = np.insert(oldt, 0, 0)
                    oldt = np.delete(oldt, -1)
                    #index list for different elements 
                    tidx = np.arange(t.shape[0])[np.in1d(t,oldt)]
                    tidx_n = np.arange(t.shape[0])[~np.in1d(t,oldt)]
                
                    if len(tidx_n) > 0:
                        for i in tidx_n:
                            lbi[i] = output[i]
                        if real_data == 'remove':
                            for j in tidx:
                                lbi[j] = 0
                    mi = np.where(lbi==0, 0, 1)
                    lb[idx] = lbi
                    m[idx] = mi  
            group['LabelTensor'][:] = lb
            group['MaskTensor'][:] = m
    
    else:
        #get patient data as a list
        ptlist, idxl_str = get_ptlist(dataset)
        #simulation
        new_ptlist = sim_data(ptlist, ptcheck, concn, real_data)
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