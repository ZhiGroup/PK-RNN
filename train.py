from __future__ import print_function, division
import os, sys, re, string, random, argparse, time, math
import h5py
import ast

import numpy as np
import pandas as pd
import torch
from torch import optim
from termcolor import colored
from torchinfo import summary

#ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#import models and utils files
from models import EHREmbeddings, PKRNN_2CM
import utils as ut
import simulation_utils as ut1

#function to set the random seed manually
def set_seed(manualSeed):
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#  --------------------- Class for embedding --------------------- #
class DataFromH5py:    
    def __init__(self, file):
        self.file = file
    def get_batch(self):
        datasets = []
        with h5py.File(self.file, 'r') as f:
            for setname in ['train', 'valid', 'test']:
                loader = []
                data = f[setname]
                for index, subdata in data.items():
                        group_key = ['ContTensor', 'CatTensor', 'LabelTensor', 'MaskTensor', 'DoseTensor', 'TimeDiffTensor', 'VTensor', 'VancoClTensor', 'PtList', 'LengList']  #use this to ensure order
                        list_key = ['PtList', 'LengList']
                        loader.append([subdata[key].value.tolist() if key in list_key 
                                                                   else torch.from_numpy(subdata[key].value) for key in group_key])
                datasets.append(loader)
                print(f'Batch loaded for {setname} set! {len(loader)} items included')
        return datasets
    
    def get_dictionary(self):
        with h5py.File(self.file, 'r') as f:
            dictionary = f['CodeDict']['data'][()]  
        dictionary = ast.literal_eval(dictionary)
        print(f'Dictionary loaded! {len(dictionary)} items included')
        return dictionary

    def get_stats(self):
        data = pd.read_hdf(self.file, key = 'stats') 
        print(f'Stats table loaded, containing statistics (mean, std) for {len(data)} features')
        return data
        
    def show_batch_info(self):
        print('-'*30, 'INFORMATION PER BATCH', '-'*30) 
        print('10 items included: ')
        with h5py.File(self.file, 'r') as f:
            data = f['train']['0']
            for item in  ['ContTensor', 'CatTensor', 'LabelTensor', 'MaskTensor', 'DoseTensor', 'TimeDiffTensor', 'VTensor', 'VancoClTensor', 'PtList', 'LengList']:
                print(item + " "*(20 - len(item)), data[item].shape)   

#  --------------------- Training function --------------------- #

def run_model(inFile, device, paramdict, dataFile = '', embed_dim = 8, hidden_size = 64, n_layers = 1, dropout_r = 0, 
              batchsize = 50, n_epoch = 1500, optimizer_name = 'adamax', lr = 0.01, L2 = 0.2, patience = 10, 
              simulation = False, loc = None, sim_type = None, real_data = None):

    # extracting data from inFile: train, test, valid, stats_data
    dataload = DataFromH5py(inFile)
    trainloader, validloader, testloader = dataload.get_batch()
    dictionary = dataload.get_dictionary()
    stats_data = dataload.get_stats()
    
    input_size = [len(dictionary) + 1]  
    n_cont = trainloader[0][0].shape[2] # get the number of n_cont
    conList = stats_data['FEATURES'].values.tolist()
   
    print(colored(f'\n{"="*40}SUMMARY{"="*40}', 'green'))
    print(f'> Using [{n_cont}] continuous features: {conList}.')
    print(f'> Using [{input_size[0]}] codes. Each will be mapped to a [{embed_dim}] dimension embedding\n')
    
    trainloader = [loader for loader in trainloader if len(loader[0]) ==batchsize]
    validloader = [loader for loader in validloader if len(loader[0]) ==batchsize]
    testloader = [loader for loader in testloader if len(loader[0]) ==batchsize]
    
    model = PKRNN_2CM(input_size = input_size, embed_dim = embed_dim, hidden_size = hidden_size, n_layers = n_layers, 
                      dropout_r = dropout_r, n_cont = n_cont, batchsize = batchsize, paramdict = paramdict).cuda()
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=L2)
    paramdict.update({'EmbSize': embed_dim,'n_Layers': n_layers, 'Hidden_size': hidden_size, 
                      'Dropout':  dropout_r, 'Optimizer' : optimizer_name, 'LR': lr, 'L2': L2})
    model_header = str(paramdict)    
    
    print('> Hyperparams used: \n', model_header)
    print(colored(f'{"="*90}\n', 'green'))

    
    #main training function
    if simulation != True:
        RMSE_train, RMSE_valid, RMSE_test, best_epo = ut.epochs_run(n_epoch, train = trainloader, valid = validloader, test = testloader, stats_data = stats_data, model = model, optimizer = optimizer, patience = patience, output_dir = 'results/', output_header = 'PKRNN_2CM', model_header = model_header, dataFile = dataFile)
    
    #run the simulation process when simulation = True
    else:
        #-------------------Step 1: depulicate the original data file-------------------#
        print('\n')
        print(colored('Duplicate the original data file:', 'green'))
        sim_inFile = ut.duplicate_file(inFile, loc = loc, sim_type = sim_type, real_data = real_data, step = 'step1')
        
        print("Simulation step 1 starts...")
        for loader in trainloader, validloader, testloader:            
            sim_file = h5py.File(sim_inFile, 'a')
            if loader == trainloader:
                dataset = sim_file['train']
            elif loader == testloader:
                dataset = sim_file['test']
            else:
                dataset = sim_file['valid']
                    
            #-------------------Step 2: simulate new time and other data (expect measurements)-------------------#
            if sim_type == 'move':
                sim_dataset = ut1.simulate_move(dataset = dataset, loc=loc)
                # Save the changes
                sim_file.flush()
                # Close the simulated file
                sim_file.close()

            elif sim_type in ['add_half', 'add_all']:
                sim_dataset = ut1.simulate_add(dataset = dataset, loc=loc, sim_type = sim_type, real_data = real_data)
                # Save the changes
                sim_file.flush()
                # Close the simulated file
                sim_file.close()

            else:
                raise ValueError("Invalid 'sim_type' parameter. Must be 'move', 'add_half' or 'add_all'.")

        print("Simulation step 1 complete, the step 1 simulated dataset is ready. \n")
        
        #-------------------Step 3: run the model to get the simulated measurements-------------------#
        sim_dataload = DataFromH5py(sim_inFile)
        print(colored('Loading the simulated data...\n', 'green'))

        sim_trainloader, sim_validloader, sim_testloader = sim_dataload.get_batch()
        sim_dict = sim_dataload.get_dictionary()
        sim_stats_data = sim_dataload.get_stats()
        sim_input_size = [len(sim_dict) + 1]  
        sim_n_cont = sim_trainloader[0][0].shape[2] # get the number of n_cont

        sim_trainloader = [loader for loader in sim_trainloader if len(loader[0]) ==batchsize]
        sim_validloader = [loader for loader in sim_validloader if len(loader[0]) ==batchsize]
        sim_testloader = [loader for loader in sim_testloader if len(loader[0]) ==batchsize]

        sim_model = PKRNN_2CM(input_size = sim_input_size, embed_dim = embed_dim, hidden_size = hidden_size, n_layers = n_layers,
                              dropout_r = dropout_r, n_cont = sim_n_cont, batchsize = batchsize, paramdict = paramdict).cuda()
        sim_optimizer = optim.Adamax(sim_model.parameters(), lr=lr, weight_decay=L2)
        paramdict.update({'EmbSize': embed_dim,'n_Layers': n_layers, 'Hidden_size': hidden_size, 
                          'Dropout':  dropout_r, 'Optimizer' : optimizer_name, 'LR': lr, 'L2': L2})
        sim_model_header = str(paramdict)
        
        #main training function
        sim_out_header = f"PKRNN_2CM_sim_{loc}_{sim_type}_{real_data}_step1"
        RMSE_train, RMSE_valid, RMSE_test, best_epo = ut.epochs_run(n_epoch, train = sim_trainloader, valid = sim_validloader, test = sim_testloader, stats_data = sim_stats_data, model = sim_model, optimizer = sim_optimizer, patience = patience, output_dir = 'results/', output_header = sim_out_header, model_header = sim_model_header, dataFile = dataFile)

        #-------------------Step 4: update the simulated file-------------------#
        print(colored('Duplicate the simulated step 1 file:', 'green'))
        sim_inFile2 = ut.duplicate_file(sim_inFile, loc = loc, sim_type = sim_type, real_data = real_data, step = 'step2')
        
        print("Simulation step 2 starts...")        
        step2_File = f"results/PKRNN_2CM_sim_{loc}_{sim_type}_{real_data}_step1.pth"
        for loader in sim_trainloader, sim_validloader, sim_testloader:
            sim_model2 = torch.load(step2_File, map_location=device)
            concn = ut.get_concn(sim_model2, loader=loader)
            
            org_file = h5py.File(inFile, 'r')
            sim_file2 = h5py.File(sim_inFile2, 'a')
            
            if loader == sim_trainloader:
                org_d = org_file['train']
                sim_d = sim_file2['train']
            elif loader == sim_testloader:
                org_d = org_file['test']
                sim_d = sim_file2['test']
            else:
                org_d = org_file['valid']
                sim_d = sim_file2['valid']
            
            sim_d2 = ut.simulate(org_d, sim_d, concn = concn, sim_type = sim_type, real_data = real_data)
            sim_file2.flush()
            sim_file2.close()
        
        print("Simulation completed, the simulated dataset is ready. \n")
        
        #-------------------Step 3: run the model to get the simulated measurements-------------------#
        file_name = f"batches_{loc}_{sim_type}_{real_data}_step2.h5"
        sim_dataload2 = DataFromH5py(file_name)
        print(colored('Loading the simulated data...\n', 'green'))

        sim_trainloader2, sim_validloader2, sim_testloader2 = sim_dataload2.get_batch()
        sim_dict2 = sim_dataload2.get_dictionary()
        sim_stats_data2 = sim_dataload2.get_stats()
        sim_input_size2 = [len(sim_dict2) + 1]  
        sim_n_cont2 = sim_trainloader2[0][0].shape[2] # get the number of n_cont

        sim_trainloader2 = [loader for loader in sim_trainloader2 if len(loader[0]) ==batchsize]
        sim_validloader2 = [loader for loader in sim_validloader2 if len(loader[0]) ==batchsize]
        sim_testloader2 = [loader for loader in sim_testloader2 if len(loader[0]) ==batchsize]

        final_sim_model = PKRNN_2CM(input_size = sim_input_size2, embed_dim = embed_dim, hidden_size = hidden_size, n_layers = n_layers,
                                    dropout_r = dropout_r, n_cont = sim_n_cont2, batchsize = batchsize, paramdict = paramdict).cuda()
        sim_optimizer2 = optim.Adamax(final_sim_model.parameters(), lr=lr, weight_decay=L2)
        paramdict.update({'EmbSize': embed_dim,'n_Layers': n_layers, 'Hidden_size': hidden_size, 
                          'Dropout':  dropout_r, 'Optimizer' : optimizer_name, 'LR': lr, 'L2': L2})
        sim_model_header2 = str(paramdict)
        
        #main training function
        sim_out_header2 = f"PKRNN_2CM_sim_{loc}_{sim_type}_{real_data}"
        RMSE_train, RMSE_valid, RMSE_test, best_epo = ut.epochs_run(n_epoch, train = sim_trainloader2, valid = sim_validloader2, test = sim_testloader2, stats_data = sim_stats_data2, model = final_sim_model, optimizer = sim_optimizer2, patience = patience, output_dir = 'results/', output_header = sim_out_header2, model_header = sim_model_header2, dataFile = dataFile)
        
    return RMSE_train, RMSE_valid, RMSE_test, best_epo  
