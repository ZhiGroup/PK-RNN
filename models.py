#import necessary packages
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

#check GPU availability
use_cuda = torch.cuda.is_available()

class EHREmbeddings(nn.Module):
    def __init__(self, input_size, embed_dim = 8, hidden_size = 64, n_layers = 1, dropout_r = 0, n_cont = 2, batchsize = 50):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_r = dropout_r
        self.n_cont = n_cont
        self.batchsize = batchsize
        #Embedding
        input_size = input_size[0]
        self.embed = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.in_size = embed_dim + self.n_cont # for continuous predictors
        #GRU
        self.cell = nn.GRU
        self.onecell = nn.GRUCell(self.in_size + 1, self.hidden_size, bias = True)
        self.rnn_c = self.cell(self.in_size, self.hidden_size, 
                               num_layers=self.n_layers, dropout=self.dropout_r, bidirectional= False)
        self.out = nn.Linear(self.hidden_size, 4, bias=True) #output 4 parameters
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def EmbedPatients_MB(self,input):
        ContTensor, CatTensor, LabelTensor, MaskTensor, DoseTensor, TimeDiffTensor, VTensor, VancoClTensor, PtList, LengList = input        
        if use_cuda: 
            CatTensor = CatTensor.cuda()
            ContTensor = ContTensor.cuda()
            LabelTensor = LabelTensor.cuda()
            MaskTensor = MaskTensor.cuda()
            DoseTensor = DoseTensor.cuda()
            TimeDiffTensor = TimeDiffTensor.cuda()
            outEmb = ContTensor
        if self.embed_dim ==0: # Only use continuous predictors: 
            outEmb = ContTensor
        else: 
            # Embedding categorical data
            catEmb = self.embed(CatTensor)  ## Embedding for codes [bz, max_visit, max_cat, embed]                  
            ### Use only categorical embedding
            outEmb = torch.sum(catEmb, dim=2) #[bz, max_visit, embed] 
            outEmb = torch.cat((outEmb, ContTensor), dim=2) ##[bz, max_visit, embed+2]                  
        return outEmb, LabelTensor,LengList, MaskTensor, DoseTensor, TimeDiffTensor, VTensor, VancoClTensor, PtList 
    
class PKRNN_1CM(EHREmbeddings):
    def __init__(self, input_size, embed_dim=8, hidden_size=64, n_layers=1, dropout_r=0, n_cont=2, batchsize=50,
                 # special param:
                 paramdict = {'real_vanco_to_feedback': True,
                              'change_regularize': 'square',
                              'scale_start_v': 1000, 
                              'scale_start_k': 1000, 
                              'scale_change_v': 100000, 
                              'scale_change_k': 10000}):
        EHREmbeddings.__init__(self, input_size, embed_dim, hidden_size, n_layers, dropout_r, n_cont, batchsize)       
        self.real_vanco_to_feedback = paramdict['real_vanco_to_feedback']
        self.change_regularize = paramdict['change_regularize']
        self.scale_start_v = paramdict['scale_start_v']
        self.scale_start_k = paramdict['scale_start_k']
        self.scale_change_v = paramdict['scale_change_v']
        self.scale_change_k = paramdict['scale_change_k']

    # embedding function
    def EmbedPatient_MB(self, input):
        return EHREmbeddings.EmbedPatients_MB(self, input)
    
    def forward(self, input):
        outEmb, lb, LengList, mask, d, td, v_from_weight, VancoClTensor, PtList = self.EmbedPatient_MB(input)        
        # --------------------- FEEDBACK RNN  --------------------- #
        h_0 = (torch.zeros(len(outEmb), self.hidden_size)).cuda()
        hidden = [h_0]
        # --------------------- 2CM PK MODEL --------------------- #
        self.batchsize = outEmb.shape[0]
        # initialize K: [bs, 1] --> [bs, seq, 1]
        K0 = VancoClTensor[:,0] 
        V0 = v_from_weight[:,0]
        K0 = K0.cuda()
        V0 = V0.cuda()        
        td = td*24
        TotalMassList = [torch.zeros((self.batchsize, 1)).cuda()]
        VancoConcnList = [torch.zeros((self.batchsize)).cuda()]
        KV = []
         
        for i in range(outEmb.shape[1]): # embed [bs, visit, emb_size+n_conts]
        #range(outEmb.shape[1])=# of visit
            if self.real_vanco_to_feedback:
                if i == 0: # no data yet
                    vanco_to_feedback = VancoConcnList[-1].detach()
                else:
                    vanco_to_feedback = VancoConcnList[-1].detach() * (1 - mask[:, i-1].float()) + lb[:, i-1] * mask[:, i-1].float()
            
            else:
                vanco_to_feedback = VancoConcnList[-1].detach()

            Input_plus_Output = torch.cat((outEmb[:, i, :], vanco_to_feedback.reshape(-1,1).detach()), dim = 1)      
            h_i = self.onecell(Input_plus_Output, hidden[-1])
            kv = torch.exp(self.out(h_i)).cuda()
            k = kv[:,0]
            v = kv[:,1]
            A = torch.exp(-k*td[:,i])  #A.shape = [bs, seq]
            B = 1/k*(-torch.exp(-k*d[:,i]) + 1) * torch.exp(-k*(td[:,i]-d[:,i]))           
            totalmass = (TotalMassList[-1][:,0]*A + B)
            VancoConcn = (totalmass/(v+1e-6))
            
            VancoConcnList.append(VancoConcn)  # [bs, 1]
            TotalMassList.append(totalmass.reshape(-1, 1))
            KV.append(kv)
            hidden.append(h_i)
        output = torch.stack(VancoConcnList[1:]).permute(1,0) # [bs, seq_length]
        #  --------------------- EXTRACT K,V FOR PLOTTING --------------------- #       
        KV = torch.stack(KV).permute(1,0,2) #[bs, seq, 2]
        K, V = KV[:,:,0], KV[:,:,1]       
         #  --------------------- GET FLATTEN OUTPUT AND FLATTEN LABEL --------------------- #
        mask = (lb != 0) #new
        outLeng = mask.shape[0] * mask.shape[1]
        flatten_mask = mask.view(outLeng)
        flatten_mask = torch.as_tensor(flatten_mask, dtype=torch.uint8)
        flatten_output = output.contiguous().view(outLeng)[flatten_mask]
        flatten_label = lb.view(outLeng)[flatten_mask]
         #  --------------------- CALCULATING LOSS --------------------- #            
        if self.change_regularize == 'abs':            
            kv_change = (torch.abs(torch.sub(K[:,1:],K[:,:-1])).mean() * self.scale_change_k
                         + torch.abs(torch.sub(V[:,1:],V[:,:-1])).mean() * self.scale_change_v)           
        else:
            kv_change = ((torch.sub(K[:,1:],K[:,:-1])).pow(2).mean() * self.scale_change_k
                         + (torch.sub(V[:,1:],V[:,:-1])).pow(2).mean() * self.scale_change_v)        
        MSE = (flatten_label - flatten_output).pow(2).mean()
        kv_loss = (K0 - K[:,0]).pow(2).mean() * self.scale_start_k  + (V0 - V[:,0]).pow(2).mean() * self.scale_start_v
        loss = MSE + kv_loss + kv_change
        
        # FOR DEBUGGING PURPOSE
        for i, item in enumerate([lb, output]):
            if (torch.isnan(item)).any():
                print(i, 'IsNan\n', item)
                print(PtList)
                print(item.sum(dim = 0))
                print(item.sum(dim = 1))
                loss = 1
        return K, V, lb , output, d, td/24, mask, v_from_weight, LengList, PtList , outEmb[:,:,-self.n_cont:], loss

       
class PKRNN_2CM(EHREmbeddings):
    def __init__(self, input_size, embed_dim=8, hidden_size=64, n_layers=1, dropout_r=0, n_cont=2, batchsize=50,
                 # special param:
                 paramdict = {'real_vanco_to_feedback': True,
                              'change_regularize': 'square',
                              'eta1_var': 0.12, 
                              'eta2_var': 0.149, 
                              'eta3_var': 0.416,
                              'scale_change_eta': 100}):
        EHREmbeddings.__init__(self, input_size, embed_dim, hidden_size, n_layers, dropout_r, n_cont, batchsize)        
        self.real_vanco_to_feedback = paramdict['real_vanco_to_feedback']
        self.change_regularize = paramdict['change_regularize']
        self.eta1_var = paramdict['eta1_var']
        self.eta2_var = paramdict['eta2_var']
        self.eta3_var = paramdict['eta3_var']
        self.scale_change_eta = paramdict['scale_change_eta']

    # embedding function
    def EmbedPatient_MB(self, input):
        return EHREmbeddings.EmbedPatients_MB(self, input)
    
    def forward(self, input):
        outEmb, lb, LengList, mask, d, td, v_from_weight, VancoClTensor, PtList = self.EmbedPatient_MB(input)        
        age = input[0][:, :1, 2]*16.936469 + 58.239251
        weight = input[0][:, :1, 21]*30.519849 + 87.575200
        creatinine = input[0][..., 11]*1.418099 + 1.314668
        gender = (input[1][:, :1, :] == 5).sum(2).type(torch.float)*0.15+0.85
        dose_mask = (d != 0).type(torch.float)
        ccl = (140 - age)*weight*gender/72/creatinine
        ccl = ccl.cuda()      
        # --------------------- FEEDBACK RNN  --------------------- #
        h_0 = (torch.zeros(len(outEmb), self.hidden_size)).cuda()
        hidden = [h_0]
        # --------------------- 2CM PK MODEL --------------------- #
        self.batchsize = outEmb.shape[0]
        td = td*24
        VancoConcnList = [torch.zeros((self.batchsize)).cuda()]
        CList, DList = [torch.zeros((self.batchsize)).cuda()],[torch.zeros((self.batchsize)).cuda()]
        ETA, K1, K2, V1, V2 = [], [], [], [], []
        
        for i in range(outEmb.shape[1]): # embed [bs, visit, emb_size+n_conts]
        #range(outEmb.shape[1])=# of visit
            if self.real_vanco_to_feedback:
                if i == 0: # no data yet
                    vanco_to_feedback = VancoConcnList[-1].detach()
                else:
                    vanco_to_feedback = VancoConcnList[-1].detach() * (1 - mask[:, i-1].float()) + lb[:, i-1] * mask[:, i-1].float()
            
            else:
                vanco_to_feedback = VancoConcnList[-1].detach()
            Input_plus_Output = torch.cat((outEmb[:, i, :], vanco_to_feedback.reshape(-1,1).detach()), dim = 1)       
            h_i = self.onecell(Input_plus_Output, hidden[-1])
            eta = self.out(h_i).cuda()
            eta1 = eta[:,0]
            eta2 = eta[:,1]
            eta3 = eta[:,2]
            eta4 = eta[:,3]
            #PK parameters: k1, v1, v2, R
            v1 = 33.1 * torch.exp(eta1)
            k1 = 3.96 * ccl[:,i] / 100 * torch.exp(eta2)
            r = 1000 / v1
            v2 = 48.3 * torch.exp(eta4)
            R = - 6.99 / 48.3 * torch.exp(eta3)
            k2 = R * v2  
            delta = torch.sqrt((k1 - k2 - R * v1) ** 2 + 4 * k1 * R * v1) / v1
            lam1 = (((-k1 + k2 + R * v1) / v1) - delta) / 2
            lam2 = (((-k1 + k2 + R * v1) / v1) + delta) / 2            
            C1 = r * (k2 / v1) / lam1 / delta
            C2 = -r * (k2 / v1) / lam2 / delta
            C3 = - (lam1 - R) / (k2 / v1)
            C4 = - (lam2 - R) / (k2 / v1)           
            C = ((CList[-1] + C1) * torch.exp(lam1 * d[:,i]) - C1) * torch.exp(lam1 *(td[:,i] - d[:,i]))
            D = ((DList[-1] + C2) * torch.exp(lam2 * d[:,i]) - C2) * torch.exp(lam2 *(td[:,i] - d[:,i]))
            CList.append(C)
            DList.append(D)            
            A = C * C3 + D * C4
            VancoConcnList.append(A) # [bs, 1]
            
            ETA.append(eta)
            K1.append(k1)
            K2.append(k2)
            V1.append(v1)
            V2.append(v2)                     
            hidden.append(h_i)            
        # convert lists to tensors
        K1 = torch.stack((K1)).permute(1,0).cuda()
        K2 = torch.stack((K2)).permute(1,0).cuda()
        V1 = torch.stack((V1)).permute(1,0).cuda()
        V2 = torch.stack((V2)).permute(1,0).cuda()
        output = torch.stack(VancoConcnList[1:]).permute(1,0) # [bs, seq_length]
        #  --------------------- EXTRACT ETA FOR PLOTTING --------------------- #
        ETA = torch.stack(ETA).permute(1,0,2) #[bs, seq, 2]
        ETA1, ETA2, ETA3, ETA4 = ETA[:,:,0], ETA[:,:,1], ETA[:,:,2], ETA[:,:,3]       
        #  --------------------- GET FLATTEN OUTPUT AND FLATTEN LABEL --------------------- #
        mask = (lb != 0)
        outLeng = mask.shape[0] * mask.shape[1]
        flatten_mask = mask.view(outLeng)
        flatten_mask = torch.as_tensor(flatten_mask, dtype=torch.uint8)
        flatten_output = output.contiguous().view(outLeng)[flatten_mask]
        flatten_label = lb.view(outLeng)[flatten_mask]
         #  --------------------- CALCULATING LOSS --------------------- #
        if self.change_regularize == 'abs':
            eta_change = torch.abs(torch.sub(ETA1[:,1:],ETA1[:,:-1])).mean() * self.scale_change_eta +\
                         torch.abs(torch.sub(ETA2[:,1:],ETA2[:,:-1])).mean() * self.scale_change_eta +\
                         torch.abs(torch.sub(ETA3[:,1:],ETA3[:,:-1])).mean() * self.scale_change_eta         
        else:
            eta_change = (torch.sub(ETA1[:,1:],ETA1[:,:-1])).pow(2).mean() * self.scale_change_eta +\
                         (torch.sub(ETA2[:,1:],ETA2[:,:-1])).pow(2).mean() * self.scale_change_eta +\
                         (torch.sub(ETA3[:,1:],ETA3[:,:-1])).pow(2).mean() * self.scale_change_eta        
        loss = ((flatten_label - flatten_output).pow(2).mean() + eta_change)
        l2_lambda = 1e-1
        l2_norm = torch.diff(ETA1).pow(2).mean()+ torch.diff(ETA2).pow(2).mean() + torch.diff(ETA3).pow(2).mean() + torch.diff(ETA4).pow(2).mean()
        loss = loss + l2_lambda * l2_norm
        
        # FOR DEBUGGING PURPOSE
        for i, item in enumerate([lb, output]):
            if (torch.isnan(item)).any():
                print(i, 'IsNan\n', item)
                print(PtList)
                print(item.sum(dim = 0))
                print(item.sum(dim = 1))
                loss = 1
        return ETA1, ETA2, ETA3, ETA4, K1, K2, V1, V2, lb, output, ccl, d, td/24, mask, LengList, PtList, loss
