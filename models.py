#import necessary packages
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

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
          
            ### Use only categorical embedding, specify MEDICATION_ID
            outEmb = torch.sum(catEmb, dim=2) #[bz, max_visit, embed] 
            outEmb = torch.cat((outEmb, ContTensor), dim=2) ##[bz, max_visit, embed+2]                  

        return outEmb, LabelTensor,LengList, MaskTensor, DoseTensor, TimeDiffTensor, VTensor, VancoClTensor, PtList 


    
class PKRNN(EHREmbeddings):
    # check if VancoCLTensor is actually elimination rate using Matzke equation, which is 8.3e-4 * CrCL + 4.4e-3
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
        self.d_excess = torch.zeros(batchsize)
        if use_cuda:
            self.d_excess = self.d_excess.cuda()

    # embedding function
    def EmbedPatient_MB(self, input):
        return EHREmbeddings.EmbedPatients_MB(self, input)
    
    def forward(self, input):
        outEmb, lb, LengList, mask, d, td, v_from_weight, VancoClTensor, PtList = self.EmbedPatient_MB(input)
        self.d_excess.fill_(0)
        # --------------------- FEEDBACK RNN  --------------------- #
        h_0 = (torch.zeros(len(outEmb), self.hidden_size))
        if use_cuda:
            h_0 = h_0.cuda()
        hidden = [h_0]

        # --------------------- 2CM PK MODEL --------------------- #
        self.batchsize = outEmb.shape[0]
        # initialize K: [bs, 1] --> [bs, seq, 1]
        K0 = VancoClTensor[:,0] 
        V0 = v_from_weight[:,0]
        if use_cuda:
            K0 = K0.cuda()
            V0 = V0.cuda()
        
        td = td*24
        TotalMassList = [torch.zeros((self.batchsize, 1))]
        VancoConcnList = [torch.zeros((self.batchsize))]
        if use_cuda:
            TotalMassList[0] = TotalMassList[0].cuda()
            VancoConcnList[0] = VancoConcnList[0].cuda()
        KV = []
        auc = torch.zeros(self.batchsize).to(K0.device)

        
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
            # outEmb [bs, in_size], VancoConcnList[-1] [bs, 1] ----> Input_plus_Output [bs, in_size+1]
            
            h_i = self.onecell(Input_plus_Output, hidden[-1]) # h_i.shape: [bs, hidden_size]
            kv = torch.exp(self.out(h_i))
            k = kv[:,0]
            v = kv[:,1]
            total_infusion = d[:, i] + self.d_excess[:self.batchsize]
            t_infusion = torch.clamp(total_infusion, max=td[:, i])
            A = torch.exp(-k*td[:,i])  #A.shape = [bs, seq]
            B = 1/k*(-torch.exp(-k*t_infusion) + 1) * torch.exp(-k*(td[:, i] - t_infusion))
            self.d_excess[:self.batchsize] = (total_infusion - t_infusion).detach()
            M0 = TotalMassList[-1][:,0]
            totalmass = (M0*A + B)
            VancoConcn = (totalmass/(v+1e-6))
            
            # --- Analytical AUC for this interval (1-compartment) ---
            # Infusion phase integral (0 to t_infusion)
            # mass M(t) = a * exp(-k*t) + 1/k, where a = M0 - 1/k
            a_inf = M0 - 1.0 / k
            auc_mass_inf = (a_inf / -k) * (torch.exp(-k * t_infusion) - 1) + t_infusion / k
            
            # Decay phase integral (0 to t_decay)
            M_peak = a_inf * torch.exp(-k * t_infusion) + 1.0 / k
            t_decay = td[:, i] - t_infusion
            auc_mass_dec = (M_peak / -k) * (torch.exp(-k * t_decay) - 1)
            
            auc = auc + (auc_mass_inf + auc_mass_dec) / v

            VancoConcnList.append(VancoConcn)  # [bs, 1]
            TotalMassList.append(totalmass.reshape(-1, 1))
            KV.append(kv)
            hidden.append(h_i)
            #print(A.shape, B.shape, DList[-1].shape, totalmass.shape) # [bs], [bs], [bs,1], [bs]
            

        output = torch.stack(VancoConcnList[1:]).permute(1,0) # [bs, seq_length]

        #  --------------------- EXTRACT K,V FOR PLOTTING --------------------- #
        
        KV = torch.stack(KV).permute(1,0,2) #[bs, seq, 2]
        K, V = KV[:,:,0], KV[:,:,1]
        
         #  --------------------- GET FLATTEN OUTPUT AND FLATTEN LABEL --------------------- #
        mask = (lb != 0) #new
        outLeng = mask.shape[0] * mask.shape[1]
        flatten_mask = mask.view(outLeng)
        flatten_mask = torch.as_tensor(flatten_mask, dtype=torch.uint8)
        
        #[bs,seq] ---> mask
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

        return K, V, lb , output, d, td/24, mask, v_from_weight, LengList, PtList , outEmb[:,:,-self.n_cont:], loss, auc


class PKRNN_CC(EHREmbeddings):
    # check if VancoCLTensor is actually elimination rate using Matzke equation, which is 8.3e-4 * CrCL + 4.4e-3
    # This model preserves concentration continuity instead of mass continuity
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
        self.d_excess = torch.zeros(batchsize)
        if use_cuda:
            self.d_excess = self.d_excess.cuda()

    # embedding function
    def EmbedPatient_MB(self, input):
        return EHREmbeddings.EmbedPatients_MB(self, input)
    
    def forward(self, input):
        outEmb, lb, LengList, mask, d, td, v_from_weight, VancoClTensor, PtList = self.EmbedPatient_MB(input)
        self.d_excess.fill_(0)
        # --------------------- FEEDBACK RNN  --------------------- #
        h_0 = (torch.zeros(len(outEmb), self.hidden_size))
        if use_cuda:
            h_0 = h_0.cuda()
        hidden = [h_0]

        # --------------------- 2CM PK MODEL --------------------- #
        self.batchsize = outEmb.shape[0]
        # initialize K: [bs, 1] --> [bs, seq, 1]
        K0 = VancoClTensor[:,0] 
        V0 = v_from_weight[:,0]
        if use_cuda:
            K0 = K0.cuda()
            V0 = V0.cuda()
        
        td = td*24
        VancoConcnList = [torch.zeros((self.batchsize, 1))]
        if use_cuda:
            VancoConcnList[0] = VancoConcnList[0].cuda()
        KV = []

        
        for i in range(outEmb.shape[1]): # embed [bs, visit, emb_size+n_conts]
        #range(outEmb.shape[1])=# of visit
            if self.real_vanco_to_feedback:
                if i == 0: # no data yet
                    vanco_to_feedback = VancoConcnList[-1].detach()
                else:
                    
                    vanco_to_feedback = VancoConcnList[-1].detach() * (1 - mask[:, i-1:i].float()) + lb[:, i-1:i] * mask[:, i-1:i].float()
            
            else:
                vanco_to_feedback = VancoConcnList[-1].detach()
            Input_plus_Output = torch.cat((outEmb[:, i, :], vanco_to_feedback.reshape(-1,1).detach()), dim = 1)
            # outEmb [bs, in_size], VancoConcnList[-1] [bs, 1] ----> Input_plus_Output [bs, in_size+1]
            
            h_i = self.onecell(Input_plus_Output, hidden[-1]) # h_i.shape: [bs, hidden_size]
            kv = torch.exp(self.out(h_i))
            k = kv[:,0]
            v = kv[:,1]
            total_infusion = d[:, i] + self.d_excess[:self.batchsize]
            t_infusion = torch.clamp(total_infusion, max=td[:, i])
            A = torch.exp(-k*td[:,i])  #A.shape = [bs, 1]
            B = 1/k*(-torch.exp(-k*t_infusion) + 1) * torch.exp(-k*(td[:, i] - t_infusion)) / (v+1e-6)
            self.d_excess[:self.batchsize] = (total_infusion - t_infusion).detach()
            VancoConcn = VancoConcnList[-1]*A.unsqueeze(1) + B.unsqueeze(1)
            VancoConcnList.append(VancoConcn)  # [bs, 1]
            KV.append(kv)
            hidden.append(h_i)
            #print(A.shape, B.shape, DList[-1].shape, totalmass.shape) # [bs], [bs], [bs,1], [bs]
            

        output = torch.cat(VancoConcnList[1:], dim=1) # [bs, seq_length]

        #  --------------------- EXTRACT K,V FOR PLOTTING --------------------- #
        
        KV = torch.stack(KV).permute(1,0,2) #[bs, seq, 2]
        K, V = KV[:,:,0], KV[:,:,1]
        
         #  --------------------- GET FLATTEN OUTPUT AND FLATTEN LABEL --------------------- #
        mask = (lb != 0) #new
        outLeng = mask.shape[0] * mask.shape[1]
        flatten_mask = mask.view(outLeng)
        flatten_mask = torch.as_tensor(flatten_mask, dtype=torch.uint8)
        
        #[bs,seq] ---> mask
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
        self.d_excess = torch.zeros(batchsize)
        if use_cuda:
            self.d_excess = self.d_excess.cuda()

    # embedding function
    def EmbedPatient_MB(self, input):
        return EHREmbeddings.EmbedPatients_MB(self, input)
    
    def forward(self, input):
        outEmb, lb, LengList, mask, d, td, v_from_weight, VancoClTensor, PtList = self.EmbedPatient_MB(input)
        if use_cuda:
            input[0] = input[0].cuda()
            input[1] = input[1].cuda()
        age = input[0][:, :1, 2]*16.936469 + 58.239251
        weight = input[0][:, :1, 21]*30.519849 + 87.575200
        creatinine = input[0][..., 11]*1.418099 + 1.314668
        gender = (input[1][:, :1, :] == 5).sum(2).type(torch.float)*0.15+0.85
        dose_mask = (d != 0).type(torch.float)
        ccl = (140 - age)*weight*gender/72/creatinine
        
        # --------------------- FEEDBACK RNN  --------------------- #
        h_0 = (torch.zeros(len(outEmb), self.hidden_size))
        if use_cuda:
            h_0 = h_0.cuda()
        hidden = [h_0]
        self.d_excess.fill_(0)
        # --------------------- 2CM PK MODEL --------------------- #
        self.batchsize = outEmb.shape[0]
        td = td*24
        VancoConcnList = [torch.zeros((self.batchsize))]
        BList = [torch.zeros((self.batchsize))]        
        if use_cuda:
            VancoConcnList[0] = VancoConcnList[0].cuda()
            BList[0] = BList[0].cuda()
        ETA, K1, K2, V1, V2 = [], [], [], [], []
        auc = torch.zeros(self.batchsize)
        if use_cuda:
            auc = auc.cuda()

         
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
            # outEmb [bs, in_size], VancoConcnList[-1] [bs, 1] ----> Input_plus_Output [bs, in_size+1]
            
            h_i = self.onecell(Input_plus_Output, hidden[-1]) # h_i.shape: [bs, hidden_size]
            eta = self.out(h_i)

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
            total_infusion = d[:, i] + self.d_excess[:self.batchsize]
            t_infusion = torch.clamp(total_infusion, max=td[:, i])
            C_prev = (VancoConcnList[-1] - C4 * BList[-1]) / (C3 - C4)
            D_prev = (BList[-1] * C3 - VancoConcnList[-1]) / (C3 - C4)
            C_peak = (C_prev + C1) * torch.exp(lam1 * t_infusion) - C1
            D_peak = (D_prev + C2) * torch.exp(lam2 * t_infusion) - C2
            t_decay = td[:, i] - t_infusion
            C = C_peak * torch.exp(lam1 * t_decay)
            D = D_peak * torch.exp(lam2 * t_decay)
            BList.append(C + D)
            A = C * C3 + D * C4
            VancoConcnList.append(A) # [bs, 1]
            self.d_excess[:self.batchsize] = (total_infusion - t_infusion).detach()

            # --- Analytical AUC for this interval ---
            # Infusion phase integral (0 to t_infusion)
            term1_inf = (C_prev + C1) / lam1 * (torch.exp(lam1 * t_infusion) - 1) - C1 * t_infusion
            term2_inf = (D_prev + C2) / lam2 * (torch.exp(lam2 * t_infusion) - 1) - C2 * t_infusion
            auc_inf = term1_inf * C3 + term2_inf * C4
            # Decay phase integral (0 to t_decay)
            term1_dec = (C_peak / lam1) * (torch.exp(lam1 * t_decay) - 1)
            term2_dec = (D_peak / lam2) * (torch.exp(lam2 * t_decay) - 1)
            auc_dec = term1_dec * C3 + term2_dec * C4
            auc = auc + auc_inf + auc_dec

            ETA.append(eta)
            K1.append(k1)
            K2.append(k2)
            V1.append(v1)
            V2.append(v2)
                      
            hidden.append(h_i)
            
        # convert lists to tensors
        K1 = torch.stack((K1)).permute(1,0)
        K2 = torch.stack((K2)).permute(1,0)
        V1 = torch.stack((V1)).permute(1,0)
        V2 = torch.stack((V2)).permute(1,0)

 
        output = torch.stack(VancoConcnList[1:]).permute(1,0) # [bs, seq_length]

        #  --------------------- EXTRACT ETA FOR PLOTTING --------------------- #
        ETA = torch.stack(ETA).permute(1,0,2) #[bs, seq, 2]
        ETA1, ETA2, ETA3, ETA4 = ETA[:,:,0], ETA[:,:,1], ETA[:,:,2], ETA[:,:,3]
        
        #  --------------------- GET FLATTEN OUTPUT AND FLATTEN LABEL --------------------- #
        mask = (lb != 0)
        outLeng = mask.shape[0] * mask.shape[1]
        flatten_mask = mask.view(outLeng)
        flatten_mask = torch.as_tensor(flatten_mask, dtype=torch.uint8)
        
        #[bs,seq] ---> mask
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

        return ETA1, ETA2, ETA3, ETA4, K1, K2, V1, V2, lb, output, ccl, d, td/24, mask, LengList, PtList, loss, auc


class PKRNN_2CM_Sylvester(EHREmbeddings):
    def __init__(self, input_size, embed_dim=8, hidden_size=64, n_layers=1, dropout_r=0, n_cont=2, batchsize=500,
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
        self.k2_base = paramdict.get('k2_base', 6.99)
        self.v1_base = paramdict.get('v1_base', 33.1)
        self.k1_base = paramdict.get('k1_rate_base', 3.96)
        
        self.d_excess = torch.zeros(batchsize)
        if use_cuda:
            self.d_excess = self.d_excess.cuda()
        #print(self.embed.weight.shape)
        #print('size', self.in_size)

    # embedding function
    def EmbedPatient_MB(self, input):
        return EHREmbeddings.EmbedPatients_MB(self, input)

    def forward(self, input):
        outEmb, lb, LengList, mask, d, td, v_from_weight, VancoClTensor, PtList = self.EmbedPatient_MB(input)
        self.d_excess.fill_(0)
        #print('vanco_dose', d)
        #print('vanco_level', lb)

        #print('ziqian2', Cont_mTensor)
        device = self.embed.weight.device 
        
        # Cont_mTensor: mask tensor for continous values
        # IVPOtensor: differentiate dose is PO or IV: This is not used currently. But future to switch po to IV
        if use_cuda:
            input[0] = input[0].cuda()
            input[1] = input[1].cuda()
        age = input[0][:, :1, 2]*16.936469 + 58.239251
        weight = input[0][:, :1, 21]*30.519849 + 87.575200
        creatinine = input[0][..., 11]*1.418099 + 1.314668
        gender = (input[1][:, :1, :] == 5).sum(2).type(torch.float)*0.15+0.85
        # dose_mask = (d != 0).type(torch.float)
        ccl = (140 - age)*weight*gender/72/creatinine
        #print('ccl', ccl)
        ccl = ccl.to(device)      
        
        #print(Cont_mTensor)
        #Cont_mTensor = Cont_mTensor.view(-1, 1, 1)  # [B, 1, 1]
        #Cont_mTensor = Cont_mTensor.expand(-1, outEmb.size(1), 1).to(device)  # [B, T, 1]
        #print(Cont_mTensor.shape)
        #print('outEmb', outEmb.shape)
        # outEmb = torch.cat([outEmb, Cont_mTensor], dim=2)  ##MN Added Mask tansor 8/24/2025
        
        # --------------------- FEEDBACK RNN  --------------------- #
        h_0 = (torch.zeros(len(outEmb), self.hidden_size)).to(device)
        hidden = [h_0]
        # --------------------- 2CM PK MODEL --------------------- #
        self.batchsize = outEmb.shape[0]
        td = td*24
        state = [torch.zeros((self.batchsize, 2)).to(device)]
        ETA, K1, K2, V1, V2 = [], [], [], [], []
        
        for i in range(outEmb.shape[1]): # embed [bs, visit, emb_size+n_conts]
        #range(outEmb.shape[1])=# of visit
            if self.real_vanco_to_feedback:
                if i == 0: # no data yet
                    vanco_to_feedback = state[-1][:,0].detach()
                else:
                    vanco_to_feedback = state[-1][:,0].detach() * (1 - mask[:, i-1].float()) + lb[:, i-1] * mask[:, i-1].float()
            
            else:
                vanco_to_feedback = state[-1][:,0].detach()
            Input_plus_Output = torch.cat((outEmb[:, i, :], vanco_to_feedback.reshape(-1,1).detach()), dim = 1)       
            h_i = self.onecell(Input_plus_Output, hidden[-1])
            eta = self.out(h_i).to(device)
            eta1 = eta[:,0]
            eta2 = eta[:,1]
            eta3 = eta[:,2]
            eta4 = eta[:,3]
            #PK parameters: k1, v1, v2, R
            v1 = self.v1_base * torch.exp(eta1)
            k1 = self.k1_base * ccl[:,i] / 100 * torch.exp(eta2)
            v2 = 48.3 * torch.exp(eta4)
            k2 = self.k2_base * torch.exp(eta3)
            m11 = -(k1 + k2) / v1
            m12 = k2 / v1
            m21 = k2 / v2
            m22 = -k2 / v2
            trace = m11 + m22
            det = m11 * m22 - m12 * m21
            sqrt_discriminant = torch.clamp(torch.sqrt(trace ** 2 - 4 * det), min=1e-3)
            l1 = 0.5 * (trace + sqrt_discriminant)
            l2 = 0.5 * (trace - sqrt_discriminant)
            denom = l1 - l2
            total_infusion = d[:, i] + self.d_excess[:self.batchsize]
            t_infusion = torch.clamp(total_infusion, max=td[:, i])
            t_rem = td[:, i] - t_infusion
            
            w1 = torch.exp(l1 * td[:, i]) / denom
            w2 = torch.exp(l2 * td[:, i]) / denom
            
            # Integral of exponential: \int e^{\lambda (T-t)} ds
            # = \frac{e^{\lambda td} - e^{\lambda t_{rem}}}{\lambda \times denom}
            # When \lambda = 0, limit is \frac{t_{infusion} \times e^{\lambda t_{rem}}}{denom}
            # To prevent PyTorch autograd from computing NaN in the false branch of torch.where,
            # we must guarantee the divisor is never 0 in ANY computation path.
            
            l1_safe = torch.where(torch.abs(l1) < 1e-6, -torch.ones_like(l1), l1)
            l2_safe = torch.where(torch.abs(l2) < 1e-6, -torch.ones_like(l2), l2)
            
            dw1_std = (torch.exp(l1_safe * td[:, i]) - torch.exp(l1_safe * t_rem)) / (denom * l1_safe)
            dw1_lim = t_infusion / denom * torch.exp(l1 * t_rem)
            dw1 = torch.where(torch.abs(l1) < 1e-6, dw1_lim, dw1_std)
            
            dw2_std = (torch.exp(l2_safe * td[:, i]) - torch.exp(l2_safe * t_rem)) / (denom * l2_safe)
            dw2_lim = t_infusion / denom * torch.exp(l2 * t_rem)
            dw2 = torch.where(torch.abs(l2) < 1e-6, dw2_lim, dw2_std)
            
            p11 = w1 * (m11 - l2) - w2 * (m11 - l1)
            p12 = (w1 - w2) * m12
            p21 = (w1 - w2) * m21
            p22 = w1 * (m22 - l2) - w2 * (m22 - l1)
            # The input matrix G applies (M - \lambda I) to the integral terms dw1 and dw2
            g1 = dw1 * (m11 - l2) - dw2 * (m11 - l1)
            g2 = dw1 * m21 - dw2 * m21

           
            # C1 = state * P + Input * G
            C1 = state[-1][:, 0] * p11 + state[-1][:, 1] * p12 + g1 * 1000 / v1
            C2 = state[-1][:, 0] * p21 + state[-1][:, 1] * p22 + g2 * 1000 / v1
            state.append(torch.stack((C1, C2), dim=1))
            self.d_excess[:self.batchsize] = (total_infusion - t_infusion).detach()
            #print('A', A[:10])
            #print('C3', C3[:10])
            #print('D', D[:10])
            #print('C4', C4[:10])
            #test_MH
            #print('VancoConcnList', VancoConcnList[:5])
            ETA.append(eta)
            K1.append(k1)
            K2.append(k2)
            V1.append(v1)
            V2.append(v2)                     
            hidden.append(h_i) 
            
        #print("Eta:", ETA[1])
        #print('K1:', K1[1])
        #print('K2:', K2[1])
        #print('V1:', V1[1])
        #print('V2:', V2[1])

        # convert lists to tensors
        K1 = torch.stack((K1)).permute(1,0).to(device)
        K2 = torch.stack((K2)).permute(1,0).to(device)
        V1 = torch.stack((V1)).permute(1,0).to(device)
        V2 = torch.stack((V2)).permute(1,0).to(device)
        #print('VancoConcnList', VancoConcnList)
        #test_MH
        output = torch.stack(state[1:], dim=1)[..., 0] # [bs, seq_length]
        #print('Output', output)
        #print('Output Shape', output.shape)
        #test_MH
        #print('Label', lb[1])
        #  --------------------- EXTRACT ETA FOR PLOTTING --------------------- #
        ETA = torch.stack(ETA).permute(1,0,2) #[bs, seq, 2]
        ETA1, ETA2, ETA3, ETA4 = ETA[:,:,0], ETA[:,:,1], ETA[:,:,2], ETA[:,:,3]       
        #  --------------------- GET FLATTEN OUTPUT AND FLATTEN LABEL --------------------- #
        mask = (lb != 0)
        
        outLeng = mask.shape[0] * mask.shape[1]
        flatten_mask = mask.view(outLeng)
        flatten_mask = torch.as_tensor(flatten_mask, dtype=torch.uint8)
        
        flatten_output = output.contiguous().view(outLeng)[flatten_mask]
        #print('flatten_output', flatten_output)
        
        flatten_label = lb.view(outLeng)[flatten_mask]
        #print('flatten_label', flatten_label)

        # Added this as some results gives outlier on 11/22/25
        #outlier_mask = flatten_output <= 100 
        #flatten_output = flatten_output[outlier_mask]
        #flatten_label  = flatten_label[outlier_mask]

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
    
    
class VTDM(nn.Module):
    def __init__(self, stats_data, dictionary):
        super().__init__()
        self.eta1 = nn.Parameter(torch.zeros(1))
        self.eta2 = nn.Parameter(torch.zeros(1))
        self.eta3 = nn.Parameter(torch.zeros(1))
        self.V2 = 48.3
        self.A = [torch.zeros(1)]
        self.B = [torch.zeros(1)]
        self.male_code = dictionary['Male']
        self.creatinine_idx = stats_data.INDEX[stats_data.FEATURES == "CREATININE B"].values[0]
        self.creatine_mean = stats_data.MEANS[stats_data.FEATURES == "CREATININE B"].values[0]
        self.creatine_std = stats_data.STD[stats_data.FEATURES == "CREATININE B"].values[0]
        self.age_idx = stats_data.INDEX[stats_data.FEATURES == "AGE"].values[0]
        self.age_mean = stats_data.MEANS[stats_data.FEATURES == "AGE"].values[0]
        self.age_std = stats_data.STD[stats_data.FEATURES == "AGE"].values[0]
        self.wt_idx = stats_data.INDEX[stats_data.FEATURES == "WT"].values[0]
        self.wt_mean = stats_data.MEANS[stats_data.FEATURES == "WT"].values[0]
        self.wt_std = stats_data.STD[stats_data.FEATURES == "WT"].values[0]
        self.d_excess = 0
        self.AUC = 0
    
    def reset(self):
        with torch.no_grad():
            self.eta1.fill_(0)
            self.eta2.fill_(0)
            self.eta3.fill_(0)
        self.A = [torch.zeros(1)]
        self.B = [torch.zeros(1)]
        self.T = [torch.zeros(1)]
        self.dose = []
        self.AUC = 0
        self.d_excess = 0


    def reset_state(self):
        self.A = [torch.zeros(1)]
        self.B = [torch.zeros(1)]
        self.T = [torch.zeros(1)]
        self.dose = []
        self.AUC = 0
        self.d_excess = 0

    def ccl(self, cont, cat):
        male = (cat == self.male_code).sum() > 0
        creatinine = cont[..., self.creatinine_idx] * self.creatine_std + self.creatine_mean
        wt = cont[..., self.wt_idx] * self.wt_std + self.wt_mean
        age = cont[..., self.age_idx] * self.age_std + self.age_mean
        ccl = (140 - age) * wt / 72 / creatinine
        ccl = ccl * 0.85 if not male else ccl
        return ccl
    
    def dense_update(self, dose, tdiff, ccl):
        V1 = 33.1 * torch.exp(self.eta1)
        Q = 6.99 * torch.exp(self.eta3)
        k10 = ccl * 3.96e-2 * torch.exp(self.eta2)/V1
        k12 = Q/V1
        k21 = Q/self.V2
        R = 1000/V1
        delta = torch.sqrt((k10+k12+k21)**2 - 4*k10*k21)
        lambda1 = (-k10-k12-k21-delta)/2
        lambda2 = (-k10-k12-k21+delta)/2
        # print(lambda1, lambda2)
        # print(lambda1, lambda2, k10, k12, k21)
        c1 = -R*k12/lambda1/delta
        c2 = R*k12/lambda2/delta
        c3 = (lambda1 + k21)/k12
        c4 = (lambda2 + k21)/k12
        int_elapsed = int(tdiff)
        time_grid = torch.cat([torch.arange(1, int_elapsed + 1), tdiff.reshape(1)])
        total_infusion = dose + self.d_excess
        t_infusion = torch.clamp(total_infusion, max=tdiff)
        C_prev = (self.A[-1][-1] - c4 * self.B[-1][-1]) / (c3 - c4)
        D_prev = (self.B[-1][-1] * c3 - self.A[-1][-1]) / (c3 - c4)
        C_infusing = (C_prev + c1) * torch.exp(lambda1 * time_grid) - c1
        C_peak = (C_prev + c1) * torch.exp(lambda1 * t_infusion) - c1
        C_decaying = C_peak * torch.exp(lambda1 * (time_grid - t_infusion))
        C = torch.where(time_grid <= t_infusion, C_infusing, C_decaying)
        D_infusing = (D_prev + c2) * torch.exp(lambda2 * time_grid) - c2
        D_peak = (D_prev + c2) * torch.exp(lambda2 * t_infusion) - c2
        D_decaying = D_peak * torch.exp(lambda2 * (time_grid - t_infusion))
        D = torch.where(time_grid <= t_infusion, D_infusing, D_decaying)
        self.d_excess = (total_infusion - t_infusion)
        self.B.append(C + D)
        self.A.append(C * c3 + D * c4)
        self.T.append(self.T[-1][-1] + time_grid)
        self.dose.append(torch.cat([dose.reshape(1), torch.zeros(int_elapsed)]))
        
        # --- Analytical AUC for this interval ---
        # Infusion phase: integral of concentration from 0 to t_infusion
        term1_inf = (C_prev + c1) / lambda1 * (torch.exp(lambda1 * t_infusion) - 1) - c1 * t_infusion
        term2_inf = (D_prev + c2) / lambda2 * (torch.exp(lambda2 * t_infusion) - 1) - c2 * t_infusion
        auc_inf = term1_inf * c3 + term2_inf * c4
        # Decay phase: integral from t_infusion to tdiff
        t_decay = tdiff - t_infusion
        term1_dec = (C_peak / lambda1) * (torch.exp(lambda1 * t_decay) - 1)
        term2_dec = (D_peak / lambda2) * (torch.exp(lambda2 * t_decay) - 1)
        auc_dec = term1_dec * c3 + term2_dec * c4
        self.AUC = self.AUC + auc_inf + auc_dec
        
    def update(self, dose, tdiff, ccl):
        V1 = 33.1 * torch.exp(self.eta1)
        Q = 6.99 * torch.exp(self.eta3)
        k10 = ccl * 3.96e-2 * torch.exp(self.eta2)/V1
        k12 = Q/V1
        k21 = Q/self.V2
        R = 1000/V1
        delta = torch.sqrt((k10+k12+k21)**2 - 4*k10*k21)
        lambda1 = (-k10-k12-k21-delta)/2
        lambda2 = (-k10-k12-k21+delta)/2
        # print(lambda1, lambda2)
        # print(lambda1, lambda2, k10, k12, k21)
        c1 = -R*k12/lambda1/delta
        c2 = R*k12/lambda2/delta
        c3 = (lambda1 + k21)/k12
        c4 = (lambda2 + k21)/k12
        total_infusion = dose + self.d_excess
        t_infusion = torch.clamp(total_infusion, max=tdiff)
        C_prev = (self.A[-1] - c4 * self.B[-1]) / (c3 - c4)
        D_prev = (self.B[-1] * c3 - self.A[-1]) / (c3 - c4)
        C_peak = (C_prev + c1) * torch.exp(lambda1 * t_infusion) - c1
        D_peak = (D_prev + c2) * torch.exp(lambda2 * t_infusion) - c2
        C = C_peak * torch.exp(lambda1 * (tdiff - t_infusion))
        D = D_peak * torch.exp(lambda2 * (tdiff - t_infusion))
        self.B.append(C + D)
        self.d_excess = (total_infusion - t_infusion).detach()
        self.A.append(C * c3 + D * c4)
        
        # --- Analytical AUC for this interval ---
        # Infusion phase: integral of concentration from 0 to t_infusion
        term1_inf = (C_prev + c1) / lambda1 * (torch.exp(lambda1 * t_infusion) - 1) - c1 * t_infusion
        term2_inf = (D_prev + c2) / lambda2 * (torch.exp(lambda2 * t_infusion) - 1) - c2 * t_infusion
        auc_inf = term1_inf * c3 + term2_inf * c4
        # Decay phase: integral from t_infusion to tdiff
        t_decay = tdiff - t_infusion
        term1_dec = (C_peak / lambda1) * (torch.exp(lambda1 * t_decay) - 1)
        term2_dec = (D_peak / lambda2) * (torch.exp(lambda2 * t_decay) - 1)
        auc_dec = term1_dec * c3 + term2_dec * c4
        self.AUC = self.AUC + auc_inf + auc_dec
    
    def predict(self, dose_tensor, tdiff_tensor, ccl_tensor):
        self.reset_state()
        with torch.no_grad():
            for i in range(len(dose_tensor)):
                self.update(dose_tensor[i], tdiff_tensor[i], ccl_tensor[i])
        return torch.cat(self.A)[1:]
    
    def dense_predict(self, dose_tensor, tdiff_tensor, ccl_tensor):
        self.reset_state()
        self.A[0] = self.A[0].reshape(1, 1)
        self.B[0] = self.B[0].reshape(1, 1)
        with torch.no_grad():
            for i in range(len(dose_tensor)):
                self.dense_update(dose_tensor[i], tdiff_tensor[i], ccl_tensor[i])
        return torch.cat(self.A), torch.cat(self.dose), torch.cat(self.T)[1:]
    
    def loss(self, dose_tensor, tdiff_tensor, ccl_tensor, vanco_level_tensor):
        self.reset_state()
        index = int(vanco_level_tensor.nonzero()[0])
        for i in range(index+1):
            self.update(dose_tensor[i], tdiff_tensor[i], ccl_tensor[i])
        loss = (self.A[index] - vanco_level_tensor[index])**2/4**2 + self.eta1**2/0.12 + self.eta2**2/0.149 + self.eta3**2/0.416
        return loss

    def online_predict(self, dose_tensor, tdiff_tensor, ccl_tensor, vanco_level_tensor, lr=1e-3):
        # Setup optimizer
        optimizer = optim.SGD([self.eta1, self.eta2, self.eta3], lr=lr)
        
        final_predictions = []
        online_auc = 0  # Accumulates AUC using pre-update parameters
        self.reset_state()
        
        # Main loop over time steps
        for i in range(len(dose_tensor)):
            
            # 1. Predict (using current parameters, BEFORE seeing measurement)
            # Save AUC before this step to capture the delta
            auc_before = self.AUC
            with torch.no_grad():
                self.update(dose_tensor[i], tdiff_tensor[i], ccl_tensor[i])
                pred = self.A[-1]
                final_predictions.append(pred)
                # Capture this step's AUC contribution using pre-update parameters
                online_auc = online_auc + (self.AUC - auc_before)

            # 2. Update parameters if measurement exists
            if vanco_level_tensor[i] > 0:
                # Perform 10 SGD updates
                for _ in range(10):
                    optimizer.zero_grad()
                    self.reset_state()
                    
                    # Re-simulate from start to current step i
                    current_loss = 0
                    valid_measurements_count = 0
                    
                    for k in range(i + 1):
                        self.update(dose_tensor[k], tdiff_tensor[k], ccl_tensor[k])
                        if vanco_level_tensor[k] > 0:
                            # Standardize loss terms as in original loss function
                            step_loss = (self.A[-1] - vanco_level_tensor[k])**2/4**2
                            current_loss += step_loss
                            valid_measurements_count += 1
                    
                    # Add regularization (priors)
                    reg_loss = self.eta1**2/0.12 + self.eta2**2/0.149 + self.eta3**2/0.416
                    
                    total_loss = current_loss + reg_loss
                    total_loss.backward()
                    
                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_([self.eta1, self.eta2, self.eta3], 1.0)
                    
                    optimizer.step()
                    
                    # Clamp parameters to prevent explosion (e.g. exp(10) is huge)
                    with torch.no_grad():
                        self.eta1.clamp_(-3, 3)
                        self.eta2.clamp_(-3, 3)
                        self.eta3.clamp_(-3, 3)
                
                # 3. Finalize State: Re-run history with NEW parameters to be ready for step i+1
                self.reset_state()
                with torch.no_grad():
                    for k in range(i + 1):
                        self.update(dose_tensor[k], tdiff_tensor[k], ccl_tensor[k])
        
        return torch.cat(final_predictions), online_auc