import pickle as pkl
import torch
from model import VTDM


_, _, label, dose, timediff, _, vanco_er, _, _ = pkl.load(open('sample_data.pkl', 'rb'))
label = torch.from_numpy(label)
dose = torch.from_numpy(dose)
timediff = torch.from_numpy(timediff)
vanco_er = torch.from_numpy(vanco_er)
ccl = (vanco_er - 0.0044)/0.00083
vtdm = VTDM()
vtdm.inference(dose, timediff, ccl, label)
prediction = vtdm.predict(dose, timediff, ccl)