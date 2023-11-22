from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import pickle as pkl
import torch


class Example(Dataset):
    def __init__(self):
        self.data = [pkl.load(open("sample_data.pkl", 'rb'))]

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.data[item]


def collate_batch(batch):
    conts, cats, labels, doses, tds, vs, vanco_ers, pts, lengs = [[] for _ in range(9)]
    for (cont, cat, label, dose, td, v, vanco_er, pt, leng) in batch:
        conts.append(torch.from_numpy(cont))
        cats.append(torch.from_numpy(cat))
        labels.append(torch.from_numpy(label))
        doses.append(torch.from_numpy(dose))
        tds.append(torch.from_numpy(td))
        vs.append(torch.from_numpy(v))
        vanco_ers.append(torch.from_numpy(vanco_er))
        pts.append(torch.from_numpy(pt))
        lengs.append(torch.from_numpy(leng))
    pad = partial(pad_sequence, batch_first=True)
    conts = pad(conts)
    cats = pad(cats)
    labels = pad(labels)
    doses = pad(doses)
    tds = pad(tds)
    vs = pad(vs)
    vanco_ers = pad(vanco_ers)
    pts = torch.cat(pts)
    lengs = torch.cat(lengs)
    return conts, cats, labels, doses, tds, vs, vanco_ers, pts, lengs

