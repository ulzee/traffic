

import os, sys
import dgl
sys.path.append('.')

from glob import glob
from configs import *
from tqdm import tqdm
from utils import *
import numpy as np
# from dataset import *k
from days import *
from time import time
tqdm.monitor_interval = 0
import torch
import json
from time import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import argparse
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('graph', type=str)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()


# graph_file = '/beegfs/ua349/archive/data/graphs/400861-400948_n2.json'
segs, adjlist = read_graph(args.graph, verbose=False, named_adj=True)
rootseg = segs[0]

TAG = 'lgcn'
save_path = '%s/%s/%s.pth' % (CKPT_STORAGE, TAG, fileName(sys.argv[1]))
print('Saving to:')
print(save_path)

n_lag = 24
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset
dset = DayHistory(segs, 'train', bsize=32)#.generator()
valset = DayHistory(segs, 'test', bsize=32)#.generator()

from models.LGCN import LGCN

g = dgl.DGLGraph()
g.add_nodes(len(segs))
source_list  = []
sink_list = []
iindex = { seg:ind for ind, seg in enumerate(segs) }
for source, ls in adjlist.items():
    # self-edges
    source_list.append(iindex[source])
    sink_list.append(iindex[source])

    for sink in ls:
        source_list.append(iindex[source])
        sink_list.append(iindex[sink])
        # TWOWAY
        source_list.append(iindex[sink])
        sink_list.append(iindex[source])
g.add_edges(source_list, sink_list) # add 4 edges 0->1, 0->2, 0->3, 0->4
# g

n_edges = g.number_of_edges()

# normalization
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
norm = norm.cuda()
g.ndata['norm'] = norm.unsqueeze(1)
# norm

lgcn = LGCN(g,
    in_feats=1,
    n_hidden=64,
    n_classes=1,
    n_layers=2,
    activation=F.relu,
    single_rnn=False,
    dropout=0.5).cuda()
criterion = torch.nn.MSELoss()
opt = torch.optim.Adam(
    lgcn.parameters(),
    lr=1e-3,
    weight_decay=5e-4)


# Train
n2t = lambda arr: torch.from_numpy(np.array(arr)).cuda().float()
def format_batch(inds, data, squeeze=True):
    batch = []
    labels = []

    for di, hi in inds:
        X = data[di][hi-n_lag:hi].copy()
        X[np.isnan(X)] = -1
        Y = data[di][hi-n_lag+1:hi+1].copy()
        batch.append(X)
        labels.append(Y)

    batch = np.array(batch).swapaxes(1, 2)
    labels = np.array(labels).swapaxes(1, 2)
    batch = n2t(batch).squeeze(0)
    labels = n2t(labels).squeeze(0)

    lmasks = ((~torch.isnan(labels))).type(torch.bool).cuda()

#     print(batch.size(), labels.size())
    return batch, labels, lmasks

inds = trainable_inds(dset.data)
eval_inds = trainable_inds(valset.data)

print('Pre-evaluate:')
evf = lambda: evaluate_v2(
	eval_inds, valset, lgcn, 
	lambda preds, label, mask: criterion(preds[:, -1][mask[:, -1]], label[:, -1][mask[:, -1]]), 
	format_batch)
best_eval = evf()

print('Train:')
eval_mse = [best_eval]
for epoch in range(args.epochs):
	lgcn.train()
	shuffle(inds)
	for ii, (di, hi) in enumerate(inds):
		# forward
		batch, labels, lmasks = format_batch([(di, hi)], dset.data)
		preds = lgcn(batch)

		loss = criterion(preds[lmasks], labels[lmasks])

		opt.zero_grad()
		loss.backward()
		opt.step()

		sys.stdout.write('[%d/%d]: %d/%d  \r' % (
			epoch+1, args.epochs,
			ii, len(inds)))
	sys.stdout.write('\n')
	sys.stdout.flush()

	last_eval = evf()
	if last_eval < best_eval:
		torch.save(lgcn.state_dict(), save_path)
		best_eval = last_eval
	eval_mse.append(last_eval)


logfl = '%s/%s/%s_log.json' % (LOG_PATH, TAG, fileName(sys.argv[1]))
print('Log:', logfl)
with open(logfl, 'w') as fl:
	json.dump([
		eval_mse,
		best_eval,
	], fl, indent=4)
