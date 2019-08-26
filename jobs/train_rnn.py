

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

TAG = 'mxrnn'
save_path = '%s/%s/%s.pth' % (CKPT_STORAGE, TAG, fileName(sys.argv[1]))
print('Saving to:')
print(save_path)

n_lag = 24
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset
dset = DayHistory(segs, 'train', bsize=32)#.generator()
valset = DayHistory(segs, 'test', bsize=32)#.generator()

from models.temporal.RNN import *

model = RNN(hidden_size=256, steps=len(segs)).to(device)
model.device = device
criterion, opt, sch = model.params(lr=0.001)

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
	
	batch = np.array(batch).swapaxes(0, 1)
	# labels = np.array(labels).swapaxes(0, 1)
	batch = list(map(n2t, batch))
	labels = n2t(labels)
		
	lmasks = ((~torch.isnan(labels))).type(torch.bool).cuda()

	return batch, labels, lmasks

inds = trainable_inds(dset.data)
eval_inds = trainable_inds(valset.data)

print('Pre-evaluate:')
# evf_full = lambda: evaluate_v2(
# 	eval_inds, valset, model, criterion, 
# 	format_batch)
evf = lambda: evaluate_v2(
	eval_inds, valset, model, 
	lambda preds, label, mask: criterion(preds[:, -1][mask[:, -1]], label[:, -1][mask[:, -1]]), 
	format_batch)
best_eval = evf()

print('Train:')
eval_mse = [best_eval]
for epoch in range(args.epochs):
	model.train()
	shuffle(inds)
	for ii, (di, hi) in enumerate(inds):
		# forward
		batch, labels, lmasks = format_batch([(di, hi)], dset.data)
		preds = model(batch)

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
		torch.save(model.state_dict(), save_path)
		best_eval = last_eval
	eval_mse.append(last_eval)


logfl = '%s/%s/%s_log.json' % (LOG_PATH, TAG, fileName(sys.argv[1]))
print('Log:', logfl)
with open(logfl, 'w') as fl:
	json.dump([
		eval_mse,
		best_eval,
	], fl, indent=4)
