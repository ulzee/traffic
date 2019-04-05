
import os, sys
sys.path.append('.')
from glob import glob
from configs import *
from tqdm import tqdm
from utils import *
import numpy as np
from dataset import *
from time import time
tqdm.monitor_interval = 0
import torch
import json
from time import time
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

graph_file = sys.argv[1]

SROUTE, ADJ = read_graph(graph_file,
                         verbose=False, named_adj=True)
# SROUTE, ADJ = complete_graph(SROUTE, ADJ)
# graph = show_graph(SROUTE, ADJ)

EPS = 10
LAG = 18 + 1
hops = int(graph_file[:-5].split('_n')[1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TAG = 'mpfcast'
# save_path = '%s/%s/%s.pth' % (CKPT_STORAGE, TAG, fileName(sys.argv[1]))
# print('Saving to:')
# print(save_path)

dset = SpotHistory(
    SROUTE, 'train', 32,
    clip_hours=8,
    lag=LAG, res=10).generator()
valset = SpotHistory(
    SROUTE, 'test', 32,
    clip_hours=8,
    lag=LAG, res=10, shuffle=False).generator()

from models.temporal.RNN import *
from models.MPRNN import *
from models.Variants import *

HSIZE = 128
# AUTO_ITER = hops + 1
AUTO_ITER = 3

model = MPRNN_FCAST(
    nodes=SROUTE, adj=ADJ,
#     rnnmdl=RNN_HDN_LOSSY,
#     mpnmdl=MPN_DEEP_LOSSY,
    iters=AUTO_ITER,
    iter_indep=False,

    hidden_size=HSIZE,
    verbose=True)

model.to(device)
model.device = device
model.hops = hops

criterion, opt, sch = model.params(lr=0.001)
sch = optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.5)

def lock_subgraph(lock=True):
    pfile = graph_file.replace('_n%d' % hops, '_n%d' % (hops-1))
    pvs, padj = read_graph(pfile, verbose=False, named_adj=True)
    pfringes = find_fringes(pvs, padj, twoway=True)
    subvs = [vert for vi, vert in enumerate(pvs) if vi not in pfringes]

    indmap = {}
    nLocked = 0
    nTot = 0
    for ni, name in enumerate(model.nodes):
        indmap[ni] = name
    for name, param in model.named_parameters():
        ind = int(name.split('.')[1])
        vert = indmap[ind]
        if vert in subvs:
            param.requires_grad = not lock
#             if lock:
        nLocked += not param.requires_grad
        nTot += 1

    print('Locked: %d/%d' % (nLocked, nTot))

evf = lambda: evaluate(
    valset, model,
    crit=lambda _y, y: criterion(_y[:, :, 0], y[:, :, 0]).item())

print('Pre-evaluate:')
best_eval = evf()

if hops > 1:
    print('With trasnfer:')
    model.load_prior()
    _ = evf()
    lock_subgraph(True)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

print('Train:')
train_mse = []
eval_mse = []
eval_mape = []

for eii  in range(EPS):

    bls = []
    # sch.step()
    print('LR', get_lr(opt))

    t0 = time()
    for bii, batch in enumerate(dset):
        model.train()
        Xs, Ys = model.format_batch(batch)

        outputs = model(Xs)

        opt.zero_grad()
        loss = criterion(outputs, Ys)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 0.25)

        opt.step()

        bls.append(loss.item())
        bmse = ''
        if bii == len(dset) - 1:
            bmse = (10 ** 2 * np.mean(bls))
            train_mse.append(bmse)
            bmse = '(avg %.2f  %.1fs)' % (bmse, time() - t0)
        sys.stdout.write('[%d/%d : %d/%d] - L%.2f %s  \r' % (
            eii+1, EPS,
            bii+1, len(dset),
            10**2 * loss.item(),
            bmse
        ))
    sys.stdout.write('\n')

    last_eval = evf()
    print('Last best:', best_eval, '(now: %.2f)' % last_eval)
    if last_eval < best_eval:
        best_eval = last_eval
    eval_mse.append(last_eval)

    sys.stdout.flush()

model.save()

with open('%s/%s/%s_log.json' % (LOG_PATH, TAG, fileName(sys.argv[1])), 'w') as fl:
	json.dump([
		eval_mse,
		train_mse,
		# np.mean(sqerr),
	], fl, indent=4)
