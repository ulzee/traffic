
from glob import glob
from configs import *
from tqdm import tqdm
from utils import *
import numpy as np
# import matplotlib.pyplot as plt
from dataset import *
from time import time
tqdm.monitor_interval = 0
import torch
import json
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

with open(sys.argv[1]) as fl:
    SROUTE, adjspec = json.load(fl)

ADJ = {}
for ind, ls in adjspec:
    ADJ[SROUTE[ind]] = [SROUTE[ii] for ii in ls]

print(SROUTE)
print(ADJ)

DENSE = False
EPS = 120
LAG = 24 + 1
HSIZE = 128
STOPS = len(SROUTE)
device = torch.device(sys.argv[2] if torch.cuda.is_available() else "cpu")

dset = SpotHistory(SROUTE, 'train', 32, lag=LAG, res=10).generator()
evalset = SpotHistory(SROUTE, 'test', 32, lag=LAG, res=10).generator()

from models.temporal.RNN import *
from models.MPRNN import *

model = MPRNN(
    nodes=SROUTE, adj=ADJ,
    hidden_size=HSIZE,

    rnnmdl=RNN_MIN,
    mpnmdl=MP_DENSE,

    verbose=True).to(device)
model.device = device
criterion, opt, sch = model.params(lr=0.05)

model.clear_stats()
evf = lambda: evaluate(evalset, model, crit=lambda _y, y: criterion(_y[:, :, 0], y[:, :, 0]).item())
_ = evf()

losses = []
eval_mse = []
eval_mape = []
from numpy.random import randint
for eii  in range(EPS):
    for bii, batch in enumerate(dset):
        model.train()
        Xs, Ys = model.format_batch(batch)

        outputs = model(Xs)

        opt.zero_grad()
        loss = criterion(outputs, Ys)
        loss.backward()
        losses.append(loss.item())
        opt.step()

        sys.stdout.write('[%d/%d : %d/%d] - L%.2f      \r' % (
            eii+1, EPS,
            bii+1, len(dset),
            10**2 * loss.item()
        ))
    sys.stdout.write('\n')

    eval_mse.append(evf())
    sys.stdout.flush()
    sch.step()

# plt.figure(figsize=(14, 3))
# plt.plot(eval_mse)
# plt.show(); plt.close()

from utils import *
viewset = SpotHistory(SROUTE, 'test', 16, lag=None, res=10, shuffle=False, verbose=False)
def xfmt(datain):
    bynode = torch.split(datain.to(device).float().unsqueeze(1), 1, 2)
    return bynode

sqerr = eval_rnn(viewset, model, plot=False, xfmt=xfmt)
print('Eval segments:', len(viewset))
print('Eval MSE: %.4f' % np.mean(sqerr))

torch.save(
    model.state_dict(),
    'checkpoints/mpnn_%s.pth' % sys.argv[1].replace('.json', ''))

# plt.figure(figsize=(14, 3))
# plt.plot(sorted(sqerr)); plt.ylim(0, 5)
# plt.show(); plt.close()

# _ = eval_rnn(viewset[:1], model, plot=True, xfmt=xfmt)
