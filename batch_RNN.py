
from glob import glob
from configs import *
from tqdm import tqdm
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from dataset import *
from time import time
tqdm.monitor_interval = 0
import torch
import json
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

SROUTE = sys.argv[1]
SIND = int(sys.argv[2])

LAG = 12
EPS = 30
DEEP = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

dset = SingleStop(SROUTE, SIND, 'train', 32, lag=LAG).generator()
evalset = SingleStop(SROUTE, SIND, 'test', 32, lag=LAG).generator()

yavg = SingleStop(SROUTE, SIND, 'train', 32, lag=LAG).yavg()
print('Y-avg:', yavg)

from models.temporal.RNN import RNN

model = RNN(hidden_size=256, deep=DEEP, lag=LAG).to(device)
model.device = device

criterion, opt, sch = model.params(lr=0.001)

evaluate(evalset, model, crit=lambda _y, y: wape(tonpy(_y), tonpy(y), yavg))
evaluate(evalset, model, crit=lambda _y, y: criterion(_y, y).item())

losses = []
eval_mse = []
eval_mape = []
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
            loss.item()
        ))
    sys.stdout.write('\n')

    eval_mse.append(
        evaluate(evalset, model, crit=lambda _y, y: criterion(_y[:, 4], y[:, 4]).item(), result=True))
    eval_mape.append(
        evaluate(evalset, model,
                 crit=lambda _y, y: wape(tonpy(_y)[:, 4], tonpy(y)[:, 4], yavg), result=True))
    sys.stdout.flush()
    #sch.step()

name = 'n10-%s-%d'% (SROUTE, SIND)
torch.save(model.state_dict(), 'checkpoints/rnn/%s.pth' % name)

with open('checkpoints/rnn/%s.json' % name, 'w') as fl:
    json.dump(dict(
        mse=eval_mse[-1],
        mape=eval_mape[-1],
    ), fl, indent=4)
