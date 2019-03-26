
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
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

# graph file
SROUTE, ADJ = read_graph(sys.argv[1], verbose=False, named_adj=True)
# graph = show_graph(SROUTE, ADJ)
# render_graph(fileName(sys.argv[1]), SROUTE, ADJ)

TAG = 'rnn'
save_path = '%s/%s/%s.pth' % (CKPT_STORAGE, TAG, fileName(sys.argv[1]))
print('Saving to:')
print(save_path)

EPS = 40
LAG = 24 + 1
HSIZE = 256
STOPS = len(SROUTE)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dset = SpotHistory(SROUTE, 'train', 32, lag=LAG, res=10).generator()
evalset = SpotHistory(SROUTE, 'test', 32, lag=LAG, res=10).generator()

from models.temporal.RNN import *

model = RNN(hidden_size=HSIZE, steps=STOPS).to(device)
model.device = device
criterion, opt, sch = model.params(lr=0.001)
evf = lambda: evaluate(
	evalset, model,
	crit=lambda _y, y: criterion(_y[:, :, 0], y[:, :, 0]).item())


print('Pre-evaluate:')
_ = evf()

print('Train:')
train_mse = []
eval_mse = []
eval_mape = []
for eii  in range(EPS):
	bls = []
	for bii, batch in enumerate(dset):
		model.train()
		Xs, Ys = model.format_batch(batch)

		outputs = model(Xs)

		opt.zero_grad()
		loss = criterion(outputs, Ys)
		loss.backward()
		opt.step()

		bls.append(loss.item())
		bmse = ''
		if bii == len(dset) - 1:
			bmse = (10 ** 2 * np.mean(bls))
			train_mse.append(bmse)
			bmse = '(avg %.2f)' % bmse
		sys.stdout.write('[%d/%d : %d/%d] - L%.2f %s  \r' % (
			eii+1, EPS,
			bii+1, len(dset),
			10**2 * loss.item(),
			bmse
		))
	sys.stdout.write('\n')

	eval_mse.append(evf())
	sys.stdout.flush()
	sch.step()

viewset = SpotHistory(SROUTE, 'test', 18, lag=None, res=10, shuffle=False, verbose=False)

sqerr = eval_rnn(viewset, model, plot=False)
print('Eval segments:', len(viewset))
print('Eval MSE: %.4f' % np.mean(sqerr))

torch.save(model.state_dict(), save_path)
with open('%s/%s/%s_log.json' % (LOG_PATH, TAG, fileName(sys.argv[1])), 'w') as fl:
	json.dump([
		eval_mse,
		train_mse,
		np.mean(sqerr),
	], fl, indent=4)
