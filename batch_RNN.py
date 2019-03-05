
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


LAG = 12
EPS = 30
SIND = int(sys.argv[2])
HSIZE = 256
DEEP = False
SROUTE = sys.argv[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


dset = SingleStop(SROUTE, SIND, 'train', 32, lag=LAG).generator()
evalset = SingleStop(SROUTE, SIND, 'test', 32, lag=LAG).generator(shuffle=False)


from models.temporal.RNN import RNN


model = RNN(hidden_size=HSIZE, deep=DEEP, lag=LAG, steps=1).to(device)
model.device = device


criterion, opt, sch = model.params(lr=0.001)


evaluate(evalset, model, crit=lambda y, _y: mape(tonpy(y)[:, 0], tonpy(_y)[:, 0]))
evaluate(evalset, model, crit=lambda y, _y: criterion(y[:, 0], _y[:, 0]).item())

losses = []
eval_mse = []
eval_mape = []
from numpy.random import randint
for eii  in range(EPS):
    for bii, batch in enumerate(dset):
        model.train()
#         indend = 2 + randint(10)
        indend = 12
        Xs, Ys = model.format_batch(batch[:, :indend])
        
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
        evaluate(
            evalset, model, 
            crit=lambda y, _y: criterion(y[:, 0], _y[:, 0]).item(), result=True))
    eval_mape.append(
        evaluate(
            evalset, model, 
            crit=lambda y, _y: mape(tonpy(y)[:, 0], tonpy(_y)[:, 0]), result=True))
    sys.stdout.flush()
#     sch.step()


# In[17]:


name = 'n10-%s-%d'% (SROUTE, SIND)
torch.save(model.state_dict(), 'checkpoints/rnn/%s.pth' % name)


# In[18]:


print(name)


# In[19]:


# plt.figure(figsize=(14, 4))
# plt.subplot(1, 2, 1)
# plt.plot(losses)
# plt.subplot(1, 2, 2)
# plt.plot(eval_mse)
# plt.show();plt.close()


# In[29]:


kcast = 5
model.eval()
klosses = [list() for _ in range(kcast)]
for batch in evalset:
    Xs, Ys = model.format_batch(batch[:, :6, :])
    yinit, hidden, yall = model(Xs, dump=True)
    ycasts = [yinit.detach()] # k=1
    xnext = [yinit.detach()]
    for ki in range(1, kcast):
        ynext, hidden, _ = model(xnext, hidden=hidden, dump=True)
        ycasts.append(ynext)
        xnext = [ynext.detach()]

    Xs, _ = model.format_batch(batch[:, 6:, :])
    ypred, ytrue = torch.stack(ycasts, dim=1), torch.stack(Xs[:kcast], dim=1)
    for ki in range(kcast):
        kmse = criterion(ypred[:, ki, 0], ytrue[:, ki, 0]).item()
        kmape = mape(tonpy(ypred[:, ki, 0]), tonpy(ytrue[:, ki, 0]))
        klosses[ki].append([kmse, kmape])
#     bi = 0
#     plt.figure(figsize=(14, 3))
#     plt.plot(tonpy(batch[bi, -2, :]), color='#DDDDDD')
# #     for yent in yall:
# #         print(yent.size())
# #         plt.plot(tonpy(yent[bi]), color='#CCCCCC')
# #     for ki in range(kcast):
#     plt.plot(range(5, 10), tonpy(ypred)[bi])
#     plt.show(); plt.close()
#     assert False
fcast = []
for ki in range(kcast):
    kmse, kmape = zip(*klosses[ki])
    kmse = np.mean(kmse)
    kmape = np.mean(kmape)
    fcast.append(dict(mse=kmse, mape=kmape))
    
# print(kmse, kmape)

with open('checkpoints/rnn/%s.json' % name, 'w') as fl:
    json.dump(dict(
        mse=eval_mse,
        mape=eval_mape,
        fcast=fcast,
    ), fl, indent=4)

#     print(kmse)
#     print(kmape)
#     break
        
#     for ki 
        
#     bi = 0
#     plt.figure(figsize=(14, 3))
#     plt.plot(tonpy(batch[bi, -2, :]), color='#DDDDDD')
#     for yent in yall:
# #         print(yent.size())
#         plt.plot(tonpy(yent[bi]), color='#CCCCCC')
#     for ki in range(kcast):
#         plt.plot(ycasts[ki][bi])
#     plt.show(); plt.close()
#     assert False
        


# In[ ]:


# data = history_byname(SROUTE)

# sample = data[int(TSTEPS*0.8):]
# inds, integs = high_integ(sample[LAG:])
# inds += LAG
# # print(inds[0])
# print('Visualize period', inds[0])
# sample = sample[inds[0]-LAG:inds[0]]
# # sample = hist_smooth(sample)
# print(np.count_nonzero(np.isnan(sample)))
# for jj, ii in zip(*np.where(np.isnan(sample))):
#     sample[jj, ii] = sample[jj-1, ii]
# show_context(sample)


# In[ ]:


# bi = 30
# hist = torch.Tensor(np.expand_dims(sample[:, bi-10:bi], 0))
# Xs, _ = model.format_batch(hist)
# yout = tonpy(model(Xs))
# print(yout.shape)
# #     preds.append(tonpy(yout))


# In[ ]:


# l = show_context(sample, draw=False)
# ylast = np.flip(yout[0, :])
# xpos = list(range(bi-10, bi))
# # plt.plot([25, 25], [-5, 1], color='#666666')
# # plt.plot([25, 25], [9, 7], color='#666666')
# p = plt.plot(xpos, ylast, color='C1')
# plt.xlim(19, 31)
# plt.ylim(0, 8)
# plt.legend([l[0], p[0]], ['Known Travel Speed', 'Predicted Travel Speed'])
# plt.show(); plt.close()

