
# In[2]:


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
np.random.seed(0)


# In[3]:


EPS = 30
LAG = 12
SROUTE = sys.argv[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
SIND = int(sys.argv[2])


# In[4]:


dset = SingleStop(SROUTE, SIND, 'train', 32, lag=LAG).generator()
evalset = SingleStop(SROUTE, SIND, 'test', 32, lag=LAG).generator()


# In[5]:


from models.Linear import Linear


# In[6]:


model = Linear(lag=5).to(device)
model.device = device


# In[7]:


criterion, opt, sch = model.params(lr=0.001)


# In[8]:


evaluate(evalset, model, crit=lambda _y, y: mape(tonpy(_y)[:, 0], tonpy(y)[:, 0]))
evaluate(evalset, model, crit=lambda _y, y: criterion(_y[:, 0], y[:, 0]).item())


# In[9]:


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


# In[33]:


name = 'n10-%s-%d'% (SROUTE, SIND)
torch.save(model.state_dict(), 'checkpoints/linear/%s.pth' % name)
print(name)


# In[32]:


kcast = 5
model.eval()
klosses = [list() for _ in range(kcast)]
for batch in evalset:
    Xs, Ys = model.format_batch(batch[:, :6, :])
    yinit = model(Xs)
    ycasts = [yinit.detach()] # k=1
    Xs = torch.cat([Xs, yinit], dim=1)
    xnext = Xs[:, -5:]
    for ki in range(1, kcast):
        ynext = model(xnext)
        ycasts.append(ynext)
        Xs = torch.cat([Xs, ynext], dim=1)
        xnext = Xs[:, -5:]
        
    Xs, _ = model.format_batch(batch[:, 6:])
    ypred, ytrue = torch.stack(ycasts, dim=1).squeeze(), Xs
    for ki in range(kcast):
        kmse = criterion(ypred[:, ki], ytrue[:, ki]).item()
        kmape = mape(tonpy(ypred[:, ki]), tonpy(ytrue[:, ki]))
        klosses[ki].append([kmse, kmape])
        
#     plt.figure(figsize=(14, 3))
#     plt.plot(tonpy(batch[0, :, 0]))
#     plt.plot(range(5, 10), tonpy(Xs[0, 5:]))
#     plt.show(); plt.close()
#     break
    
fcast = []
for ki in range(kcast):
    kmse, kmape = zip(*klosses[ki])
    kmse = np.mean(kmse)
    kmape = np.mean(kmape)
    fcast.append(dict(mse=kmse, mape=kmape))
    
print(kmse, kmape)
with open('checkpoints/linear/%s.json' % name, 'w') as fl:
    json.dump(dict(
        mse=eval_mse,
        mape=eval_mape,
        fcast=fcast,
    ), fl, indent=4)

