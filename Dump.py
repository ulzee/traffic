
# coding: utf-8

# In[1]:


# In[2]:


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
from time import time, strptime, mktime
torch.manual_seed(0)
np.random.seed(0)


# In[5]:


from glob import glob
from utils import *

# st, ed = '403259', '401851'
st, ed = sys.argv[1].split('-')
#401851', '401852'

# 400860-400861
# 400861-403781
# 400861-400948
# 400861-405376

segstr = '\n'.join([st, ed])
matches = []
sfiles = glob('data/stopcodes_sequence/*.txt')
for sname in sfiles:
    if 'X' in sname:
        continue
    with open(sname) as fl:
        raw = fl.read()
    if segstr in raw:
        matches.append(fileName(sname).split('_')[0])
print(matches)


# In[13]:


mtafiles = sorted(glob('/home/ubuntu/datasets-aux/mta/*.txt'))
start = int(sys.argv[4])
bdir = int(sys.argv[2])
mode = int(sys.argv[3])
mtafiles = mtafiles[start:start+25]
print(len(mtafiles))
print(mtafiles[0])


# In[14]:


from utils import *

for fii, fname in tqdm(enumerate(mtafiles)):
    ftag = fname.split('.')[1]
    raw_seghist = []
    with open(fname) as fl:
        _ = fl.readline()
        line = fl.readline()
        while line:
            if 'NULL' not in line and 'IN_PROGRESS' in line:
                if any([bid in line for bid in matches]):
    #                 if any([sid in line for sid in [st, ed]]):
                    try:
                        raw_seghist.append(fmt(line))
                    except:
                        print(line)
                        assert False
            line = fl.readline()

    seghist = list(filter(lambda obj: obj['direction'] == bdir, raw_seghist))
    seghist = list(filter(lambda obj: obj['phase'], seghist))

    bybus = group(seghist, 'busid')

    useable = []
    useable_post = []
    PLOT=False
    # for ii, (bid, travel) in enumerate(list(bybus.items())[:50]):
    for ii, (bid, travel) in enumerate(bybus.items()):
        if len(travel) < 40:
            # too few data from this bus
            continue
        inseg = [(mktime(ent['time']), ent['dist']) for ent in travel if ed in ent['stop']]
        if len(inseg) == 0:
            # no rel. stops from this bus
            continue

        segs = find_segs(travel)

        tlast = None
        for si, seg in enumerate(segs):
            if len(seg) <= 1:
                continue
            if len(seg) < 40: continue
            vs = est_velocity(seg)
            # vs = strip_seg(vs, prop='vel')
            # if len(vs) < 30: continue
            if any(['vel' not in entry for entry in vs]): continue
            useable.append(lscopy(vs))

            velocity = np.array([obj['vel'] for obj in vs])
            ts = np.array([mktime(ent['time']) for ent in vs])
            if tlast is not None: ts -= (ts[0] - tlast)
            tlast = ts[-1]


            _ = np.array([obj['vel'] for obj in remove_stops(vs)])
            vs, rmd = remove_peaks(vs, maxv=60)
            if rmd > 3: # fix max 3 peaks.. otherwise unreliable bus
                continue

            msize=5
            mhalf = int(msize//2)
            mean_vs = smooth_mean(lscopy(vs), fsize=msize)

            ssize=5
            shalf = int(ssize//2)
            skewed_vs = smooth_skewed(lscopy(vs), fsize=ssize, expval=1.1)

            rsize=5
            rhalf = int(rsize//2)
            range_vs = smooth_range(lscopy(vs), fsize=rsize)

    #         velocity = np.array([obj['vel'] for obj in vs])
            useable_post.append([mean_vs, skewed_vs, range_vs])
    #     break

#         sys.stdout.write('[%d/%d]: %d/%d    \r' % (fii, len(mtafiles), ii, len(bybus)))
#     sys.stdout.flush()

    if not len(useable_post):
        continue

    names = 'msr'
    options = list(zip(*useable_post))

    tag = names[mode]
    m5, m10 = bucket_segs(options[mode], st, ed, tints=[5, 10])
    # m5, m10 = bucket_segs(svs, st, ed, tints=[5, 10])

    droot = '/home/ubuntu/datasets-aux/mta/parsed'
    import json
    with open('%s/%s05_%s-%s_%s.json' % (droot, tag, st, ed, ftag), 'w') as fl:
        json.dump(m5, fl, indent=4)
    with open('%s/%s10_%s-%s_%s.json' % (droot, tag, st, ed, ftag), 'w') as fl:
        json.dump(m10, fl, indent=4)
#     break

