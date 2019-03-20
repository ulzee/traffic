
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
from time import time, strptime, mktime

def by(line):
#     if root in line:
	try:
		fobj = fmt(line)
		bus_id = fobj['routeid'].split('_')[1]
		if 'X' not in bus_id:
			return fobj
	except:
		print('WARN: Skipping line with missing columns...')
		print('>', line)
		# assert False
	return None

def next_stops(travel, adj={}):
	prev = travel[0]
	for _, stop in enumerate(travel[1:]):
		if stop['stop'] != prev['stop']:
			if np.abs(prev['dist'] - stop['dist']) > 2.5 * 1000:
				# sensor reset? or loop-around... either way unreliable
				# should be okay to ignore when inferring stops from lots of data
				pass
			elif mktime(stop['time']) - mktime(prev['time']) > 45 * 60:
				# skip if >1hr difference
				pass
			else:
				if prev['stop'] == 'MTA_903147' and stop['stop'] == 'MTA_404008':
					print('>')
					print(stop)
					print('=')
					print(prev)

				if prev['stop'] not in adj: adj[prev['stop']] = {}
				if stop['stop'] not in adj[prev['stop']]: adj[prev['stop']][stop['stop']] = 0
				adj[prev['stop']][stop['stop']] += 1
			prev = stop

mtafiles = sorted(glob('/home/ubuntu/datasets-aux/mta/*.txt'))
start = int(sys.argv[1])
end = int(sys.argv[2])
mtafiles = mtafiles[start:end]

for mi, sample_file in enumerate(mtafiles):
	print('[%d/%d] %s' % (mi+1, len(mtafiles), sample_file))

	raw_seghist = collect(by, sample_file)
	byroute = group(raw_seghist, 'routeid')

	route_adjs = {}
	for rid, route in tqdm(byroute.items()):

		bydir = group(route, 'direction')
		for dir, direction in bydir.items():
			adj = {}
			bytravel = group(direction, 'busid', kf=lambda key: key.replace('+', ''))
			for bid, travel in bytravel.items():
				next_stops(travel, adj)
	#         print(dir, rid, len(bytravel.keys()), len(adj))
			route_adjs['%d_%s' % (dir, rid)] = adj

	sname = 'data/next_stops/%s.json' % sample_file.split('.')[1]

	with open(sname, 'w') as fl:
		json.dump(route_adjs, fl, indent=4)