
from tqdm import tqdm
from glob import glob
import os, sys
import numpy as np
from configs import *
from utils import *
import json
from time import time
from numpy.random import shuffle as npshuff

class Routes:
	def __init__(self, mode, bsize, minValid=0.7,
		index_file='metadata.json',
		reserved='reserved_routes.json',
		overlap=True):

		self.bsize = bsize
		self.mode = mode

		t0 = time()
		with open(index_file) as fl:
			meta = json.load(fl)


		# rfiles = glob('%s/%s/*.txt' % (DPATH, ROUTES))
		# routes = []
		# for rfile in rfiles:
		# 	with open(rfile) as fl:
		# 		stops = fl.read().split('\n')
		# 	routes.append(stops)

		print('Routes dataset: %s' %mode)
		print(' [*] Loaded routes:', len(meta), '(%.2fs)' % (time() - t0))

		# Filter reserved routes
		with open(reserved) as fl:
			res_metas = json.load(fl)
		res_names = [entry['name'] for entry in res_metas]
		if mode == 'train':
			meta = [entry for entry in meta if entry['name'] not in res_names]
		else:
			meta = [entry for entry in meta if entry['name'] in res_names]
			if not overlap:
				for route in meta:
					route['trainable'] = dedupe(route['trainable'])
		assert len(meta)
		print(' [*] Subset %s: %d (%s)' % (mode, len(meta), reserved))
		self.meta = meta

		t0 = time()
		self.refs = []
		for route in meta:
			for pair in route['trainable']:
				self.refs.append([route['name']] + pair)
		print(' [*] Loaded trainable inds:', len(self.refs), '(%.2fs)' % (time() - t0))
		t0 = time()

		if mode == 'train':
			npshuff(self.refs)
		self.ind = 0

	def size(self):
		return int(len(self.refs) // self.bsize)

	def next(self, progress=True, maxval=40):
		batch = []
		refs = self.refs[self.ind:self.ind+self.bsize]
		for rname, ti, si in refs:
			mat = np.load('data/history/%s.npy' % (rname))
			hist = mat[ti-6:ti, si-10:si]

			batch.append(hist)

		if progress:
			self.ind += self.bsize
			if self.ind >= len(self.refs):
				self.ind = 0
				if self.mode == 'train':
					npshuff(self.refs)

		Xs = np.array(batch)
		Xs[Xs > maxval] = maxval
		Ys = Xs[:, -1, :] # most recent timestep is Y
		Xs = Xs[:, :-1, :] # all previous
		return Xs, Ys

if __name__ == '__main__':
	dset = Routes(bsize=32, index_file='min-data.json')

	# ts = []
	# for _ in tqdm(range(100)):
	# 	t0 = time()
	# 	batch = dset.next()
	# 	dt = time() - t0
	# 	ts.append(dt)
	# print(np.mean(ts))
	# batch = dset.next()
	# print(batch[0].shape)
	# print()

