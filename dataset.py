
from tqdm import tqdm
from glob import glob
import os, sys
import numpy as np
from configs import *
from utils import *
import json
from time import time
from numpy.random import shuffle as npshuff
from torch.utils import data
from scipy.ndimage import gaussian_filter as blur

class Routes(data.Dataset):
	def __init__(self, mode, bsize, minValid=0.7,
		lag=6,
		index_file='min-data_2h.json',
		reserved='reserved_routes.json',
		overlap=True,
		smooth=False,
		device=None):

		self.device = device
		self.bsize = bsize
		self.mode = mode
		self.smooth = smooth
		self.lag = lag

		t0 = time()
		with open(index_file) as fl:
			meta = json.load(fl)

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

	def __len__(self):
		return len(self.refs)

	def __getitem__(self, index):
		ref = self.refs[index]
		rname, ti, si = ref

		# for rname, ti, si in refs:
		mat = np.load('data/history/%s.npy' % (rname))
		hist = mat[ti-self.lag:ti, si-self.stops:si]
		try:
			assert np.count_nonzero(np.isnan(hist)) == 0
		except:
			print(hist)
			raise Exception('Nan found...')
		if self.smooth:
			hist = hist_smooth(hist)
		return hist

	def generator(self, shuffle=True):
		return data.DataLoader(self,
			batch_size=self.bsize,
			shuffle=shuffle,
			num_workers=6)

	def next(self, progress=True, maxval=40):
		raise Exception('Not imp')
		# batch = []
		# refs = self.refs[self.ind:self.ind+self.bsize]
		# for rname, ti, si in refs:
		# 	mat = np.load('data/history/%s.npy' % (rname))
		# 	hist = mat[ti-6:ti, si-10:si]

		# 	batch.append(hist)

		# if progress:
		# 	self.ind += self.bsize
		# 	if self.ind + self.bsize >= len(self.refs):
		# 		self.ind = 0
		# 		if self.mode == 'train':
		# 			npshuff(self.refs)

		# Xs = np.array(batch)
		# Xs[Xs > maxval] = maxval
		# Ys = Xs[:, -1, :] # most recent timestep is Y
		# Xs = Xs[:, :-1, :] # all previous
		# return Xs, Ys

class LocalRoute(Routes):
	def __init__(self,
		local,
		mode, bsize,
		lag=6,
		stops=5,
		local_split=0.8, # 0.2 recent will be used for testing
		meta_path=None,
		min_stride=1,
		# index_file='min-data.json',
		smooth=False,
		norm=10,
		diff=None,
		device=None, verbose=False):

		self.device = device
		self.bsize = bsize
		self.mode = mode
		self.smooth = smooth
		self.lag = lag
		self.stops = stops
		self.norm = norm
		self.diff = diff

		t0 = time()
		if meta_path is None:
			meta_path = 'metadata/%dh' % (int(lag/6))
		fpath = '%s/%s.json' % (meta_path, local)
		with open(fpath) as fl:
			meta = [json.load(fl)]
		self.meta = meta

		if verbose: print('Locals dataset: %s (%s)' % (mode, fpath))
		if verbose: print(' [*] Loaded routes:', len(meta), '(%.2fs)' % (time() - t0))
		if verbose: print(' [*] Has trainable inds:', len(meta[0]['trainable']))

		self.refs = []
		split_ind = int(13248 * local_split)
		for route in meta:
			last_t = 0
			for pair in route['trainable']:
				# TODO: split train/test here
				ti, si = pair
				if (mode == 'train' and ti < split_ind) \
					or (mode == 'test' and ti >= split_ind):

					if ti >= last_t + min_stride:
						self.refs.append([route['name']] + pair)
						last_t = ti

		assert len(meta)
		if verbose: print(' [*] Subset %s: %d' % (mode, len(self.refs)))

		if mode == 'train':
			npshuff(self.refs)
		self.ind = 0
		self.mat = np.load('data/history/%s.npy' % (local))
		self.maxval=20

	def __getitem__(self, index):
		ref = self.refs[index]
		rname, ti, si = ref
		hist = self.mat[ti-self.lag:ti, si-self.stops:si]
		assert np.count_nonzero(np.isnan(hist)) == 0
		hist[hist > self.maxval] = self.maxval
		if self.smooth:
			hist = hist_smooth(hist)

		if self.diff is not None:
			hist = lagdiff(hist, lag=self.diff)

		if self.norm is not None:
			hist = hist.copy() / self.norm

		return hist

	def __len__(self):
		return int((len(self.refs) // self.bsize) * self.bsize)

	def yavg(self):
		avg = 0
		for ref in self.refs:
			rname, ti, si = ref
			hist = self.mat[ti-1, si-self.stops:si]
			# hist = self.mat[ti-self.lag:ti, si-self.stops:si]
			assert np.count_nonzero(np.isnan(hist)) == 0
			avg += np.sum(hist) / (self.stops * len(self.refs))
		return avg

class SingleStop(LocalRoute):
	def __init__(self,
		local, stop_ind,
		mode, bsize,
		lag=6,
		stops=5,
		local_split=0.8, # 0.2 recent will be used for testing
		meta_path=None,
		min_stride=1,
		smooth=False,
		norm=10,
		diff=None,
		device=None, verbose=True):

		super().__init__(
			local,
			mode, bsize,
			lag,
			stops,
			local_split,
			meta_path,
			min_stride,
			smooth,
			norm,
			diff,
			device, verbose)

		self.refs = []
		split_ind = int(13248 * local_split)
		for route in self.meta:
			last_t = 0
			for pair in route['trainable']:
				# TODO: split train/test here
				ti, si = pair
				if si != stop_ind:
					continue
				if (mode == 'train' and ti < split_ind) \
					or (mode == 'test' and ti >= split_ind):

					if ti >= last_t + min_stride:
						self.refs.append([route['name']] + pair)
						last_t = ti
					# self.refs.append([route['name']] + pair)

		if verbose: print(' [*] Subset in Stop-%d: %d' % (stop_ind, len(self.refs)))

		if mode == 'train':
			npshuff(self.refs)
		self.ind = 0

		with open('data/stopcodes_sequence/%s.txt' % local) as fl:
			stops = fl.read().split('\n')
		segid = '%s-%s' % (stops[stop_ind], stops[stop_ind+1])
		with open('data/avgspeeds-full-ts-xclude/%s/%s.csv' % (segid[0], segid)) as fl:
			lines = fl.read().split('\n')[1:]
		lines = filter(lambda ent: ent, lines)
		lines = map(lambda ent: ent.split(',')[1], lines)
		avgspeeds = [float(ln) if ln != '' else np.nan for ln in lines]
		self.avgdata = np.array(avgspeeds)
		# self.avgdata =

from scipy.ndimage.filters import gaussian_filter1d as blur1d

class SpotHistory(data.Dataset):
	def __init__(self,
			segments,
			mode, bsize,
			lag=6,
			res=10,
			data_path=PARSED_PATH,
			preproc='s',
			split=0.8,

			smooth=1.2,
			ignore_missing=True,

			clip_hours=5,
			norm=(12, 10), # raw mean, scale
			shuffle=True,
			verbose=True,
		):

		self.segments = segments
		self.mode = mode
		self.bsize = bsize
		self.shuffle = shuffle
		self.lag = lag
		self.norm = norm
		self.res = res
		self.clip_hours = clip_hours

		byday = {}
		byseg = []
		for segname in segments:
			smatch = '%s/%s%02d_%s_*.json' % (data_path, preproc, res, segname)
			# print(smatch)
			dfiles = sorted(glob(smatch))
			try:
				assert len(dfiles)
			except:
				raise Exception('Missing: %s' % smatch)

			for dname in dfiles:
				day = dname.split('_')[-1].replace('.json', '')
				if day not in byday: byday[day] = []
				byday[day].append(dname)
			byseg.append(dfiles)

		all_avail = []
		for day in sorted(list(byday.keys())):
			gathered = byday[day]
			if len(gathered) < len(segments):
				continue
			all_avail.append([day, gathered])

		# gather the raw speeds per day
		for ii, (day, gathered) in enumerate(all_avail):
			vlists = []
			for seg_name_day in gathered:
				with open(seg_name_day) as fl:
					ls = json.load(fl)
				vlists.append(ls)
			all_avail[ii].append(vlists)

		# align the speeds
		tfill = nanfill if ignore_missing and lag is not None else constfill
		self.rawdata = []
		self.trange = []
		for ii, (day, gathered, vlists) in enumerate(all_avail):
			t0 = s2d(vlists[0][0]['time'])
			tf = s2d(vlists[0][-1]['time'])

			for segvs in vlists:
				if s2d(segvs[0]['time']) > t0:
					t0 = s2d(segvs[0]['time'])
				if s2d(segvs[-1]['time']) < tf:
					tf = s2d(segvs[-1]['time'])

			dt = tf - t0
			tsteps = dt.seconds // (60 * res) + 1
			vmat = np.zeros((tsteps, len(vlists)))

			for si, segvs in enumerate(vlists):
				# seek until t0 begins
				ind = 0
				while s2d(segvs[ind]['time']) < t0:
					ind += 1

				vs = np.array(tfill(segvs, res))
				if smooth is not None:
					vs = blur1d(vs, sigma=smooth)
				# vs /= norm
				nmean, nscale = norm
				vs = (vs - nmean) / nscale
				vmat[:, si] = vs[ind:ind+tsteps]
			self.trange.append((t0, tf))

			if self.clip_hours is not None:
				vmat = vmat[self.clip_hours*6:]
			self.rawdata.append(vmat)
		self.data = self.rawdata

		tsplit = int(len(self.data) * split)
		self.data = self.data[:tsplit] if mode == 'train' else self.data[tsplit:]

		if lag is not None:
			complete_samples = 0
			total = 0
			ldata = self.data
			stride = 1
			self.data = []
			nans = []
			for series in ldata:
				for ti in range(lag, len(series), stride):
					seg = series[ti-lag:ti]
					nans.append(np.count_nonzero(np.isnan(seg)))
					if np.count_nonzero(np.isnan(seg)) == 0:
						self.data.append(seg)
						complete_samples +=1
					total += 1
			# import matplotlib.pyplot as plt
			# plt.figure(figsize=(14, 3))
			# plt.plot(sorted(nans))
			# plt.show(); plt.close()

		if shuffle:
			npshuff(self.data)

		if verbose:
			avglen = lambda series: np.mean([len(seq) for seq in series])
			print('Full history' if lag is None else 'Chunks (lag %d)' % lag)
			# print(' [*] Files found:', len(dfiles))
			print(' [*] Segments: %d co-avail' % len(all_avail))
			for segname, ls in zip(segments, byseg):
				print('    * [%s]: %d' % (segname, len(ls)))
			print(' [*] Examples (%s): %d' % (mode, len(self.data)))
			if lag is not None:
				print(' [*] No missing: %d/%d' % (complete_samples,total))
			tsteps = sorted(list(byday.keys()))
			print(' [*] Time range: %s ~ %s' % (tsteps[0], tsteps[-1]))

			# print(' [*] %s-set size:' % mode, len(self.data))
			# print(' [*] avg sequence: %.2f' % )


	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		return self.data[index]

	def generator(self, shuffle=True):
		return data.DataLoader(self,
			batch_size=self.bsize,
			shuffle=shuffle,
			num_workers=2)

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

