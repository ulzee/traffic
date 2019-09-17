
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
from datetime import timedelta

class DayHistory(data.Dataset):
	def __init__(self,
			segments,
			mode, bsize,
			# lag=6,
			res=10,
			data_path=PARSED_PATH,
			preproc='s',
			split=0.8,

			# smooth=1.2,
			ignore_missing=True,

			# post=None,
			clip_hours=8,
			norm=(12, 10), # raw mean, scale
			shuffle=False,
			verbose=True,
		):

		self.segments = segments
		self.mode = mode
		self.bsize = bsize
		self.shuffle = shuffle
		# self.lag = lag
		self.norm = norm
		self.res = res
		self.clip_hours = clip_hours
		# self.post = post

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
		vfill = None
		self.rawdata = []
		self.trange = []
		self.calendar = [] # contains time encoding based on week cycle and day (24hr) cycle
		self.nancount = [[0,0] for _ in segments]
		for ii, (day, gathered, vlists) in enumerate(all_avail):
			t0 = s2d(vlists[0][0]['time'])
			tf = s2d(vlists[0][-1]['time'])

			for segvs in vlists:
				if s2d(segvs[0]['time']) > t0:
					t0 = s2d(segvs[0]['time'])
				if s2d(segvs[-1]['time']) < tf:
					tf = s2d(segvs[-1]['time'])

			if t0 > tf:
				# skip days with no parallel times
				continue

			dt = tf - t0
			tsteps = dt.seconds // (60 * res) + 1
			vmat = np.zeros((tsteps, len(vlists)))
			# print(t0, tf)

			for si, segvs in enumerate(vlists):
				# seek until t0 begins
				ind = 0
				while s2d(segvs[ind]['time']) < t0:
					ind += 1

				if vfill is not None:
					vs = np.array(vfill(segvs, res))
				else:
					vs = np.array(nanfill(segvs, res))
				# print(len(vs), ind, tsteps)

				# if smooth is not None:
				# 	vs = blur1d(vs, sigma=smooth)
				nmean, nscale = norm
				vs = (vs - nmean) / nscale
				vmat[:, si] = vs[ind:ind+tsteps]

			if self.clip_hours is not None:
				vmat = vmat[self.clip_hours*6:]
				t0 += timedelta(seconds=self.clip_hours*60*60)
			self.trange.append((t0, tf))
			self.rawdata.append(vmat)
			for si in range(vmat.shape[1]):
				segvs = vmat[:, si]
				self.nancount[si][0] += np.isnan(segvs).sum()
				self.nancount[si][1] += len(segvs)

			midnight = t0.replace(hour=0, minute=0, second=0, microsecond=0)
			seconds_since = (t0 - midnight).total_seconds()
			times = []
			for step_i in range(vmat.shape[0]):
			    seconds_enc = (seconds_since + step_i * 60 * 10) / (24 * 60 * 60)
			    time_encoding = t0.weekday() / 6 + seconds_enc * 0.1
			    times.append(time_encoding)
			self.calendar.append(np.array(times))
		self.data = self.rawdata

		tsplit = int(len(self.data) * split)
		self.data = self.data[:tsplit] if mode == 'train' else self.data[tsplit:]
		self.trange = self.trange[:tsplit] if mode == 'train' else self.trange[tsplit:]
		self.calendar = self.calendar[:tsplit] if mode == 'train' else self.calendar[tsplit:]

		if shuffle:
			npshuff(self.data)

		if verbose:
			avglen = lambda series: np.mean([len(seq) for seq in series])
			print('Full history')
			print(' [*] Segments: %d co-avail' % len(all_avail))
			for si, (segname, ls) in enumerate(zip(segments, byseg)):
				nanperc = self.nancount[si][0]/self.nancount[si][1] * 100
				print('    * [%s]: %d (%.1f%% nan)' % (
					segname,
					len(ls),
					nanperc))
			print(' [*] Examples (%s): %d' % (mode, len(self.data)))
			# if lag is not None:
			# 	print(' [*] No missing: %d/%d' % (nComplete,nTotal))
			tsteps = sorted(list(byday.keys()))
			print(' [*] Time range: %s ~ %s' % (tsteps[0], tsteps[-1]))

	# def chunks(self, datamat):
	# 	nComplete = 0
	# 	nTotal = 0
	# 	ldata = datamat
	# 	stride = 1
	# 	ls = []
	# 	nans = []
	# 	for series in ldata:
	# 		for ti in range(self.lag, len(series), stride):
	# 			seg = series[ti-self.lag:ti]
	# 			nans.append(np.count_nonzero(np.isnan(seg)))
	# 			if np.count_nonzero(np.isnan(seg)) == 0:
	# 				ls.append(seg)
	# 				nComplete +=1
	# 			nTotal += 1
	# 	return ls, nComplete, nTotal


	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		return self.data[index]

	def generator(self, shuffle=True):
		return data.DataLoader(self,
			batch_size=self.bsize,
			shuffle=shuffle,
			num_workers=2)
