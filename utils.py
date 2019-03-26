
from tqdm import tqdm
from glob import glob
import os, sys
import numpy as np
from configs import *
from scipy.ndimage import gaussian_filter as blur

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

segkey = lambda s1, s2: '%s-%s' % (s1, s2)

def get_hist(sk):
	spath = '%s/%s/**/%s.csv' % (DPATH, SPEEDS, sk)
	# print(spath)
	sfile = glob(spath)
	try:
		assert len(sfile)
	except:
		# raise Exception('%s not found' % sk)
		return None

	lines = []
	with open(sfile[0]) as fl:
		line = fl.readline()
		while line:
			lines.append(line.strip('\n'))
			line = fl.readline()
	lines = lines[1:-1]
	lines = [ln.split(',') for ln in lines]
	lines = [(tstamp, None if val is '' else val) for tstamp, val in lines]
	return lines

def get_validity(sk):
	hist = get_hist(sk)
	if hist is None: return None

	missing = 0
	for tstamp, val in hist:
		if val is None:
			missing += 1
	return missing / len(hist)

fileName = lambda raw: raw.split('/')[-1].split('.')[0]

def history_byname(name, root='data/history', ext='npy'):
	return np.load('%s/%s.%s' % (root, name, ext))

def show_travels(mat, recent=3):
	import matplotlib.pyplot as plt
	plt.figure(figsize=(14, recent + 1))
	for ii in range(recent):
		plt.subplot(recent, 1, ii+1)
		plt.plot(mat[-ii, :])
	plt.show();plt.close()

tonpy = lambda tens: tens.detach().cpu().numpy()

def dedupe(segs):
	covered = {}
	unique = []
	for ti, si in segs:
#         print(ti, si)
		tkey = '%d-%d' % (ti, si)
		if tkey not in covered:
			unique.append([ti, si])
			for jj in range(5):
				for ii in range(5):
					covered['%d-%d' % (ti+jj, si+ii)] = True
	return unique

def evaluate(dset, model, crit, result=False, norm=10, buff='          '):
	from time import time

	model.eval()
	eval_losses = []
	t0 = time()
	for bii, batch in enumerate(dset):
		# print('\nBatch: %.2fs' % (time() - t0))
		Xs, Ys = model.format_batch(batch)

		t0 = time()
		with torch.no_grad():
			outputs = model(Xs)
			# print('\nExec: %.2fs' % (time() - t0))

			t0 = time()
			loss = crit(outputs, Ys)
			# print('\nCrit: %.2fs' % (time() - t0))
			loss *= norm**2
		eval_losses.append(loss)
		sys.stdout.write('eval:%d/%d L%.2f %s\r' % (bii+1, len(dset), loss, buff))
		t0 = time()
	sys.stdout.flush()
	print('Eval loss: %.4f' % np.mean(eval_losses))
	# if result:
	return np.mean(eval_losses)

def high_integ(sample):
	row_integs = []
	for row in sample:
		row_integs.append(np.count_nonzero(np.isnan(row)) / len(row))
	return np.argsort(row_integs), row_integs
	# return np.argsort(row_integs)[0]
	# return

def show_context(sample, draw=True):
	import matplotlib.pyplot as plt
	plt.figure(figsize=(14, 4))
	for hi in range(1, 6):
		plt.plot(sample[-hi-1, :], color='#DDDDDD')
	p = plt.plot(sample[-1, :], color='C0')
	if draw:
		plt.show(); plt.close()
	return p

def hist_smooth(hist):
	return np.array([blur(row, 2) for row in hist])

def wape(_tens, tens, denom):
	return 100 * np.mean(np.abs(_tens - tens) / denom)

def mape(tens, _tens):
	ls = []
	for _y, y in zip(_tens, tens):
		# print(_y, y)
		if _y == 0:
			continue
		else:
			ls.append(np.abs(_y - y) / _y)
	assert len(ls)
	return np.mean(ls) * 100

def lagdiff(many, lag=1):
	diff = np.zeros((many.shape[0]-1, many.shape[1]))
	for ti in range(many.shape[1]):
		series = many[:, ti]
		for ii in range(many.shape[0]-1):
			diff[ii, ti] = series[ii+1] - series[ii]
	return diff

def eval_lin(viewset, model, fmax=10, meval=None, test_lag=5, target=0, norm=10, diff=None, plot=True):
	import matplotlib.pyplot as plt
	# Xs, Ys = model.format_batch(viewset)

	mses = []
	for data in viewset:
		data = torch.Tensor(data).unsqueeze(0)

		hist = tonpy(data.squeeze(0)).copy()
		# print(hist.shape)
		if plot:
			plt.figure(figsize=(14, 5))
			for hi in range(hist.shape[1]):
				if hi != target:
					plt.plot(norm * hist[:, hi], color='#EEEEEE')
			plt.plot(norm * hist[:, target], color='C0')

		xoffset = range(test_lag+1, data.size()[1]+1)
		# running eval
		y_run = []
		x_pos = []
		for ti in xoffset:
			din = data[:, ti-(test_lag+1):ti]
			Xs, Ys = model.format_batch(din)
			if meval is not None:
				yhat = meval(model, Xs)
			else:
				yhat = model(Xs)
			y_run.append(tonpy(yhat[:, target]))
			# y_run.append(tonpy(Ys[:, target]))
			x_pos.append(ti-1)
		y_run = np.array(y_run)
		if plot: plt.plot(x_pos, norm * y_run, color='red')

		# running fcast
		lamount = test_lag+1
		for f0 in range(lamount, data.size()[1], fmax):
			dwindow = data[:, f0-lamount:f0, :model.stops]
			y_cast = list(torch.split(dwindow.to(model.device), 1, 1))
			for fi in range(fmax):
				din = torch.cat(y_cast[-lamount:], dim=1)
				Xs, Ys = model.format_batch(din)
				yhat = model(Xs).unsqueeze(1)
				y_cast.append(yhat)
			y_cast = torch.cat(y_cast[lamount:], dim=1)
			if plot: plt.plot(
				range(f0, f0+fmax),
				norm * tonpy(y_cast[:, :, target].squeeze()), color='C1')

		if plot:
			plt.legend(['ground truth', 'next pred.', 'forecast'])
		# if plot: plt.ylim(-3, 35)


		ytrue = hist[test_lag:, target]
		yguess = y_run[:, target]
		# print(ytrue.shape)
		# print(yguess.shape)
		assert len(ytrue) == len(yguess)
		diff = (ytrue - yguess) * norm
		diff = diff ** 2
		mses += diff.tolist()

		if plot:
			plt.title('%.4f' % np.mean(diff))
			plt.show(); plt.close()
	return mses

def eval_rnn(
	viewset, model,
	flag=5, fmax=10,
	target=0, norm=10,
	plot=True,
	xfmt=None):

	import matplotlib.pyplot as plt

	losses = []
	plots = []
	for data in viewset:
		data = torch.Tensor(data).unsqueeze(0)

		hist = tonpy(data.squeeze(0))
		if plot:
			others = []
			plt.figure(figsize=(14, 5))
			plots.append(plt.plot(hist[:, target] * norm))
			for ni in range(hist.shape[1]):
				if ni != target:
					others.append(hist[:, ni] * norm)
					plt.plot(hist[:, ni] * norm, color='#EEEEEE')
			plt.plot(np.mean(np.array(others), 0), color='#BBBBBB')


		# running eval
		y_run = []
		hidden = None
		xoffset = range(data.size()[1])
		for ti in xoffset:
			din = data[:, ti, :model.steps]
			if xfmt is not None:
				Xs = xfmt(din)
			else:
				Xs = [din.to(model.device).float()]

			yhat, hidden = model(Xs, hidden=hidden, dump=True)

			y_run.append(tonpy(yhat[:, -1, target]))
		y_run = np.array(y_run)
		if plot:
			plots.append(plt.plot(range(1, data.size()[1] + 1), y_run * norm, color='red'))

		# Running conditional forecast
		# only possible with >1 nodes
		if plot:
			if data.size()[-1] > 1:
				print('Conditional Forecast')
				ccast = []
				hidden = None
				local_data = data.clone()
				for ti in range(data.size()[1]-1):
					# batch x time steps x stops
					dslice = local_data[:, ti:ti+2, :]
					if len(ccast):
						# last known value is used for target in conditional forecasting
						dslice[:, 0, 0] = ccast[-1]
					Xs, Ys = model.format_batch(dslice)

					yhat, hidden = model(Xs, hidden=hidden, dump=True)
					ccast.append(yhat[:, :, target].squeeze().item())
					# ccast.append(Ys[:, :, target].squeeze().item())
			if plot:
				plots.append(plt.plot(
					range(1, data.size()[1]),
					np.array(ccast) * norm, color='C2'))

			# Self-predictions (forecast) fmax-ahead given flag
			fcast = []
			fcast_t = []
			hidden = None
			for ti in range(flag, data.size()[1]-fmax, flag):
				# batch x time steps x stops
				dslice = data[:, ti-flag:ti+fmax, :].clone()

				# first observe the lag
				lagX, _ = model.format_batch(dslice[:, :flag+1])
				yhat, hidden = model(lagX, hidden=hidden, dump=True)
				dslice[:, flag] = yhat[:, -1, target]

				# now running forecast (fmax-ahead)
				for fi in range(fmax-1):
					tind = flag+fi
					x_t, _ = model.format_batch(dslice[:, tind:tind+2])
					yhat, hidden = model(x_t, hidden=hidden, dump=True)
					dslice[:, tind+1] = yhat

				fcast.append(tonpy(dslice)[0, flag:, target])
				fcast_t.append(range(ti, ti+fmax))
			if plot:
				for erange, entry in zip(fcast_t, fcast):
					pass
					# plots.append(plt.plot(
					# 	erange,
					# 	np.array(entry) * norm, color='C1'))

		diff = (hist[1:, target] - y_run[:-1, target]) * norm
		diff = diff ** 2
		if plot:
			plt.legend(
				[p[0] for p in plots],
				['ground truth', 'next pred.', 'cond. fcast', 'forecast'])
			plt.title('%.4f' % np.mean(diff))
			plt.show(); plt.close()
		losses += diff.tolist()
	return losses
		# print(len(hist[:, target]), len(y_run))

from time import time, strptime, mktime

def fmt(raw):
	parts = raw.strip().split('\t')
	tobj = strptime(parts[2],'%Y-%m-%d %H:%M:%S')
	return dict(
		time=tobj,
		routeid=parts[7],
		direction=int(parts[5]),
		busid=parts[3],
		dist=float(parts[4]),
		phase=parts[6] == 'IN_PROGRESS',
		stop=parts[-1],
	)

def group(ls, field, show=False, kf=None):
	buses = {}
	for obj in ls:
		key = obj[field]
		if kf is not None: key = kf(key)
		if key not in buses:
			buses[key] = []
		buses[key].append(obj)
	if show:
		for k, v in buses.items():
			print(k, len(v))
	return buses

from datetime import datetime, timedelta
# import datetime

def est_velocity(ls, max_jump=1500, lag=1):
	assert len(ls) > lag
	vs = []
	dts = [ent['time'] for ent in ls]
	for ii in range(1, len(ls)):
		dist = ls[ii]['dist'] - ls[ii-lag]['dist']
		secs = mktime(ls[ii]['time']) - mktime(ls[ii-lag]['time'])
		assert secs > 0

		if np.abs(dist) > max_jump:
			# super big jump
			# print('JUMP', dist)
			if 'vel' in ls[ii-lag]:
				ls[ii]['vel'] = ls[ii-lag]['vel']
				vs.append(ls[ii])
			continue

		vel = (dist / secs) * MMPH

		ls[ii]['vel'] = vel

	# for ii in range(1, len(ls)):
	# 	secs = mktime(dts[ii]) - mktime(dts[ii-lag])

	# 	dt = datetime.fromtimestamp(mktime(dts[ii-lag]))
	# 	tadd = timedelta(seconds=secs/2)
	# 	dt = dt + tadd
	# 	ls[ii]['time'] = dt.timetuple()
	return ls[1:]

def find_segs(ls, maxdiff=30 * 4):
	assert len(ls) > 1
	ch = []

	prev = ls[0]
	cont = [ls[0]]
	for ii in range(1, len(ls)):
		tdiff = mktime(ls[ii]['time']) - mktime(prev['time'])
#         print(tdiff)
		if tdiff < maxdiff:
			cont.append(ls[ii])
		else:
			assert len(cont)
			ch.append(cont)
			cont = [ls[ii]]
		prev = ls[ii]
	if len(cont):
		ch.append(cont)

	assert len(ls) == sum([len(cont) for cont in ch])
	return ch

def strip_seg(ls,
	prop='dist',
	prange=[4, 7, 10], crit=np.mean,
	zerolim=0.1, skip=2):

	# remove static values on either end of a series:
	# either based on equal values or ~zeros

	for pr in prange:
		if len(ls) <= prange:
			break

		ii = skip
		peek = [obj[prop] for obj in ls[ii:ii+pr]]
		while crit(peek) < zerolim:
			peek = [obj[prop] for obj in ls[ii:ii+pr]]
			ii += 1
		ls = ls[ii:]

		ii = len(ls) - skip
		tail = [obj[prop] for obj in ls[ii-pr:ii]]
		while crit(tail) < zerolim:
			tail = [obj[prop] for obj in ls[ii-pr:ii]]
			ii -= 1
		ls = ls[:ii]

	return ls

	# peek = [obj[prop] for obj in ls[1:prange]]
	# if all([peek[0] == ent for ent in peek]):
	# 	ii = prange
	# 	while ii < len(ls) and peek[0] == ls[ii][prop]:
	# 		ii += 1
	# 	# print(peek)
	# 	# print([ent[prop] for ent in ls[:20]])
	# 	ls = ls[ii:]

	# tail = [obj[prop] for obj in ls[-prange:-1]]
	# print(np.mean(tail), tail)
	# if all([tail[0] == ent for ent in tail]):
	# 	ii = len(ls) - prange - 1
	# 	while ii >= 0 and tail[0] == ls[ii][prop]:
	# 		ii -= 1
	# 	# print([ent[prop] for ent in ls[]])
	# 	ls = ls[:ii]
	# return ls

def remove_stops(ls, minv=0.05):
	vs = np.array([obj['vel'] for obj in ls])
	for ii in range(1, len(ls)-1):
		# print(ls[ii]['vel'])
		if ls[ii]['vel'] < minv and \
			ls[ii-1]['vel'] != 0 and ls[ii-1]['vel'] != 0:
			vs[ii] = np.mean([ls[ii+1]['vel'], ls[ii-1]['vel']])

	for newvel, obj in zip(vs, ls):
		obj['vel'] = newvel
	return ls

def remove_peaks(ls, maxv=40):
	rmd = 0
	for ii in range(1, len(ls)):
		if ls[ii]['vel'] > maxv:
			# not reliable peak
			ls[ii]['vel'] = ls[ii-1]['vel']
			rmd += 1
	return ls, rmd

def srange(ls):
	ts = [0]
	for ii in range(0, len(ls)-1):
		tdiff = mktime(ls[ii+1]['time']) - mktime(ls[ii]['time'])
		assert tdiff >= 0
		ts.append(tdiff + ts[-1])
	return np.array(ts)

def seg_avg(ls):
	segavg = (ls[-1]['dist'] - ls[0]['dist']) / (mktime(ls[-1]['time']) -mktime(ls[0]['time']))
	return MMPH * segavg

def group_time(ls, tint=60*10):
	by10mins = {}
	for obj in ls:
		pos10min = int(mktime(obj['time']) // (tint))
		if pos10min not in by10mins:
			by10mins[pos10min] = []
		by10mins[pos10min].append(obj)
	return by10mins

def smooth_speed(ls, fsize=5):
	vs = np.array([obj['vel'] for obj in ls])
	off = int(fsize//2)
	for ii in range(off, len(ls) - off):
		vs[ii] = np.mean([ent['vel'] for ent in ls[ii-off:ii+off+1]])

	for ii in range(off, len(ls)-off):
		ls[ii]['vel'] = vs[ii]

	return ls[off:-off]

def smooth_mean(ls, fsize=5):
	vs = np.array([obj['vel'] for obj in ls])
	off = int(fsize//2)
	for ii in range(off, len(ls) - off):
		vs[ii] = np.mean([ent['vel'] for ent in ls[ii-off:ii+off+1]])

	for ii in range(off, len(ls)-off):
		ls[ii]['vel'] = vs[ii]

	return ls[off:-off]

def smooth_skewed(ls, fsize=5, expval=np.e):
	vs = np.array([obj['vel'] for obj in ls])
	off = int(fsize//2)
	for ii in range(off, len(ls) - off):
		window = np.array([ent['vel'] for ent in ls[ii-off:ii+off+1]])
		# if all([window[off] >= ent for ent in window]):
		# 	# just keep if itself is the max
		# 	vs[ii] = window[off]
		# else:
		wlist = expval ** (window)
		wlist /= np.sum(wlist)
		wmean = np.dot(window, wlist)
		vs[ii] = wmean

	for ii in range(off, len(ls)-off):
		ls[ii]['vel'] = vs[ii]

	return ls[off:-off]

def smooth_range(ls, fsize=5):
	vs = np.array([obj['vel'] for obj in ls])
	smoothed = vs.copy()
	off = int(fsize//2)

	# for ii in range(0, len(ls) - fsize):
	ii = 0
	while ii < len(ls) - fsize:
		window = vs[ii+1:ii+fsize]
		nextmax = np.argmax(window)
		if nextmax == 0:
			# ~monotonic decrease
			ii += 1
		else:
			# interpolate between
			smoothed[ii:ii+1+nextmax+1] = np.linspace(
				vs[ii],
				vs[ii+1+nextmax],
				nextmax+2)
			# smoothed[ii:ii+nextmax] = vs[ii]
			ii += nextmax + 1

	for obj, vel in zip(ls, smoothed):
		obj['vel'] = vel

	return ls[:-fsize]

def bucket_segs(ls, st, ed, tints=[1, 2, 5, 10]):
	inseg = []
	for seg in ls:
		for ent in seg:
			# if any(sid in ent['stop'] for sid in [st, ed]):
			if ed in ent['stop']:
				# assert ent['vel'] < 20
				inseg.append(ent)

	flat = [(datetime.fromtimestamp(mktime(ent['time'])), ent['vel']) for ent in inseg]
	flat = sorted(flat, key=lambda ent: ent[0])

	many = []
	for ti, tr in enumerate(tints):
		clipped = [(tm - timedelta(
			minutes=tm.minute % tr,
			seconds=tm.second,
			microseconds=tm.microsecond), vel) \
		for (tm, vel) in flat]
		bymins = []
		while len(clipped):
			head = clipped[0]
			bucket = [head]
			while len(clipped) and clipped[0][0] == head[0]:
				bucket.append(clipped[0])
				clipped = clipped[1:]
			bymins.append((bucket[0][0], [ent[1] for ent in bucket]))
		assert len(bymins)
		bymins = [dict(time=mktime(buck[0].timetuple()), vel=np.mean(buck[1])) for buck in bymins]
		many.append(bymins)

	return many

def seg_scatter(ls, st, ed, title='', tail=None, tints=[1, 2, 5, 10]):
	import matplotlib.pyplot as plt
	inseg = []
	for seg in ls:
		for ent in seg:
			# if any(sid in ent['stop'] for sid in [st, ed]):
			if ed in ent['stop']:
				# assert ent['vel'] < 20
				inseg.append(ent)

	plots = []
	tvs = [(mktime(ent['time']), ent['vel'], ent['busid']) for ent in inseg]
	# tvs = sorted(tvs, key=lambda ent: ent[0])[-tail:]
	# tvs = sorted(tvs, key=lambda ent: ent[0])[:tail]
	tvs = sorted(tvs, key=lambda ent: ent[0])
	trange = tvs[-1][0] - tvs[0][0]
	if tail is not None:
		tvs = tvs[-tail:]
	ts, vs, bids = zip(*tvs)

	flat = [(datetime.fromtimestamp(mktime(ent['time'])), ent['vel']) for ent in inseg]
	flat = sorted(flat, key=lambda ent: ent[0])

	bcolors = []
	bd = {}
	for bid in bids:
		if bid not in bd:
			bd[bid] = len(bd)
		bcolors.append('C%d' % (bd[bid] % 9))
		# bcolors.append('C%d' % bd[bid])

	# print(bcolors)

	plt.figure(figsize=(14, 3))

	plt.scatter(ts, vs, color=bcolors, alpha=0.15)
	plt.ylim(-2, 35)

	bavail = []
	for ti, tr in enumerate(tints):
		clipped = [(tm - timedelta(
			minutes=tm.minute % tr,
			seconds=tm.second,
			microseconds=tm.microsecond), vel) \
		for (tm, vel) in flat]
		bymins = []
		while len(clipped):
			head = clipped[0]
			bucket = [head]
			while len(clipped) and clipped[0][0] == head[0]:
				bucket.append(clipped[0])
				clipped = clipped[1:]
			bymins.append((bucket[0][0], [ent[1] for ent in bucket]))
		assert len(bymins)
		bymins = [(mktime(buck[0].timetuple()), np.mean(buck[1])) for buck in bymins]

		bts, bvs = zip(*bymins)
		pp = plt.plot(bts, bvs, color='C%d' % ti)
		plots.append(pp[0])
		avail = len(bts) / (trange / 60 / tr) * 100
		bavail.append('%.1f%%' % avail)

	try:
		with open('%s/%s/%s/%s-%s.csv' % (DPATH, SPEEDS, st[0], st, ed)) as fl:
			lines = fl.read().split('\n')[1:-1]
		lines = [ent.split(',') for ent in lines]
		olddata = [(datetime.strptime(ent[0], '%Y-%m-%d %H:%M:%S'), ent[1]) for ent in lines]
		oldInRange = [ent for ent in olddata if ent[0] >= flat[0][0] and ent[0] <= flat[-1][0]]
		oldRaw = [(mktime(ent[0].timetuple()), ent[1]) for ent in oldInRange]
		oldRaw = [(ent[0], float(ent[1])) for ent in oldRaw if ent[1] is not '']
		ots, ovs = zip(*oldRaw)
		pp = plt.plot(ots, np.array(ovs) * MMPH, color='red')
		plots.append(pp[0])
	except:
		print('Kdd data not found.')

	plt.legend(plots, ['%dm' % ival for ival in tints] + ['kdd'])
	# print(oldInRange)


	# print(olddata[0])
	# print(len(lines))

	plt.plot(ts, [5] * len(ts), color='black')
	plt.xticks(
		[mktime(flat[0][0].timetuple()), mktime(flat[-1][0].timetuple())],
		[flat[0][0], flat[-1][0]])
	plt.title('%s: %s' % (title, ','.join(bavail)))
	plt.show(); plt.close()

def lscopy(ls):
	return [dict(
		time=ent['time'],
		vel=ent['vel'],
		stop=ent['stop'],
		busid=ent['busid'],
		) for ent in ls]

def vonly(ls):
	return np.array([obj['vel'] for obj in ls])

def dhm(raw):
	dobj = datetime.fromtimestamp(mktime(raw))
	return dobj.strftime('%H:%M:%S')

def shm(raw):
	dobj = datetime.fromtimestamp(raw)
	return dobj.strftime('%H:%M:%S')

def hdiff(r1, r2):
	d1 = datetime.fromtimestamp(mktime(r1))
	d2 = datetime.fromtimestamp(mktime(r2))
	td = d2 - d1
	return '%02d:%02d' % (td.seconds//(60*60), td.seconds//60)
	# return dobj.strftime('%H:%M:%S')

s2d = lambda secs: datetime.fromtimestamp(secs)

def tfill(tvlist, res):
	t0 = datetime.fromtimestamp(tvlist[0]['time'])
	tf = datetime.fromtimestamp(tvlist[-1]['time'])
	dt = tf - t0
	tsteps = dt.seconds // (60 * res) + 1
	vs = []
	for entry in tvlist:
		te = datetime.fromtimestamp(entry['time'])
		tind = (te - t0).seconds // (60 * res)

		# fill previous if missing
		while len(vs) < tind:
			vs.append(vs[-1])

		vs.append(entry['vel'])
	assert tsteps == len(vs)

	return vs

def show_graph(vs, es, vdesc=lambda ent: ent):
	from graphviz import Digraph
	dot = Digraph(comment=vs[0], format='jpg')
	for vi, vert in enumerate(vs):
		shape='box' if vi == 0 else None
		dot.node(vert, vdesc(vert), shape=shape)
		# dot.node(vert, vert, shape=shape)
	for vi, (vert, alist) in enumerate(es.items()):
		for vto in alist:
			dot.edge(vert, vto)
	return dot

def read_graph(fname, verbose=True, named_adj=False):
	import json
	with open(fname) as fl:
		SROUTE, adjspec = json.load(fl)

	if named_adj:
		ADJ = adjspec
	else:
		ADJ = {}
		for ind, ls in adjspec:
			ADJ[SROUTE[ind]] = [SROUTE[ii] for ii in ls]

	if verbose:
		print(SROUTE)
		print(ADJ)

	return SROUTE, ADJ

def prune_graph(vs, adj, validf, minv=0.5):
    pvs = []
    padj = {}

    queue = [vs[0]]
    while len(queue):
        vert = queue[0]
        queue = queue[1:]

        pvs.append(vert)
        padj[vert] = []
        for child in adj[vert]:
            if validf(child) >= minv:
                if child not in padj[vert]: padj[vert].append(child)
                if child not in queue:
                    queue.append(child)

    return pvs, padj

def rev_graph(vs, adj):
	rvs = vs
	radj = {}

	for vert in vs:
		if vert not in radj: radj[vert] = []
		for child in adj[vert]:
			if child not in radj: radj[child] = []
			if vert not in radj[child]:
				radj[child].append(vert)

	return rvs, radj

def complete_graph(vs, adj):
	rvs = vs
	radj = {}

	for vert in vs:
		if vert not in radj: radj[vert] = []
		for child in adj[vert]:
			if child not in radj: radj[child] = []

			if vert not in radj[child]:
				radj[child].append(vert)
			if child not in radj[vert]:
				radj[vert].append(child)

	return rvs, radj

def render_graph(name, vs, adj, save=True, path='jobs/outputs'):
	with open('../shiva-traffic/Valid-Counts.txt') as fl:
		lines = fl.read().split('\n')[1:-1]
	valids = {}
	for ln in lines:
		stop, count = ln.split(',')
		valids[stop] = int(count)

	gobj = show_graph(vs, adj, vdesc=lambda vert: ('%.2f' % (valids[vert]/13248)))
	if save:
		gobj.render('%s/%s' % (path, name))
	else:
		return gobj

def find_fringes(vs, adj, twoway=False):
	# default only returns downstream fringes

	is_pointed = {}
	no_children = {}
	for vert, ls in adj.items():
		for other in ls:
			is_pointed[other] = True
		if len(ls) == 0:
			no_children[vert] = True
	fringes = []
	if twoway:
		for vert in vs:
			if vert not in is_pointed:
				fringes.append(vs.index(vert))
	for vert in no_children.keys():
		ind = vs.index(vert)
		if ind not in fringes:
			fringes.append(ind)
	assert len(fringes) <= len(vs)
	return fringes

def collect(byf, fname, verbose=True):
    res = []
    li = 0
    added = 0
    with open(fname) as fl:
        _ = fl.readline()
        line = fl.readline()
        while line:
            if 'NULL' not in line and 'IN_PROGRESS' in line:
                obj = byf(line)
                if obj is not None:
                    res.append(obj)
                    added += 1
            line = fl.readline()
            li += 1
            if li % 10000 == 0 and verbose:
                sys.stdout.write('%d       \r' % li)
        if verbose: sys.stdout.flush()
    if verbose: print('Collected %d/%d' % (added, li))
    return res