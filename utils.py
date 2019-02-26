
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

def evaluate(dset, model, crit, result=False):
	model.eval()
	eval_losses = []
	for bii, batch in enumerate(dset):
		Xs, Ys = model.format_batch(batch)
		outputs = model(Xs)

		loss = crit(outputs, Ys)
		eval_losses.append(loss)
		sys.stdout.write('eval:%d/%d L%.2f    \r' % (bii+1, len(dset), loss))
	sys.stdout.flush()
	print('Eval loss: %.4f' % np.mean(eval_losses))
	if result:
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

def show_eval(viewset, model, fmax):
	import matplotlib.pyplot as plt
	# Xs, Ys = model.format_batch(viewset)

	for data in viewset:
		data = torch.Tensor(data).unsqueeze(0)

		hist = tonpy(data.squeeze())
		plt.figure(figsize=(14, 3))
		plt.plot(hist[:, 0])

		xoffset = range(model.lag+1, data.size()[1])
		# running eval
		y_run = []
		for ti in xoffset:
			din = data[:, ti-(model.lag+1):ti]
			Xs, Ys = model.format_batch(din)
			yhat = model(Xs)
			y_run.append(tonpy(yhat.squeeze()))
		y_run = np.array(y_run)
		plt.plot(xoffset, y_run)

		# running fcast
		y_cast = list(torch.split(data[:, :model.lag+1, :model.stops].to(model.device), 1, 1))
		for f0 in xoffset:
			din = torch.cat(y_cast[-model.lag-1:], dim=1)
			Xs, Ys = model.format_batch(din)
			yhat = model(Xs).unsqueeze(1)
			y_cast.append(yhat)
		y_cast = torch.cat(y_cast[model.lag+1:], dim=1)
		plt.plot(xoffset, tonpy(y_cast.squeeze()))

		plt.legend(['running', 'forecast'])

		plt.show(); plt.close()
