
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

def evaluate(dset, model, crit, result=False, norm=10):
	model.eval()
	eval_losses = []
	for bii, batch in enumerate(dset):
		Xs, Ys = model.format_batch(batch)
		outputs = model(Xs)

		loss = crit(outputs, Ys)
		loss *= norm**2
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

def lagdiff(many, lag=1):
    diff = np.zeros((many.shape[0]-1, many.shape[1]))
    for ti in range(many.shape[1]):
        series = many[:, ti]
        for ii in range(many.shape[0]-1):
            diff[ii, ti] = series[ii+1] - series[ii]
    return diff

def show_eval(viewset, model, fmax=10, meval=None, test_lag=5, target=0, norm=10, diff=None):
	import matplotlib.pyplot as plt
	# Xs, Ys = model.format_batch(viewset)

	for data in viewset:
		data = torch.Tensor(data).unsqueeze(0)

		hist = tonpy(data.squeeze(0))
		# print(hist.shape)
		plt.figure(figsize=(14, 5))
		for hi in range(hist.shape[1]):
			plt.plot(norm * hist[:, hi], color='#EEEEEE')
		plt.plot(norm * hist[:, target], color='C0')

		xoffset = range(test_lag+1, data.size()[1])
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
		plt.plot(x_pos, norm * y_run, color='C1')

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
			plt.plot(
				range(f0, f0+fmax),
				norm * tonpy(y_cast[:, :, target].squeeze()), color='C2')

		plt.legend(['measured', 'running', 'forecast'])

		plt.show(); plt.close()

def show_eval_rnn(viewset, model, fmax=10, test_lag=5, target=0):
	import matplotlib.pyplot as plt
	# Xs, Ys = model.format_batch(viewset)

	for data in viewset:
		data = torch.Tensor(data).unsqueeze(0)

		hist = tonpy(data.squeeze(0))
		plt.figure(figsize=(14, 5))
		plt.plot(hist[:, target])

		# chunk = 12
		# # chunked eval
		# xoffset = range(chunk, data.size()[1], chunk)
		# # running eval
		# y_chunks = []
		# x_chunks = []
		# hidden = None
		# for ti in xoffset:
		# 	din = data[:, ti-chunk:ti, :model.steps]
		# 	Xs, _ = model.format_batch(din)

		# 	yhat = model(Xs)

		# 	y_chunks.append(tonpy(yhat[0, :, target]))
		# 	# y_chunks.append(tonpy(Ys[0, :, target]))
		# 	x_chunks.append(list(range(ti-chunk + 1, ti)))

		# for xc, yc in zip(x_chunks, y_chunks):
		# 	plt.plot(xc, yc, color='C1')

		xoffset = range(data.size()[1])
		# running eval
		y_run = []
		hidden = None
		for ti in xoffset:
			din = data[:, ti, :model.steps]
			Xs = [din.to(model.device).float()]

			yhat, hidden = model(Xs, hidden=hidden, dump=True)

			y_run.append(tonpy(yhat[:, -1, target]))
		y_run = np.array(y_run)
		plt.plot(range(1, data.size()[1] + 1), y_run, color='red')

		# FIXME: RNN forecast
		# # running fcast
		# lamount = test_lag+1
		# for f0 in range(lamount, data.size()[1], fmax):
		# 	dwindow = data[:, f0-lamount:f0, :model.stops]
		# 	y_cast = list(torch.split(dwindow.to(model.device), 1, 1))
		# 	for fi in range(fmax):
		# 		din = torch.cat(y_cast[-lamount:], dim=1)
		# 		Xs, Ys = model.format_batch(din)
		# 		yhat = model(Xs).unsqueeze(1)
		# 		y_cast.append(yhat)
		# 	y_cast = torch.cat(y_cast[lamount:], dim=1)
		# 	plt.plot(
		# 		range(f0, f0+fmax),
		# 		tonpy(y_cast[:, :, target].squeeze()), color='C2')

		# plt.legend(['measured', 'running', 'forecast'])
		plt.legend(['measured', 'running'])

		plt.show(); plt.close()
