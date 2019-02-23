
from tqdm import tqdm
from glob import glob
import os, sys
import numpy as np
from configs import *
from scipy.ndimage import gaussian_filter as blur

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

def history_byname(name, root='history', ext='npy'):
	return np.load('data/%s/%s.%s' % (root, name, ext))

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