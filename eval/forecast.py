
import torch
import json, os, sys
from utils import *
import cv2

def bynode(blob):
	bynodes = list(torch.split(blob, 1, dim=2))
	bynodes = [torch.split(ent, 1, dim=1) for ent in bynodes]
	bynodes = [[vel.squeeze(1) for vel in ent] for ent in bynodes]
	return bynodes

def forecast_mprnn(
	evaldata, model, graph_file,
	runlen=48, norm=10, explicit=False, norm_mean=12, target=0, plot_targets=[0], twoway=True, verbose=False, plot=True):

	model.eval()

	vs, adjs = read_graph(graph_file, verbose=False, named_adj=True)
	fringes = find_fringes(vs, adjs, twoway=twoway) # indexes

	if verbose:
		print('Using fringes:', len(fringes))
		fringe_verts = [vs[ind] for ind in fringes]
		def vlabel(seg):
			if seg in fringe_verts:
				return '---'
			else:
				return seg[0] + seg[-2:]
		gobj = show_graph(vs, adjs, vdesc=vlabel)
		gobj.render('temp')

		import matplotlib.pyplot as plt
		img = cv2.imread('temp.jpg')
		plt.figure(figsize=(8, 8))
		plt.imshow(img)
		plt.show(); plt.close()

	data = torch.from_numpy(evaldata).unsqueeze(0).to(model.device).float()
	run_window = data[:, :runlen, :]
	runX = bynode(run_window)

	# cond. forecast
	hidden = None
	ccast = []
	for ti in range(data.size()[1]-1):
		# batch x time steps x stops
		dslice = data[:, ti:ti+2, :].clone()
		if len(ccast):
			for ni in range(len(vs)):
				if ni not in fringes:
					# self-predicted values are used for non-fringes
					if explicit:
						dslice[:, 0, ni] = -1
					else:
						dslice[:, 0, ni] = ccast[-1][0, 0, ni].item()
		Xs, _ = model.format_batch(dslice)

		yhat, hidden = model(Xs, hidden=hidden, dump=True)
		ccast.append(yhat)
	ccast = torch.cat(ccast[:], dim=1)
#     y_run, hidden = model(runX, dump=True)
#     ccast = list(torch.split(y_run, 1, 1)) # store by timestep
#     for ti in range(runlen, data.size()[1]):
#         xt = bynode(ccast[-1])
#         xnext, hidden = model(xt, hidden=hidden, dump=True)
#         for ni in range(1, data.size()[2]): # FIXME: decide which nis to observe
#             xnext[0, 0, ni] = data[0, ti, ni].item()
#         ccast.append(xnext)
#     ccast = torch.cat(ccast[runlen:], dim=1)
	__ccast = ccast
	ccast = norm * tonpy(ccast[:, :, target].squeeze()) + norm_mean
	hist = norm * tonpy(data[0, :, target]) + norm_mean
#     print(ccast.shape, hist.shape)
	sqerr = ((ccast - hist[1:])**2).tolist()

	err_bynode = {}
	all_predictions = {}
	for ni in range(len(vs)):
		# if ni not in fringes:
		ncast = norm * tonpy(__ccast[:, :, ni].squeeze()) + norm_mean
		all_predictions[vs[ni]] = ncast
		nhist = norm * tonpy(data[0, :, ni]) + norm_mean
		nerr = ((ncast - nhist[1:])**2).tolist()
		err_bynode[ni] = nerr
	if plot:
		import matplotlib.pyplot as plt
		for target in plot_targets:
			plt.figure(figsize=(14, 3))
			plt.title('[%s] RMSE %.2f' % (vs[target], np.mean(sqerr)**0.5))
		#     fcast = norm * tonpy(ycast[:, :, target].squeeze()) + norm_mean
			ccast = norm * tonpy(__ccast[:, :, target].squeeze()) + norm_mean
			hist = norm * tonpy(data[0, :, target]) + norm_mean

			# for targ in range(data.size()[2]):
			# 	if targ != target:
			# 		hist = norm * tonpy(data[0, :, targ])
			# 		plt.plot(hist, color='#CCCCCC')

			plt.plot(hist)

		#     plt.plot(
		#         range(1 + runlen, 1 + runlen + len(fcast)),
		#         fcast, color='C1')
			runlen=0
			plt.plot(
				range(1 + runlen, 1 + runlen + len(ccast)),
				ccast, color='C2')
		#     plt.ylim(-0.5, max(hist) + 1)
			plt.show(); plt.close()
	# return sqerr, err_bynode
	return sqerr, all_predictions

def forecast_rnn(evaldata, model, graph_file,
	target=0,
	plot_targets=[0],
	twoway=False,
	verbose=False, plot=True):
	# if verbose:
	# 	with open('%s/rnn/%s_log.json' % (LOG_PATH, fileName(graph_file))) as fl:
	# 		log = json.load(fl)
	# 	print(log[2])

	vs, adjs = read_graph(graph_file, verbose=False, named_adj=True)
	fringes = find_fringes(vs, adjs, twoway=twoway) # indexes
	if verbose:
		print('Using fringes:', len(fringes))
		fringe_verts = [vs[ind] for ind in fringes]
		def vlabel(seg):
			if seg in fringe_verts:
				return '---'
			else:
				return seg[0] + seg[-2:]
		gobj = show_graph(vs, adjs, vdesc=vlabel)
		gobj.render('temp')

		import matplotlib.pyplot as plt
		img = cv2.imread('temp.jpg')
		plt.figure(figsize=(8, 8))
		plt.imshow(img)
		plt.show(); plt.close()

#     init_skip = 0 # skip first part of series
	runlen = 24*2 # start evaluating after __
	norm = 10
	norm_mean = 12

	data = torch.from_numpy(evaldata).unsqueeze(0).to(model.device).float()
	run_window = data[:, :runlen, :]


	runX = list(map(lambda ent: ent.squeeze(1), list(torch.split(run_window, 1, 1))))

	hidden = None
	ccast = [data[:, 0, :].clone()]
	for ti in range(1, data.size()[1]):
		xt = ccast[-1:]
		xnext, hidden = model(xt, hidden=hidden, dump=True)
		for ni in range(1, data.size()[2]):
			if ni in fringes:
				# fringes observe the true data
				xnext[0, 0, ni] = data[0, ti, ni].item()
		ccast.append(xnext.squeeze(1))
	ccast = torch.stack(ccast[:], dim=1)

	hist = norm * tonpy(data[0, :, target])
	__ccast = ccast
	ccast = norm * tonpy(ccast[:, :, target].squeeze())
	predicted = {}
	for ni in range(len(vs)):
		predicted[vs[ni]] = norm * tonpy(__ccast[:, :, ni].squeeze()) + norm_mean
	sqerr = ((ccast[:-1] - hist[1:])**2).tolist()
	if plot:
		import matplotlib.pyplot as plt
		for target in plot_targets:
			plt.figure(figsize=(14, 3))
			plt.title('[%s] RMSE %.2f' % (vs[target], np.mean(sqerr)**0.5))
		#     fcast = norm * tonpy(ycast[:, :, target].squeeze())


			# for targ in range(data.size()[2]):
			# 	if targ != target:
			# 		hist = norm * tonpy(data[0, :, targ])
			# 		plt.plot(hist, color='#CCCCCC')

			hist = norm * tonpy(data[0, :, target])
			plt.plot(hist)

		#     plt.plot(
		#         range(1 + runlen, 1 + runlen + len(fcast)),
		#         fcast, color='C1')
			ccast = norm * tonpy(__ccast[:, :, target].squeeze())
			runlen=0
			plt.plot(
				range(1 + runlen, 1 + runlen + len(ccast)),
				ccast, color='C2')
			# plt.ylim(-0.5, max(hist) + 5)
			plt.show(); plt.close()
	return sqerr, predicted