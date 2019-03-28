
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.MPRNN import *
from time import time
from utils import *

class MPRNN_ITER(MPRNN):
	'''
	Iteratively applies I-iterations of message passing and updating before inferring values.

	Iterations affects the range of information propagation from an observed state.

	TODO
	* Remove unnecessary dense layers on value input
	* Reduce MPN dense layer size
	'''

	name = 'mpiter'
	def __init__(self,
		nodes, adj,

		iters=1,
		iter_indep=True, # defines an independent operator corresp. to iteration level

		hidden_size=256,
		rnnmdl=RNN_MIN,
		mpnmdl=MP_THIN,
		single_mpn=False,
		verbose=False):
		super().__init__(nodes, adj, hidden_size, rnnmdl, mpnmdl, single_mpn, verbose)

		self.iters = iters
		self.iter_indep = iter_indep

		if iter_indep:
			newmap = {}
			for it in range(iters):
				newmap[it] = {}

			if single_mpn:
				for nname in nodes:
					ind = self.mpn_ind[nname]
					newmap[0][nname] = ind
					for it in range(1, iters):
						# each iteration level shares one global messenger
						newmap[it][nname] = it
				for it in range(1, iters):
					self.mpns_list.append(mpnmdl(hsize=hidden_size))
				self.mpns = nn.ModuleList(self.mpns_list)
			else:
				# print(self.mpn_ind)
				for nname in nodes:
					if not len(adj[nname]):
						continue
					ind = self.mpn_ind[nname]
					newmap[0][nname] = ind
					for it in range(1, iters):
						newmap[it][nname] = len(self.mpns_list)
						self.mpns_list.append(mpnmdl(hsize=hidden_size))
				self.mpns = nn.ModuleList(self.mpns_list)
			self.mpn_ind = newmap
		else:
			pass
			# newmap = {}
			# for it in range(iters):
			# 	newmap[it] = { k: v for k, v in self.mpn_ind.items() }

			# # TODO: single_mpn
			# self.mpn_ind = newmap

		if verbose:
			print('MPRNN_ITER')
			print('iters:', iters)
			print('indep:', iter_indep)

	def eval_message(self, it, hevals):
		msgs = []
		for ni, (hval, nname) in enumerate(zip(hevals, self.nodes)):
			# only defined over nodes w/ adj
			if not len(self.adj[nname]):
				self.stats['noadj'][nname] = True
				msgs.append(None)
				continue

			if self.iter_indep:
				# NOTE: indexed by iteration
				mpn = self.mpns[self.mpn_ind[it][nname]]
			else:
				mpn = self.mpns[self.mpn_ind[nname]]
			many = []
			ninds = [self.nodes.index(neighb) for neighb in self.adj[nname]]
			assert len(ninds)
			for neighbor_i in ninds:
				many.append(mpn.msg(hval, hevals[neighbor_i]))
			many = torch.stack(many, -1)
			msg = torch.sum(many, -1)
			msgs.append(msg)

			# self.msgcount[ni] += 1
		return msgs

	def eval_update(self, it, hevals, msgs):
		for ni, (hval, msg, nname) in enumerate(zip(hevals, msgs, self.nodes)):
			# only defined over nodes w/ adj
			if msg is None: continue

			if self.iter_indep:
				# NOTE: indexed by iteration
				mpn = self.mpns[self.mpn_ind[it][nname]]
			else:
				mpn = self.mpns[self.mpn_ind[nname]]

			uval = mpn.upd(hval, msg)
			if it == self.iters-1:
				uval = mpn.lossy(uval)

			# replaces hvalues before update
			hevals[ni] = uval

	def forward(self, series, hidden=None, dump=False):
		# print(len(series), len(series[0]), series[0][0].size())
		assert len(self.rnns) == len(series)

		# lstm params
		if hidden is None:
			# hidden = [None] * len(series)
			bsize = series[0][0].size()[0]
			hshape = (1, bsize, self.hidden_size)
			hidden = [(
					torch.rand(*hshape).to(self.device),
					torch.rand(*hshape).to(self.device)
				) for _ in range(len(series))]

		# defined over input timesteps
		tsteps = len(series[0])
		outs_bynode = [list() for _ in series]
		for ti in range(tsteps):

			# eval up to latent layer for each node
			hevals = self.eval_hidden(ti, series, hidden)

			for it in range(self.iters):
				# message passing
				msgs = self.eval_message(it, hevals)

				# updating hidden
				self.eval_update(it, hevals, msgs)

			# read out values from hidden
			values_t = self.eval_readout(hevals, hidden)

			for node_series, value in zip(outs_bynode, values_t):
				node_series.append(value)

		# print(len(outs_bynode), len(outs_bynode[0]), outs_bynode[0][0].size())
		out = list(map(lambda tens: torch.cat(tens, dim=1), outs_bynode))
		out = torch.stack(out, dim=-1)

		if dump:
			return out, hidden
		else:
			return out

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.Adam(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
		return criterion, opt, sch

class MP_ENC(MP_THIN):
	def __init__(self, hsize):
		super(MP_ENC, self).__init__(hsize)

		isize = hsize//2
		self.msg_op = nn.Sequential(
			nn.Linear(hsize*2, isize),
			# nn.ReLU(),
			# nn.Linear(isize, isize),
			# nn.ReLU(),
			# nn.Linear(isize, isize),
		)
		self.upd_op = nn.Sequential(
			nn.Linear(hsize + isize, isize),
			nn.ReLU(),
			# nn.Dropout(0.2),
			# nn.Linear(isize, isize),
			# nn.ReLU(),
			nn.Linear(isize, hsize),
		)


class MP_DEEP(MP_THIN):
	def __init__(self, hsize):
		super(MP_DEEP, self).__init__(hsize)

		self.msg_op = nn.Sequential(
			nn.Linear(hsize*2, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
		)
		self.upd_op = nn.Sequential(
			nn.Linear(hsize*2, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
		)

class RNN_HDN(RNN_MIN):
	def __init__(self, hidden_size=256, steps=10):
		super().__init__(hidden_size, steps)

		hsize = hidden_size
		self.inp = nn.Sequential(
			nn.Linear(self.insize + hsize, hsize),
			# nn.ReLU(),
			# nn.Linear(hsize, hsize),
			# nn.ReLU(), # relu input to LSTM
		)
		self.out = nn.Sequential(
			# nn.Linear(hsize, hsize),
			# nn.ReLU(),
			nn.Linear(hsize, self.outsize),
		)

class MPRNN_FCAST(MPRNN_ITER):
	'''
	Only observes selected nodes and propagates information to the rest
	'''

	name = 'mpfcast'
	def __init__(self,
		nodes, adj,

		iters=1,
		iter_indep=True, # defines an independent operator corresp. to iteration level

		hidden_size=256,
		rnnmdl=RNN_HDN,
		mpnmdl=MP_DENSE,
		single_mpn=False,
		verbose=False):

		fringes = find_fringes(nodes, adj, twoway=True)
		nodes, adj = complete_graph(nodes, adj)
		# nodes, adj = causal_graph(nodes, adj)
		# nodes, adj = reverse_graph(nodes, adj)
		super().__init__(
			nodes, adj,
			iters,
			iter_indep,
			hidden_size,
			rnnmdl, mpnmdl,
			single_mpn, verbose)

		self.fringes = fringes

		if verbose:
			print('FCAST')
			print(' [*] Fringes:', fringes)
			print(' [*] RNN:', rnnmdl)
			print(' [*] MPN:', mpnmdl)

	def eval_hidden(self, ti, nodes, hidden):
		hevals = []
		for ni, (node_series, rnn, hdn) in enumerate(zip(nodes, self.rnns, hidden)):
			if ni not in self.fringes:
				# For others, the hidden layer persists
				hevals.append(hdn[0].squeeze(0))
				continue

			# True obs. at fringes are read through a FC layer
			value_t = node_series[ti]
			hin = torch.cat([hdn[0].squeeze(0), value_t], dim=-1)
			hout = rnn.inp(hin)
			hevals.append(hout)

		assert len(hevals) == len(nodes)
		return hevals

	def eval_readout(self, hevals, hidden):
		values_t = []
		for ni, (hval, rnn, hdn) in enumerate(zip(hevals, self.rnns, hidden)):

			hin = hval.unsqueeze(0)
			hout, hdn = rnn.rnn(hin, hdn)
			hout = hout.squeeze(0)

			hidden[ni] = hdn

			vout = rnn.out(hval)
			values_t.append(vout)

		assert len(hevals) == len(values_t)
		return values_t

class MPRNN_COMPL(MPRNN_FCAST):
	'''
	Observes all nodes during training
	'''

	name = 'mpcompl'
	def eval_hidden(self, ti, nodes, hidden):
		hevals = []
		for ni, (node_series, rnn, hdn) in enumerate(zip(nodes, self.rnns, hidden)):
			# if ni not in self.fringes:
			# 	# For others, the hidden layer persists
			# 	hevals.append(hdn[0].squeeze(0))
			# 	continue

			# True obs. at fringes are read through a FC layer
			value_t = node_series[ti]
			hin = torch.cat([hdn[0].squeeze(0), value_t], dim=-1)
			hout = rnn.inp(hin)
			hevals.append(hout)

		assert len(hevals) == len(nodes)
		return hevals