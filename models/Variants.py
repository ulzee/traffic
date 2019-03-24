
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.MPRNN import *

class MPRNN_ITER(MPRNN):
	'''
	Iteratively applies I-iterations of message passing and updating before inferring values.

	Iterations affects the range of information propagation from an observed state.

	TODO
	* Remove unnecessary dense layers on value input
	* Reduce MPN dense layer size
	'''

	def __init__(self,
		nodes, adj,

		iters=1,
		iter_indep=True, # defines an independent operator corresp. to iteration level

		hidden_size=256,
		rnnmdl=RNN_MIN,
		mpnmdl=MP_THIN,
		verbose=False):
		super(MPRNN_ITER, self).__init__(nodes, adj, hidden_size, rnnmdl, mpnmdl, verbose)

		self.iters = iters
		self.iter_indep = iter_indep

		if iter_indep:
			newmap = {}
			for it in range(iters):
				newmap[it] = {}

			for nname, ind in self.mpn_ind.items():
				newmap[0][nname] = ind
				for it in range(1, iters):
					newmap[it][nname] = len(self.mpns_list)
					self.mpns_list.append(mpnmdl(hsize=hidden_size))
			self.mpns = nn.ModuleList(self.mpns_list)
			self.mpn_ind = newmap

	def eval_message(self, it, hevals):
		msgs = []
		for ni, (hval, nname) in enumerate(zip(hevals, self.nodes)):
			# only defined over nodes w/ adj
			if nname not in self.mpn_ind:
				msgs.append(None)
				continue

			if self.iter_indep:
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

	def forward(self, series, hidden=None, dump=False):
		# print(len(series), len(series[0]), series[0][0].size())
		assert len(self.rnns) == len(series)

		# lstm params
		if hidden is None:
			hidden = [None] * len(series)

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
				self.eval_update(hevals, msgs)

			# read out values from hidden
			values_t = self.eval_readout(hevals)

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
		sch = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
		return criterion, opt, sch

# class MPRNN_SNG(MPRNN):
# 	'''
# 	Defines a single LSTM update which is aware of the hidden state at all observed stops
# 	(instead of individual updates)
# 	'''

# 	def __init__(self,
# 		nodes,
# 		adj,
# 		hidden_size=256,
# 		rnnmdl=RNN_MIN,
# 		mpnmdl=MP_THIN,
# 		verbose=False):

# 		super(MPRNN_SNG, self).__init__(
# 			nodes, adj, hidden_size, rnnmdl, mpnmdl, verbose)