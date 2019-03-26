
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.GRNN import *

class MP_THIN(nn.Module):
	def __init__(self, hsize):
		super(MP_THIN, self).__init__()

		self.msg_op = nn.Linear(hsize*2, hsize)
		self.upd_op = nn.Linear(hsize*2, hsize)

	def msg(self, hself, hother):
		hcat = torch.cat([hself, hother], -1)
		return self.msg_op(hcat)

	def upd(self, hself, msg):
		hcat = torch.cat([hself, msg], -1)
		return self.upd_op(hcat)

class MP_DENSE(MP_THIN):
	def __init__(self, hsize):
		super(MP_DENSE, self).__init__(hsize)

		self.msg_op = nn.Sequential(
			nn.Linear(hsize*2, hsize),
			nn.ReLU(),
			# nn.Linear(hsize, hsize),
			# nn.ReLU(),
			nn.Linear(hsize, hsize),
		)
		self.upd_op = nn.Sequential(
			nn.Linear(hsize*2, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
		)
		self.lossy = nn.Sequential(
			nn.ReLU(),
			nn.Dropout(0.5),
			# nn.Linear(hsize, hsize),
		)

class MPRNN(GRNN):
	'''
	Instantiates one RNN per location in input graph.
	Additionally, instantiates a message passing layer per node.

	MPNs are specified by the given adjacency matrix (list?).
	'''

	name = 'mprnn'
	def __init__(self,
		nodes, adj,

		hidden_size=256,

		rnnmdl=RNN_MIN,
		mpnmdl=MP_THIN,
		verbose=False):
		super(MPRNN, self).__init__(len(nodes), hidden_size, rnnmdl)

		self.adj = adj
		self.nodes = nodes

		# follows a canonical order as defined by input order of nodes
		mpns_list = []
		self.mpn_ind = {}
		for nname in nodes:
			adjnames = adj[nname]
			if len(adjnames):
				self.mpn_ind[nname] = len(mpns_list)
				mpns_list.append(mpnmdl(hsize=hidden_size))
		self.mpns = nn.ModuleList(mpns_list)
		self.mpns_list = mpns_list


		self.stats=dict(
			noadj={},
		)
		if verbose:
			print('MPRNN')
			print(' [*] Defined over: %d nodes' % len(nodes))
			print(' [*] Contains    : %d adjs' % len(adj))

	def ckpt_path(self, hops=None):
		if hops is None:
			hops = self.hops

		sfile = '%s/%s/%s_n%d.pth' % (
			CKPT_STORAGE,
			self.name,
			self.nodes[0],
			hops,
		)
		return sfile

	def load_prior(self):
		wpath = self.ckpt_path(hops=self.hops-1)

		print('Transfer:', wpath)

		pdict = torch.load(wpath)

		self_dict = self.state_dict()
		match = 0
		for k, v in self_dict.items():
			if k in pdict:
				match += 1
				self_dict[k] = pdict[k]
		self.load_state_dict(self_dict)

		print('Matched: %d params' % match)

	def load(self, wpath=None):
		if wpath is None:
			wpath = self.ckpt_path()

		print('Loading:', wpath)
		self.load_state_dict(torch.load(wpath))

	def save(self):
		sfile = self.ckpt_path()
		print('Saving to:', sfile)
		torch.save(self.state_dict(), sfile)

	def clear_stats(self):
		pass
		# self.msgcount = [0 for _ in self.nodes]
		# self.updcount = [0 for _ in self.nodes]

	def print_stats(self):
		pass
		# for ni, nname in enumerate(self.nodes):
		# 	print('Node:', nname)
		# 	print('  * Msgs received: %d' % self.msgcount[ni])
		# 	print('  * Msgs updated : %d' % self.updcount[ni])

	def eval_hidden(self, ti, nodes, hidden):
		hevals = []
		for ni, (node_series, rnn, hdn) in enumerate(zip(nodes, self.rnns, hidden)):
			value_t = node_series[ti]

			hin = rnn.inp(value_t).unsqueeze(0)
			hout, hdn = rnn.rnn(hin, hdn)
			hout = hout.squeeze(0)

			hevals.append(hout)
			hidden[ni] = hdn # replace previous lstm params
		return hevals

	def eval_message(self, hevals):
		msgs = []
		for ni, (hval, nname) in enumerate(zip(hevals, self.nodes)):
			# only defined over nodes w/ adj
			if not len(self.adj[nname]):
				self.stats['noadj'][nname] = True
				msgs.append(None)
				continue

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

	def eval_update(self, hevals, msgs):
		for ni, (hval, msg, nname) in enumerate(zip(hevals, msgs, self.nodes)):
			# only defined over nodes w/ adj
			if msg is None: continue
			mpn = self.mpns[self.mpn_ind[nname]]

			# replaces hvalues before update
			hevals[ni] = mpn.upd(hval, msg)

			# self.updcount[ni] += 1

	def eval_readout(self, hevals, hidden):
		values_t = []
		for ni, (hval, rnn) in enumerate(zip(hevals, self.rnns)):
			values_t.append(rnn.out(hval))
		return values_t

	def forward(self, series, hidden=None, dump=False):
		# print(len(series), len(series[0]), series[0][0].size())
		assert len(self.rnns) == len(series)

		# lstm params
		if hidden is None:
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

			# message passing
			msgs = self.eval_message(hevals)

			# updating hidden
			self.eval_update(hevals, msgs)

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
		sch = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
		return criterion, opt, sch