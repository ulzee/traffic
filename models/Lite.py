
class MP_LITE(MP_THIN):
	def __init__(self, hsize):
		super(MP_LITE, self).__init__(hsize)

		# passed msg is linear
		msize = hsize//2
		self.msg_op = nn.Sequential(
			nn.Linear(hsize*2, msize),
			# nn.ReLU(),
			# # nn.Linear(hsize, hsize),
			# # nn.ReLU(),
			# nn.Linear(hsize, hsize),
		)
		self.upd_op = nn.Sequential(
			nn.Linear(hsize + msize, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
		)

class MPRNN_LITE(GRNN):
	'''
	Defines one convolution operator over all nodes
	'''

	def __init__(self,
		nodes, adj,

		iters=5,

		hidden_size=256,
		rnnmdl=RNN,
		# mpnmdl=MP_THIN,
		verbose=False):
		super(MPRNN_LITE, self).__init__(len(nodes), hidden_size, rnnmdl)

		self.adj = adj
		self.nodes = nodes
		self.iters = iters

		mplite = MP_LITE(hsize=hidden_size)
		self.mpn = mplite

		self.stats=dict(
			noadj={},
		)
		if verbose:
			print('MPRNN')
			print(' [*] Defined over: %d nodes' % len(nodes))
			print(' [*] Contains    : %d adjs' % len(adj))

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

	def eval_readout(self, hevals):
		values_t = []
		for ni, (hval, rnn) in enumerate(zip(hevals, self.rnns)):
			values_t.append(rnn.out(hval))
		return values_t

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.Adam(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
		return criterion, opt, sch

	def eval_message(self, it, hevals):
		msgs = []

		mpn = self.mpn
		nmat = [] # collects matrix of neighbors whose msgs must be computed
		hmat = [] # self's hidden val is also needed to compute msg
		for ni, (hval, nname) in enumerate(zip(hevals, self.nodes)):
			if not len(self.adj[nname]):
				continue

			ninds = [self.nodes.index(neighb) for neighb in self.adj[nname]]
			for neighbor_i in ninds:
				nmat.append(hevals[neighbor_i])
				hmat.append(hval)

		nmat = torch.stack(nmat, dim=1)
		hmat = torch.stack(hmat, dim=1)
		mmat = mpn.msg(hmat, nmat)
		mflat = [tens.squeeze(1) for tens in torch.split(mmat, 1, dim=1)]

		for ni, (hval, nname) in enumerate(zip(hevals, self.nodes)):
			if not len(self.adj[nname]):
				self.stats['noadj'][nname] = True
				msgs.append(None)
				continue

			many = []
			ninds = [self.nodes.index(neighb) for neighb in self.adj[nname]]
			assert len(ninds)
			for _ in ninds:
				many.append(mflat.pop(0))
			many = torch.stack(many, -1)
			msg = torch.sum(many, -1)
			msgs.append(msg)
		assert len(mflat) == 0

		return msgs

	def eval_update(self, it, hevals, msgs):
		assert len(msgs) == len(hevals)

		mpn = self.mpn # shared mpn
		mmat = []
		hmat = []
		for ni, (hval, msg, nname) in enumerate(zip(hevals, msgs, self.nodes)):
			if msg is None: continue
			hmat.append(hval)
			mmat.append(msg)

		mmat = torch.stack(mmat, dim=1)
		hmat = torch.stack(hmat, dim=1)
		umat = mpn.upd(hmat, mmat)
		uflat = [tens.squeeze(1) for tens in torch.split(umat, 1, dim=1)]

		for ni, (hval, msg, nname) in enumerate(zip(hevals, msgs, self.nodes)):
			if msg is None: continue
			hevals[ni] = uflat.pop(0)
		assert len(uflat) == 0

	def forward(self, series, hidden=None, dump=False):
		from time import time
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