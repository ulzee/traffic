
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from models.template import Template

class Kernel(nn.Module):
	# name = 'cast'
	def __init__(self,
			insize=2, # self + 1 neighbor (default)
			hsize=64, # size of embedding vector
			iterations=5,
		):

		super(Kernel, self).__init__()

		self.insize = insize
		self.hsize = hsize

		def inst_msgop():
			msg_out = nn.Sequential(
				nn.Linear(1 + hsize, hsize)
			)
			return nn.ModuleList([msg_out])

		# NOTE: independent weights per neighbor
		# NOTE: independent weights per iteration level
		self.mops = nn.ModuleList()
		for iter in range(iterations):
			at_iter = nn.ModuleList()
			for _ in range(insize):
				at_iter.append(inst_msgop())
			self.mops.append(at_iter)
		assert len(self.mops) == iterations

	def forward(self, node, iteration):
		assert len(node.ns)

		msgs = []
		rel_nodes = node.ns
		mops = self.mops[iteration]
		for (msg_out,), other in zip(mops, rel_nodes):
			# mix = torch.cat([other.v, other.h], dim=1)
			mix = torch.cat([other.v, other.h], dim=1)
			mix = msg_out(mix)
			msgs.append(mix)

		assert len(msgs)
		msgs = torch.stack(msgs, dim=1).to(self.device)
		msg_result = torch.sum(msgs, dim=1)
		node.msg = msg_result
		return msg_result

class Update(nn.Module):
	# name = 'cast'
	def __init__(self, hsize=64):
		super(Update, self).__init__()

		self.hsize = hsize
		self.v_out = nn.Sequential(
			nn.Linear(hsize, 1)
		)
		self.h_out = nn.Sequential(
			nn.Linear(hsize, hsize)
		)
		# in: seqlen x batch x features
		# self.rnn = nn.LSTM(hsize, hsize, 1)
		self.rnn = nn.Sequential(
			nn.Linear(hsize, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize)
		)

	def forward(self, node):
		assert type(node.msg) is not type(None)

		mix = node.msg
		node.msg = None
		# in: seqlen x batch x dims
		# mix, hidden = self.rnn(mix.unsqueeze(0), node.hidden)
		# node.hidden = hidden
		# mix = mix.squeeze(0)
		mix = self.rnn(mix)

		node.h = self.h_out(mix)
		# mix = self.h_after(mix)

		# h and v are updated
		node.v = self.v_out(mix)
		return node.v

class Node:
	def __init__(self, value, zero, device=None):
		self._v = value.clone().to(device).float() # .v may be modified
		self.v = value.to(device).float()
		self.h = zero().to(device).float()
		self.hidden = None
		self.ns = [] # neighbors

	def show(self):
		pnt = self
		while len(pnt.ns):
			print(pnt.v.size(), end=' ')
			pnt = pnt.ns[0]
		print()

	def ln(self):
		l = 0
		pnt = self
		while len(pnt.ns):
			pnt = pnt.ns[0]
			l += 1
		return l

def routeToGraph(batch, zero, device=None):
	at_time = [] # t-h ... t
	for time in torch.split(batch, 1, dim=1):
		root, pnt = None, None
		for stop in torch.split(time.squeeze(1), 1, dim=1):
			if root is None:
				root = Node(stop, zero, device=device)
				pnt = root
			else:
				nd = Node(stop, zero, device=device)
				pnt.ns.append(nd)
				pnt = nd
		at_time.append(root)
	return at_time

def inst_tree(struct, node, device=None):
	ls = []
	params = []
	nodeObj = None
	if len(node.ns): # instance only nodes w/ neighbors
		inst = struct(node)
		inst.device = device
		params += list(inst.parameters())
		nodeObj = dict(
			op=inst,
			ns=ls
		)
		for neighbor in node.ns:
			ns, nparams = inst_tree(struct, neighbor, device)
			params += nparams
			if ns is not None: ls.append(ns) # gather results
	return nodeObj, params

def zip_op(t1, t2, op):
	it = 0
	ls = [(t1, t2)]
	while len(ls):
		n1, n2 = ls[0]
		# print('zip', len(n1['ns']), type(n1['op']), it)
		it += 1
		ls = ls[1:]
		op(n1, n2)
		# print('end')
		for c1, c2 in zip(n1['ns'], n2.ns):
			ls.append((c1, c2))

def message(kernels, graph_t, iteration):
	op = lambda kern, node: kern['op'](node, iteration)
	zip_op(kernels, graph_t, op=op)

def update(upops, graph_t):
	zip_op(upops, graph_t, op=lambda up, node: up['op'](node))

def count_rec(node, cf):
	return 1 + sum([count_rec(cn, cf) for cn in cf(node)])

def gather_predictions(_node, node):
	# end nodes do not hold convolved results, so they are ignored
	ls = [(_node._v, node.v)] if len(node.ns) else []
	for _nb, nb in zip(_node.ns, node.ns):
		ls += gather_predictions(_nb, nb)
	return ls

def reassign_v(states, graph_t):
	ls = [(states, graph_t)]
	while len(ls):
		n1, n2 = ls[0]
		ls = ls[1:]
		n1.v = n2._v.clone()
		for c1, c2 in zip(n1.ns, n2.ns):
			ls.append((c1, c2))

if __name__ == '__main__':
	model = Kernel()
	print(list(model.parameters()))
