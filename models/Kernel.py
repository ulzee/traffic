
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from models.template import Template

class Kernel(nn.Module):
	# name = 'cast'
	def __init__(self,
			insize=2, # self + 1 neighbor (default)
			hsize=64 # size of embedding vector
		):

		super(Kernel, self).__init__()

		self.insize = insize
		self.hsize = hsize

		def inst_msgop(has_h=True):
			v_in = nn.Sequential(
				nn.Linear(1, hsize)
			)
			if has_h:
				# h_in = nn.Sequential(
				# 	nn.Linear(hsize, hsize)
				# )
				h_in = None
			# msg_out = nn.Sequential(
			# 	nn.Linear(hsize * (2 if has_h else 1), hsize),
			# 	# nn.ReLU(),
			# 	# nn.Linear(hsize, hsize),
			# 	nn.Sigmoid(),
			# 	# nn.ReLU(),
			# )
			msg_out = None
			# return v_in, h_in, msg_out
			# if has_h:
			# 	return nn.ModuleList([v_in, h_in, msg_out])
			# else:
				# return nn.ModuleList([v_in, msg_out])
			return nn.ModuleList([v_in])

		# self.mops_0 = nn.ModuleList()
		# for _ in range(insize):
		# 	self.mops_0.append(inst_msgop(has_h=False))

		self.mops = nn.ModuleList() # independent weights per neighbor
		for _ in range(insize):
			self.mops.append(inst_msgop(has_h=True))

	def forward(self, node, initial=False):
		assert len(node.ns)

		msgs = []
		# rel_nodes = node.ns + [node] # incl. itself
		rel_nodes = node.ns # incl. itself
		# if initial:
		for (v_in,), other in zip(self.mops, rel_nodes):
			mix = v_in(other.v)
			# msgout = msg_out(mix)
			msgs.append(mix)
		# else:
		# 	for (v_in, h_in, msg_out), other in zip(self.mops, rel_nodes):
		# 		mix = torch.cat([v_in(other.v), h_in(other.h)], dim=1)
		# 		msgout = msg_out(mix)
		# 		msgs.append(msgout)
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
		# self.h_in = nn.Sequential(
		# 	nn.Linear(hsize, hsize)
		# )
		# self.msg_in = nn.Sequential(
		# 	nn.Linear(hsize, hsize)
		# )
		# self.h_dense = nn.Sequential(
		# 	# nn.Linear(hsize * 2, hsize),
		# 	# nn.ReLU(),
		# 	nn.Linear(hsize, hsize),
		# 	# nn.ReLU(),
		# 	# nn.Linear(hsize, hsize),
		# 	# nn.Dropout(0.5)
		# 	# nn.ReLU()
		# )
		# self.h_after = nn.Sequential(
		# 	# nn.Linear(hsize, hsize),
		# 	# nn.ReLU(),
		# 	nn.Linear(hsize, hsize+1),
		# 	# nn.Sigmoid(),
		# 	# nn.ReLU()
		# )
		# self.h_out = nn.Sequential(
		# 	nn.Linear(hsize, hsize),
		# 	nn.ReLU(),
		# 	nn.Linear(hsize, hsize),
		# 	nn.Sigmoid(),
		# )
		self.v_out = nn.Sequential(
			# nn.Linear(hsize, hsize),
			# nn.ReLU(),
			nn.Linear(hsize, 1)
		)
		# in: seqlen x batch x features
		self.rnn = nn.LSTM(hsize, hsize, 1)


	def forward(self, node):
		assert type(node.msg) is not type(None)

		# mix = torch.cat([self.h_in(node.h), self.msg_in(node.msg)], dim=1)

		# mix = self.h_dense(mix)
		mix = node.msg
		node.msg = None
		# in: seqlen x batch x dims
		mix, hidden = self.rnn(mix.unsqueeze(0), node.hidden)
		node.hidden = hidden
		mix = mix.squeeze(0)
		# mix = self.h_after(mix)

		# h and v are updated
		# node.v, node.h = self.v_out(mix), self.h_out(mix)
		node.v = self.v_out(mix)
		# node.v, node.h = torch.split(mix, [1, self.hsize], dim=1)
		# node.h = nn.Sigmoid()(node.h)
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

def message(kernels, graph_t, initial=False):
	# op = lambda kern, node: kern['op'](node, initial) if initial else \
	# 	lambda kern, node: kern['op'](node)
	op = lambda kern, node: kern['op'](node)
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
