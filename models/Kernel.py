
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
		# self.hsize = hsize

		def inst_msgop():
			v_in = nn.Sequential(
				nn.Linear(1, hsize)
			)
			h_in = nn.Sequential(
				nn.Linear(hsize, hsize)
			)
			msg_out = nn.Sequential(
				nn.Linear(hsize * 2, hsize)
			)
			# return v_in, h_in, msg_out
			return nn.ModuleList([v_in, h_in, msg_out])

		self.mops = nn.ModuleList() # independent weights per neighbor
		for _ in range(insize):
			self.mops.append(inst_msgop())
		# self.upop = nn.Sequential(
		# 	nn.Linear(hsize * 2, hsize + 1)
		# )

	def forward(self, node):
		if len(node.ns) == 0:
			node.msg = None
			return None

		msgs = []
		rel_nodes = node.ns + [node] # incl. itself
		for (v_in, h_in, msg_out), other in zip(self.mops, rel_nodes):
			mix = torch.cat([v_in(other.v), h_in(other.h)], dim=1)
			msgout = msg_out(mix)
			msgs.append(msgout)
		assert len(msgs)
		msgs = torch.stack(msgs, dim=1).to(self.device)
		update = torch.sum(msgs, dim=1)
		node.msg = update
		return update

class Update(nn.Module):
	# name = 'cast'
	def __init__(self, hsize=64):
		super(Update, self).__init__()

		self.hsize = hsize
		self.h_in = nn.Sequential(
			nn.Linear(hsize, hsize)
		)
		self.msg_in = nn.Sequential(
			nn.Linear(hsize, hsize)
		)
		self.h_dense = nn.Sequential(
			nn.Linear(hsize * 2, hsize),
			# nn.ReLU(),
			# nn.Linear(hsize, hsize),
			# nn.ReLU()
		)
		self.h_after = nn.Sequential(
			nn.Linear(hsize, hsize+1),
			# nn.ReLU(),
			# nn.Linear(hsize, hsize),
			# nn.ReLU()
		)
		# self.h_out = nn.Sequential(
		# 	nn.Linear(hsize, hsize + 1)
		# )
		# self.v_out = nn.Sequential(
		# 	nn.Linear(hsize, 1)
		# )
		# in: seqlen x batch x features
		self.rnn = nn.LSTM(hsize, hsize, 2)


	def forward(self, node):
		if type(node.msg) is type(None): # an end node
			return

		mix = torch.cat([self.h_in(node.h), self.msg_in(node.msg)], dim=1)

		mix = self.h_dense(mix)
		mix, hidden = self.rnn(mix.unsqueeze(0), node.hidden)
		node.hidden = hidden
		mix = mix.squeeze(0)
		mix = self.h_after(mix)

		# h and v are updated
		# node.v, node.h = self.v_out(mix), self.h_out(mix)
		node.v, node.h = torch.split(mix, [1, self.hsize], dim=1)
		return node.v, node.h

class Node:
	def __init__(self, value, zero, device=None):
		self._v = value.clone().to(device).float() # label
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

def inst_tree(struct, nodes, device=None):
	ls = []
	params = []
	for ent in nodes:
		inst = struct(ent) #.to(device)
		params += list(inst.parameters())
		inst.device = device
		ns, nparams = inst_tree(struct, ent.ns)
		params += nparams
		kobj = dict(
			op=inst,
			ns=ns
		)
		ls.append(kobj)
	return ls, params

if __name__ == '__main__':
	model = Kernel()
	print(list(model.parameters()))
