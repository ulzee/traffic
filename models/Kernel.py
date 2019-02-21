
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.template import Template

class Kernel(Template):
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

class Update(Template):
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
		self.h_out = nn.Sequential(
			nn.Linear(hsize * 2, hsize + 1)
		)

	def forward(self, node):
		if type(node.msg) is type(None): # an end node
			return

		mix = torch.cat([self.h_in(node.h), self.msg_in(node.msg)], dim=1)

		# h and v are updated
		node.v, node.h = torch.split(self.h_out(mix), [1, self.hsize], dim=1)
		return node.v, node.h
