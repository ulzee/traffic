
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.temporal.RNN import *

class GRNN(nn.Module):
	'''
	Instantiates one RNN per location in input graph
	'''

	def __init__(self,
		nodes=1,
		hidden_size=256,
		rnnmdl=RNN_MIN):
		super().__init__()

		self.lag = 5 # min needed for inference
		self.hidden_size = hidden_size

		many_rnns = [rnnmdl(hidden_size, 1) for _ in range(nodes)]
		self.rnns = nn.ModuleList(many_rnns)
		# self.inner_rnns = nn.ModuleList(many_rnns)

	def forward(self, nodes, hidden=None, dump=False):
		assert len(self.rnns) == len(nodes)
		if hidden is None:
			hidden = [None] * len(nodes)

		outs_bynode = []
		hs_bynode = []
		for inputs, mdl, hdn in zip(nodes, self.rnns, hidden):
			outputs, hout = mdl(inputs, hidden=hdn, dump=True)
			outs_bynode.append(outputs)
			hs_bynode.append(hout)

		out = torch.cat(outs_bynode, dim=2)
		return out, hs_bynode

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
		return criterion, opt, sch

	def format_batch(self, data, withold=None):
		all_known = data[:, :, :]

		bystop = torch.split(all_known, 1, 2)
		allXs, allYs = [], []
		for known in bystop:
			# raw   : batch x timelen x seqlen
			data = torch.transpose(known, 1, 0)
			# fmt   : timelen x batch x seqlen

			bytime = list(torch.split(data, 1, dim=0))
			seq = list(map(lambda tens: tens.to(self.device).float().squeeze(0), bytime))

			Xs = seq[:-1]
			Ys = seq[1:] # predict immediately following values

			Ys = torch.stack(Ys, dim=1) # restack Ys by temporal
			allXs.append(Xs)
			allYs.append(Ys)

		allYs = torch.cat(allYs, dim=2)
		# print(len(allXs), allYs.size())
		# assert False
		return allXs, allYs
