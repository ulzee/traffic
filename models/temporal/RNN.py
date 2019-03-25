
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *

class RNN_MIN(nn.Module):
	name = 'rnn'
	def __init__(self, hidden_size=256, steps=10):
		super(RNN_MIN, self).__init__()

		self.lag = 5 # min needed for inference
		self.steps = steps # spatial dimension (optional ?)
		self.hidden_size = hidden_size

		self.insize = self.steps
		self.outsize = self.steps

		hsize = hidden_size
		self.inp = nn.Sequential(
			nn.Linear(self.insize, hsize),
		)
		self.out = nn.Sequential(
			nn.Linear(hsize, self.outsize),
		)

		self.rnn = nn.LSTM(hidden_size, hidden_size, 1)

	def step(self, input, hidden=None):
		# seqlen = 1 for stepwise eval
		# in: batch x inputsize
		input = self.inp(input)
		input = input.unsqueeze(0)

		# in: seqlen x batch x inputsize
		output, hidden = self.rnn(input, hidden)
		# out: seqlen x batch x hiddensize

		output = self.out(output.squeeze(0))
		return output, hidden

	def forward(self, inputs, hidden=None, dump=False, wrap=True):
		temporal = len(inputs)

		outputs = []
		for ii in range(temporal):
			output, hidden = self.step(inputs[ii], hidden)
			outputs.append(output)

		if wrap:
			outputs = torch.stack(outputs, dim=1)

		if not dump: return outputs
		return outputs, hidden

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.Adam(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
		return criterion, opt, sch

	def format_batch(self, data, withold=None):
		known = data[:, :, :self.steps]

		# raw   : batch x timelen x seqlen
		data = torch.transpose(known, 1, 0)
		# fmt   : timelen x batch x seqlen

		bytime = list(torch.split(data, 1, dim=0))
		seq = list(map(lambda tens: tens.to(self.device).float().squeeze(0), bytime))

		Xs = seq[:-1]
		Ys = seq[1:] # predict immediately following values

		Ys = torch.stack(Ys, dim=1) # restack Ys by temporal

		return Xs, Ys

class RNN(RNN_MIN):
	name = 'rnn'
	def __init__(self, hidden_size=256, steps=10):
		super(RNN, self).__init__(hidden_size, steps)

		hsize = hidden_size
		self.inp = nn.Sequential(
			nn.Linear(self.insize, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
		)
		self.out = nn.Sequential(
			nn.Linear(hsize, hsize),
			nn.ReLU(),
			nn.Linear(hsize, self.outsize),
		)

		self.rnn = nn.LSTM(hidden_size, hidden_size, 1)

class RNN_SNG(RNN_MIN):
	def __init__(self, hidden_size=128, steps=1):
		super(RNN_SNG, self).__init__(hidden_size, steps)

		hsize = hidden_size
		self.inp = nn.Sequential(
			nn.Linear(self.insize, self.insize)
		)
		self.out = nn.Sequential(
			# nn.Linear(hsize, hsize),
			# nn.ReLU(),
			# nn.Linear(hsize, hsize),
			# nn.ReLU(),
			nn.Linear(hsize, self.outsize),
		)

		self.rnn = nn.LSTM(1, hidden_size, 1)


class RNN_FCAST(RNN):

	def __init__(self, nodes, adj, hidden_size=256, steps=10):
		super(RNN_FCAST, self).__init__(hidden_size, steps)

		fringes = find_fringes(nodes, adj, twoway=True)
		# nodes, adj = complete_graph(nodes, adj)
		self.fringes = fringes
