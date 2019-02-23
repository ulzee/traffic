
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Cast(nn.Module):
	name = 'cast'
	def __init__(self, hidden_size=256, forecast=5, deep=True, lag=6):
		super(Cast, self).__init__()
		self.lag = lag # temporal dimension
		self.steps = 10 # spatial dimension (optional ?)
		self.hidden_size = hidden_size
		self.forecast = forecast

		hsize = hidden_size
		self.insize = self.steps - self.forecast # known input
		self.outsize = self.forecast
		if deep:
			self.inp = nn.Sequential(
				nn.Linear(self.insize, hsize),
				nn.ReLU(),
				nn.Linear(hsize, hsize),
				nn.ReLU(),
				nn.Linear(hsize, hsize),
				nn.ReLU(),
				# nn.Dropout(0.5),
			)
			self.out = nn.Sequential(
				# nn.Dropout(0.5),
				nn.ReLU(),
				nn.Linear(hsize, hsize),
				nn.ReLU(),
				nn.Linear(hsize, hsize),
				nn.ReLU(),
				nn.Linear(hsize, self.outsize),
			)
		else:
			self.name += '_short'
			self.inp = nn.Sequential(
				nn.Linear(self.insize, hsize),
			)
			self.out = nn.Sequential(
				nn.Linear(hsize, self.outsize),
			)

		self.rnn = nn.LSTM(hidden_size, hidden_size, 1)
		self.bsize = 32


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

	def forward(self, inputs, hidden=None):
		steps = len(inputs)

		for ii in range(steps):
			output, hidden = self.step(inputs[ii], hidden)
			# outputs.append(output)
		return output

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.75)
		return criterion, opt, sch

	def format_batch(self, data, wrap=True):
		known = data[:, :, self.forecast:]
		# raw   : batch x timelen x seqlen
		data = torch.transpose(known, 1, 0)
		# fmt   : timelen x batch x seqlen
		sequence = list(torch.split(data, 1, dim=0))

		for ti in range(len(sequence)):
			sequence[ti] = sequence[ti].to(self.device).float().squeeze(0)

		Xs = sequence
		# print(len(Xs), Xs[0].size())
		Ys = data[-1, :, :self.forecast].to(self.device).float()
		# print(Ys.size())
		# assert False

		return Xs, Ys
