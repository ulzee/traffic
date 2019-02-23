
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNN_0(nn.Module):
	name = 'rnn_0'
	def __init__(self, hidden_size=256, deep=False, lag=12):
		super(RNN_0, self).__init__()

		self.lag = lag # temporal dimension
		self.steps = 1 # spatial dimension (optional ?)
		self.hidden_size = hidden_size

		hsize = hidden_size
		self.insize = self.steps
		self.outsize = self.steps
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
		time_steps = len(inputs)

		for ii in range(time_steps):
			output, hidden = self.step(inputs[ii], hidden)
			# outputs.append(output)
		return output

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.2)
		return criterion, opt, sch

	def format_batch(self, data, wrap=True):
		# raw   : batch x timelen x seqlen
		known = data[:, :, 0:1]
		data = torch.transpose(known, 1, 0)
		# fmt   : timelen x batch x seqlen
		timesteps = list(torch.split(data, 1, dim=0))
		sequence = []
		for ti, step in enumerate(timesteps[:-1]):
			sequence.append(step.clone().to(self.device).float().squeeze(0))

		Xs = sequence
		Ys = data[-1, :, :].clone().to(self.device).float()

		return Xs, Ys
