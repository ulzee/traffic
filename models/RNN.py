
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNN(nn.Module):
	name = 'rnn_unroll'
	# def __init__(self, hidden_size=128, forecast=5, relu=False, deep=False):
	def __init__(self, hidden_size=256, forecast=5, deep=True):
		super(RNN, self).__init__()
		# self.relu = relu
		self.lag = 6 # temporal dimension
		# self.lag = 5 # temporal dimension
		# self.steps = 10 # spatial dimension (optional ?)
		self.hidden_size = hidden_size
		self.forecast = forecast

		hsize = hidden_size
		if deep:
			self.inp = nn.Sequential(
				nn.Linear(self.lag, hsize),
				nn.ReLU(),
				nn.Linear(hsize, hsize),
				nn.ReLU(),
				nn.Linear(hsize, hsize),
				nn.ReLU(),
			)
			self.out = nn.Sequential(
				nn.ReLU(),
				nn.Linear(hsize, hsize),
				nn.ReLU(),
				nn.Linear(hsize, hsize),
				nn.ReLU(),
				nn.Linear(hsize, self.lag),
			)
			self.fcast = nn.Sequential(
				nn.ReLU(),
				nn.Linear(hsize, hsize),
				nn.ReLU(),
				nn.Linear(hsize, hsize),
				nn.ReLU(),
				nn.Linear(hsize, self.forecast),
			)
		else:
			self.name += '_short'
			self.inp = nn.Sequential(
				nn.Linear(self.lag, hsize),
			)
			self.out = nn.Sequential(
				nn.Linear(hsize, 1),
			)
			self.fcast = nn.Sequential(
				nn.Linear(hsize, self.forecast),
			)

		# else:
		# 	self.inp = nn.Linear(self.lag, hidden_size)
		# 	self.out = nn.Linear(hidden_size, 1)
		# 	self.fcast = nn.Linear(hidden_size, self.forecast)

		self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
		self.bsize = 32


	def step(self, input, hidden=None):
		# seqlen = 1 for stepwise eval
		# in: batch x inputsize
		if input is not None:
			input = self.inp(input)
			input = input.unsqueeze(0)
		else:
			input = torch.zeros(1, self.bsize, self.hidden_size).to(self.device)
		# in: seqlen x batch x inputsize
		output, hidden = self.rnn(input, hidden)
		# out: seqlen x batch x hiddensize
		output = self.out(output.squeeze(0))
		return output, hidden

	def forward(self, inputs, hidden=None):
		steps = len(inputs)
		# lastKnown = steps - self.forecast - 1
		outputs = []

		for ii in range(steps):
			output, hidden = self.step(inputs[ii], hidden)
			outputs.append(output)

		outputs = torch.stack(outputs, dim=0)
		# return outputs, hidden
		return outputs

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=60, gamma=0.5)
		return criterion, opt, sch

	def format_batch(self, data, wrap=True, normalize=1):
		# raw   : batch x timelen x seqlen
		data /= normalize
		data = torch.transpose(torch.transpose(data, 2, 1), 1, 0)
		# fmt   : seqlen x batch x timelen
		sequence = list(torch.split(data, 1, dim=0))

		for ti in range(len(sequence)):
			sequence[ti] = sequence[ti].to(self.device).float().squeeze(0)

		Xs = list(reversed(sequence[1:]))
		Ys = list(reversed(sequence[:-1])) # predict 1 stop back
		Ys = torch.stack(Ys, dim=0)

		return Xs, Ys
