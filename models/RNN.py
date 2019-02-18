
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNN(nn.Module):
	name = 'rnn_unroll'
	# def __init__(self, hidden_size=128, forecast=5, relu=False, deep=False):
	def __init__(self, hidden_size=256, forecast=5):
		super(RNN, self).__init__()
		# self.relu = relu
		self.lag = 5 # temporal dimension
		self.steps = 10 # spatial dimension (optional ?)
		self.hidden_size = hidden_size
		self.forecast = forecast

		# if deep:
		hsize = hidden_size
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
			nn.Linear(hsize, 1),
			# nn.Linear(hsize, self.lag),
		)
		self.fcast = nn.Sequential(
			nn.ReLU(),
			nn.Linear(hsize, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
			nn.ReLU(),
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
		lastKnown = steps - self.forecast - 1
		outputs = []

		for ii in range(steps - 1):
			output, hidden = self.step(inputs[ii], hidden)

			# last recurrent step starts producing first forecast
			if ii >= lastKnown:
				outputs.append(output)
				# past prediction is used as input
				# if inputs[ii+1] is None:
				# 	inputs[ii+1] = torch.zeros()

		outputs = torch.cat(outputs, dim=1)
		return outputs, hidden

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
		return criterion, opt, sch

	# import torch
	# import torch.nn as nn
	# import numpy as np

	def format_batch(self, mat, ys, gpu=None):
		# raw   : batch x timelen x seqlen
		# needed: seqlen x batch x timelen

		steps = mat.shape[2] - self.forecast
		# withold steps for forecasting

		seqX = []
		for _ in range(self.forecast - 1):
			seqX.append(None)
			# seqX.append(torch.zeros(mat.shape[0], mat.shape[1]).to(gpu))
		for si in range(steps):
			seqX.append(torch.Tensor(mat[:, :, self.forecast+si]).to(gpu))
		seqX = list(reversed(seqX))

		ys = ys[:, :5] # forecasting earlier stops
		# yhist = mat[:, :, :5]
		# ys = np.concatenate([np.expand_dims(ys, 1), yhist], axis=1)
		ys = np.flip(ys, axis=-1).copy()
		# sequence order is reversed, since rnn is unrolled upstream

		ys = torch.from_numpy(ys).float().to(gpu)
		return seqX, ys
