
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNN(nn.Module):
	name = 'rnn'
	def __init__(self, hidden_size=128, forecast=5, relu=False, deep=False):
		super(RNN, self).__init__()
		self.relu = relu
		self.lag = 5 # temporal dimension
		self.steps = 10 # spatial dimension (optional ?)
		self.hidden_size = hidden_size
		self.forecast = forecast

		if deep:
			self.name += '_deep'
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
			self.inp = nn.Linear(self.lag, hidden_size)
			self.out = nn.Linear(hidden_size, 1)
			self.fcast = nn.Linear(hidden_size, self.forecast)

		self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)


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
		outputs = []

		for ii in range(steps):
			output, hidden = self.step(inputs[ii], hidden)
			outputs.append(output)

		# uses the final hidden state to forecast further back
		if self.forecast > 0:
			h_f = hidden[0] # hvector, hparams
			h_f = h_f[-1]   # last lstm layer vector
			output = self.fcast(h_f)
			# dims: forecast x batch size
			outputs.append(output)

		# outputs = list(reversed(outputs))
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

	def format_batch(self, mat, ys):
		# raw   : batch x timelen x seqlen
		# needed: seqlen x batch x timelen

		steps = mat.shape[2] - self.forecast
		# withold steps for forecasting

		batch = []
		for si in range(steps):
			batch.append(torch.Tensor(mat[:, :, self.forecast+si]).cuda())

		batch = list(reversed(batch))
		# ys = ys[:, :5] # FIXME:
		ys = np.flip(ys, axis=1).copy()
		# sequence order is reversed, to infer traffic upstream

		ys = torch.from_numpy(ys).float().cuda()
		return batch, ys
