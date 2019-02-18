
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNN2(nn.Module):
	name = 'rnn2'
	def __init__(self, hidden_size=128, forecast=5):
		super(RNN2, self).__init__()
		self.lag = 5 # temporal dimension
		self.steps = 10 # spatial dimension (optional ?)
		self.hidden_size = hidden_size
		self.forecast = forecast

		hsize = hidden_size
		self.inp = nn.Sequential(
			nn.Linear(self.steps - self.forecast, hsize),
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
			nn.Linear(hsize, self.steps - self.forecast),
		)
		self.fcast = nn.Sequential(
			nn.ReLU(),
			nn.Linear(hsize, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
			nn.ReLU(),
			nn.Linear(hsize, self.forecast),
		)

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
			rnn_out, hidden = self.step(inputs[ii], hidden)
		# outputs.append(output)

		# uses the final hidden state to forecast further back
		h_f = hidden[0] # hvector, hparams
		h_f = h_f[-1]   # last lstm layer vector
		forecast_out = self.fcast(h_f)
		# outputs.append(output)

		# t-h ... t
		outputs = torch.cat([forecast_out, rnn_out], dim=1)
		return outputs

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
		return criterion, opt, sch

	# import torch
	# import torch.nn as nn
	# import numpy as np

	def format_batch(self, mat, ys, gpu=None):
		# raw   : batch x timelen x seqlen
		# needed: seqlen x batch x timelen

		steps = mat.shape[2] - self.forecast
		# withold steps for forecasting

		batch = []
		for ti in range(mat.shape[1]):
			batch.append(torch.Tensor(mat[:, ti, -steps:]).to(gpu))

		ys = torch.from_numpy(ys).float().to(gpu)
		return batch, ys
