
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNN(nn.Module):
	def __init__(self, hidden_size=128, forecast=5, relu=False):
		super(RNN, self).__init__()
		self.relu = relu
		self.lag = 5 # temporal dimension
		self.steps = 10 # spatial dimension (optional ?)
		self.hidden_size = hidden_size

		self.inp = nn.Linear(self.lag, hidden_size)
		self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
		self.out = nn.Linear(hidden_size, 1)

		self.forecast = forecast
		self.fcast = nn.Linear(hidden_size, self.forecast)

	def step(self, input, hidden=None):
		# seqlen = 1 for stepwise eval
		# in: batch x inputsize
		input = self.inp(input)
		input = input.unsqueeze(0)
		if self.relu:
			input = nn.ReLU()(input)
		# in: seqlen x batch x inputsize
		output, hidden = self.rnn(input, hidden)
		if self.relu:
			output = nn.ReLU()(output)
		# out: seqlen x batch x hiddensize
		output = self.out(output.squeeze(0))
		return output, hidden

	def forward(self, inputs, hidden=None):
		steps = len(inputs)
		outputs = []

		for ii in range(steps):
			output, hidden = self.step(inputs[ii], hidden)
			outputs.append(output)

		# uses the hidden state to forecast further back
		if self.forecast > 0:
			h_f = hidden[0]
			if self.relu:
				h_f = nn.ReLU()(h_f)
			h_f = h_f[-1] # last lstm layer vector
			output = self.fcast(h_f)
			# output = torch.t(output)
			# dims: forecast x batch size
			outputs.append(output)

		outputs = torch.cat(outputs, dim=1)
		return outputs, hidden

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
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
			batch.append(torch.Tensor(mat[:, :, si]).cuda())

		batch = list(reversed(batch))
		ys = np.flip(ys, axis=1).copy()
		# sequence order is reversed, to infer traffic upstream

		ys = torch.from_numpy(ys).float().cuda()
		return batch, ys
