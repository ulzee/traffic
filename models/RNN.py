
import torch
import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):
	def __init__(self, hidden_size=128, relu=False):
		super(RNN, self).__init__()
		self.relu = relu
		self.lag = 5 # temporal dimension
		self.steps = 10 # spatial dimension (optional ?)
		self.hidden_size = hidden_size

		self.inp = nn.Linear(self.lag, hidden_size)
		self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
		self.out = nn.Linear(hidden_size, 1)

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

		outputs = torch.stack(outputs, dim=0)
		return outputs, hidden

	def params(self):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=0.001)
		sch = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
		return criterion, opt, sch

	# import torch
	# import torch.nn as nn
	# import numpy as np

	def format_batch(self, mat, ys):
		# in: batch x timelen x seqlen
		# out: seqlen x batch x timelen
		# also - sequence order is reversed, to infer traffic upstream
		steps = mat.shape[2]
		batch = []
		for si in range(steps):
			batch.append(torch.Tensor(mat[:, :, si]).cuda())
		batch = list(reversed(batch))

		ys = np.flip(ys, axis=1).copy()
		return batch, torch.from_numpy(ys).float().cuda()
