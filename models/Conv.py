
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
# sys.path.append('models/temporal')
# from models.RNN import RNN

class Conv(nn.Module):
	name = 'conv'
	def __init__(self, hidden_size=256, forecast=5, lag=6):
		super(Conv, self).__init__()

		self.lag = lag
		self.hidden_size = hidden_size
		self.forecast = forecast
		self.steps = 10

		self.conv_t = nn.Sequential(
			nn.Conv1d(1, 64, 3, padding=1),
			nn.ReLU(),
			nn.Conv1d(64, 128, 3, padding=1),
			nn.MaxPool1d(2),
			nn.ReLU(),

			# nn.Conv1d(128, 128, 3, padding=1),
			# nn.ReLU(),
			# nn.Conv1d(128, 128, 3, padding=1),
			# nn.ReLU(),
		)
		# self.out = nn.Sequential(
		# 	nn.Linear(hidden_size, self.lag),
		# )

		# self.rnn = nn.LSTM(3 * 128, hidden_size, 1)
		# self.bsize = 32

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
		return criterion, opt, sch

	def format_batch(self, data):
		# raw   : batch x timelen x seqlen
		data = torch.transpose(data[:, :, self.forecast:], 1, 0)
		sequence = list(torch.split(data, 1, dim=0))

		for ti in range(len(sequence)):
			sequence[ti] = sequence[ti].to(self.device).float().squeeze(0)

		Xs = sequence
		Ys = data[-1, :, :(self.steps-self.forecast)].to(self.device).float()

		return Xs, Ys

	def forward(self, inputs, hidden=None):
		# in: time x batch x seqlen
		steps = len(inputs)
		outputs = []

		spatial = []
		for ii in range(steps):
			tconv = self.conv_s(inputs[ii].unsqueeze(1))
			spatial.append(tconv)

		print(spatial[0].size())
		assert False

		# temp = torch.stack(temp, 2)
		# spatial = self.conv_s(temp)
		# # print(spatial.size())
		# flat = spatial.squeeze(2).view(spatial.size()[0], -1)
		# # print(flat.size())
		# outputs = self.dense_out(flat)
		# outputs = outputs.view(-1, outputs.size()[0], self.lag)
		# # print(outputs.size())
		# # assert False

		# return outputs

		# steps = len(inputs)
		# outputs = []

		# for ii in range(steps):
		# 	output, hidden = self.step(inputs[ii], hidden)
		# 	outputs.append(output)

		# outputs = torch.stack(outputs, dim=0)

		# return outputs
