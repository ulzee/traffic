
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
sys.path.append('models')
from models.RNN import RNN

class Conv(RNN):
	name = 'conv'
	def __init__(self, hidden_size=256, forecast=5):
		super(Conv, self).__init__(hidden_size=256, forecast=5, deep=True)

		self.conv_t = nn.Sequential(
			nn.Conv1d(1, 64, 3, padding=1),
			nn.ReLU(),
			nn.Conv1d(64, 128, 3, padding=1),
			nn.MaxPool1d(2),
			nn.ReLU(),

			nn.Conv1d(128, 128, 3, padding=1),
			nn.ReLU(),
			nn.Conv1d(128, 128, 3, padding=1),
			nn.ReLU(),
		)
		# self.conv_s = nn.Sequential(
		# 	nn.Conv2d(128, 256, 3, padding=1),
		# 	nn.ReLU(),
		# 	nn.Conv2d(256, 256, 3, padding=1),
		# 	nn.MaxPool2d(2),
		# 	nn.ReLU(),

		# 	nn.Conv2d(256, 256, 3, padding=1),
		# 	nn.ReLU(),
		# 	nn.Conv2d(256, 256, 3, padding=1),
		# 	nn.ReLU(),
		# )
		# self.dense_out = nn.Sequential(
		# 	nn.Linear(4 * 256, 512),
		# 	nn.ReLU(),
		# 	nn.Linear(512, 512),
		# 	nn.ReLU(),
		# 	nn.Linear(512, (self.steps - 1) * self.lag),
		# )
		self.out = nn.Sequential(
			nn.Linear(hidden_size, self.lag),
		)

		# self.rnn = nn.LSTM(3 * 128, hidden_size, 2, dropout=0.05)
		self.rnn = nn.LSTM(3 * 128, hidden_size, 1)
		self.bsize = 32

	def step(self, input, hidden=None):
		# seqlen = 1 for stepwise eval
		# in: batch x inputsize
		tconv = self.conv_t(input.unsqueeze(1))
		tconv = tconv.view(tconv.size()[0], -1)
		# input = self.inp(input)
		tconv = tconv.unsqueeze(0)
		# print(tconv.size())
		# assert False
		# in: seqlen x batch x inputsize
		output, hidden = self.rnn(tconv, hidden)
		# out: seqlen x batch x hiddensize

		output = self.out(output.squeeze(0))
		return output, hidden

	def forward(self, inputs, hidden=None):
		# steps = len(inputs)
		# outputs = []

		# temp = []
		# for ii in range(steps):
		# 	tconv = self.conv_t(inputs[ii].unsqueeze(1))
		# 	temp.append(tconv)

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

		steps = len(inputs)
		outputs = []

		for ii in range(steps):
			output, hidden = self.step(inputs[ii], hidden)
			outputs.append(output)

		outputs = torch.stack(outputs, dim=0)

		return outputs
