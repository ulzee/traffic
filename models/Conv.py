
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

		self.rnn = nn.LSTM(3 * 128, hidden_size, 2, dropout=0.05)
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
		steps = len(inputs)
		# lastKnown = steps - self.forecast - 1
		outputs = []

		for ii in range(steps):
			output, hidden = self.step(inputs[ii], hidden)
			outputs.append(output)

		outputs = torch.stack(outputs, dim=0)
		# return outputs, hidden
		return outputs
