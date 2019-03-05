
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
# sys.path.append('models/temporal')
# from models.RNN import RNN

class Conv(nn.Module):
	name = 'conv'
	def __init__(self, hidden_size=256, lag=6, stops=5):
		super(Conv, self).__init__()

		self.lag = lag
		self.hidden_size = hidden_size
		self.stops = stops

		self.conv_t = nn.Sequential(
			nn.Conv1d(1, 256, 3, padding=1),
			nn.ReLU(),
			# nn.Conv1d(64, 128, 3, padding=1),
			# nn.ReLU(),
			# nn.Conv1d(128, 256, 3, padding=1),
			# nn.ReLU(),
			# nn.Conv1d(256, 256, 3, padding=1),
			# nn.ReLU(),
			nn.MaxPool1d(2),

			nn.Conv1d(256, 256, 3, padding=1),
			nn.ReLU(),
			nn.Conv1d(256, 256, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(2, stride=1),
			# nn.Conv1d(256, 256, 3),
			nn.ReLU(),
			# nn.Conv1d(256, 1, 1),
			# nn.ReLU(),
			# nn.Conv1d(256, 256, 3),
			# nn.ReLU(),
		)

		self.dense = nn.Sequential(
			nn.Linear(2 * 256, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, self.stops),
		)

	def forward(self, inputs, hidden=None):
		# in: time x batch x seqlen

		# bytime = list(torch.split(inputs, 1, 1))
		bystop = list(torch.split(inputs, 1, 2))

		out = self.conv_t(torch.transpose(bystop[0], 1, 2))
		out = self.dense(out.view(-1, 2 * 256))
		# out = out.squeeze(2)
		return out
		# sdim = []
		# for si, svect  in enumerate(bytime):
		# 	sout = self.conv_s(svect)
		# 	sdim.append(sout)
		# out: convolutions among neighbors

		# stv = torch.cat(sdim, dim=1)
		# stv = torch.transpose(stv.squeeze(-1), 1, 2)
		# print(stv.size())

		# tdim = self.conv_t(stv)
		# print(tdim.size())
		# out = tdim.squeeze(1)
		# out = self.dense(stv.view(-1, 256))
		# out = self.dense(tdim.squeeze(2))
		# print(out.size())
		# assert False
		return out

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
		return criterion, opt, sch

	def format_batch(self, data, wrap=True):
		# raw   : batch x timelen x seqlen
		known_t = self.lag
		Xs = data[:, :known_t, :self.stops].clone()
		Ys = data[:, known_t, :self.stops].to(self.device).float()

		Xs = Xs.view(-1, known_t, self.stops).to(self.device).float()

		return Xs, Ys

