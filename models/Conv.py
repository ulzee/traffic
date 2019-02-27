
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
# sys.path.append('models/temporal')
# from models.RNN import RNN

class Conv(nn.Module):
	name = 'conv'
	def __init__(self, hidden_size=256, lag=5, stops=5):
		super(Conv, self).__init__()

		self.lag = lag
		self.hidden_size = hidden_size
		self.stops = stops

		self.conv_s = nn.Sequential(
			nn.Conv1d(1, 64, 3, padding=1),
			nn.ReLU(),
			nn.Conv1d(64, 128, 3, padding=1),
			nn.ReLU(),
			# nn.Conv1d(128, 256, 3, stride=2, padding=1),
			nn.Conv1d(128, 256, 3),
			nn.ReLU(),

			nn.Conv1d(256, 256, 3, padding=1),
			nn.ReLU(),
			nn.Conv1d(256, 256, 3),
			nn.ReLU(),

			# nn.Conv1d(128, 128, 3, padding=1),
			# nn.ReLU(),
			# nn.Conv1d(128, 128, 3, padding=1),
			# nn.ReLU(),
		)

		self.conv_t = nn.Sequential(
			nn.Conv1d(256, 256, 3, padding=1),
			nn.ReLU(),
			nn.Conv1d(256, 256, 3, padding=1),
			nn.ReLU(),
			nn.Conv1d(256, 128, 3, padding=1),
			nn.ReLU(),
			nn.Conv1d(128, 1, 3, padding=1),
			# nn.Conv1d(256, 256, 3, padding=1),
			# nn.ReLU(),
			# nn.Conv1d(256, 256, 3, padding=1),
			# nn.ReLU(),

			# nn.Conv1d(128, 128, 3, padding=1),
			# nn.ReLU(),
			# nn.Conv1d(128, 128, 3, padding=1),
			# nn.ReLU(),
		)

	def forward(self, inputs, hidden=None):
		# in: time x batch x seqlen
		# print(inputs.size())

		bytime = list(torch.split(inputs, 1, 1))

		sdim = []
		for si, svect  in enumerate(bytime):
			sout = self.conv_s(svect)
			sdim.append(sout)
		# print(len(sdim))
		# print(sdim[0].size())
		stv = torch.stack(sdim, dim=1)
		stv = torch.transpose(stv.squeeze(-1), 1, 2)
		# print(stv.size())

		tdim = self.conv_t(stv)
		# print(tdim.size())
		out = tdim.squeeze(1)
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

