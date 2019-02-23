
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Linear(nn.Module):

	def __init__(self, lag=6, forecast=5, spatial=False):
		super(Linear, self).__init__()

		self.lag = lag # temporal dimension
		self.steps = 10
		self.forecast = forecast
		self.spatial = spatial

		if spatial:
			self.op = nn.Linear(
				self.lag * (self.steps - 1),
				self.lag * (self.steps - 1))
		else:
			self.op = nn.Linear(
				self.lag * self.steps,
				self.steps)
				# self.lag * (self.steps - self.forecast),
				# self.forecast)

	def forward(self, inputs, hidden=None):
		# inputs = torch.cat(inputs, dim=1)

		outputs = self.op(inputs)

		if self.spatial:
			outputs = outputs.view(outputs.size()[0], self.steps-1, self.lag)
			outputs = torch.transpose(outputs, 0, 1)
		# else:

		return outputs

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
		return criterion, opt, sch

	def format_batch(self, data, wrap=True):
		if self.spatial:
			# raw   : batch x timelen x seqlen
			data = torch.transpose(torch.transpose(data, 2, 1), 1, 0)
			# fmt   : seqlen x batch x timelen
			sequence = list(torch.split(data, 1, dim=0))

			for ti in range(len(sequence)):
				sequence[ti] = sequence[ti].to(self.device).float().squeeze(0)

			Xs = list(reversed(sequence[1:]))
			Ys = list(reversed(sequence[:-1])) # predict 1 stop back
			Ys = torch.stack(Ys, dim=0)

			return Xs, Ys
		else:
			# raw   : batch x timelen x seqlen
			# Xs = data[:, :, self.forecast:]
			# Ys = data[:, -1, :self.forecast].to(self.device).float()
			Xs = data[:, :-1, :]
			Ys = data[:, -1, :].to(self.device).float()

			bytime = torch.split(Xs, 1, 1)
			bytime = list(map(lambda ent: ent.squeeze(1), bytime))
			Xs = torch.cat(bytime, 1).to(self.device).float()

			return Xs, Ys
