
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Linear(nn.Module):

	def __init__(self, forecast=5):
		super(Linear, self).__init__()

		self.lag = 6 # temporal dimension
		self.steps = 10 # spatial dimension (optional ?)
		self.forecast = forecast # spatial dimension (optional ?)

		self.op = nn.Linear(
			self.lag * (self.steps - 1),
			self.lag * (self.steps - 1))

	def forward(self, inputs, hidden=None):
		inputs = torch.cat(inputs, dim=1)

		outputs = self.op(inputs)

		outputs = outputs.view(outputs.size()[0], self.steps-1, self.lag)
		outputs = torch.transpose(outputs, 0, 1)
		return outputs

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
		return criterion, opt, sch

	def format_batch(self, data, wrap=True, normalize=1):
		# raw   : batch x timelen x seqlen
		data /= normalize
		data = torch.transpose(torch.transpose(data, 2, 1), 1, 0)
		# fmt   : seqlen x batch x timelen
		sequence = list(torch.split(data, 1, dim=0))

		for ti in range(len(sequence)):
			sequence[ti] = sequence[ti].to(self.device).float().squeeze(0)

		Xs = list(reversed(sequence[1:]))
		Ys = list(reversed(sequence[:-1])) # predict 1 stop back
		Ys = torch.stack(Ys, dim=0)

		return Xs, Ys
