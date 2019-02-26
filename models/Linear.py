
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Linear(nn.Module):

	def __init__(self, lag=6, stops=1, forecast=5, spatial=False):
		super(Linear, self).__init__()

		self.lag = lag # temporal dimension
		self.forecast = forecast
		self.stops = stops
		self.spatial = spatial

		self.op = nn.Linear(
			self.lag,
			self.stops)

	def forward(self, inputs, hidden=None):
		outputs = self.op(inputs)

		return outputs

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
		return criterion, opt, sch

	def format_batch(self, data, wrap=True):
		# raw   : batch x timelen x seqlen
		known_t = self.lag
		Xs = data[:, :known_t, :self.stops].clone()
		Ys = data[:, known_t, :self.stops].to(self.device).float()

		Xs = Xs.view(-1, known_t * self.stops).to(self.device).float()

		return Xs, Ys
