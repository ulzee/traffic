
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Linear(nn.Module):

	def __init__(self, forecast=5):
		super(Linear, self).__init__()

		self.lag = 5 # temporal dimension
		self.steps = 10 # spatial dimension (optional ?)
		self.forecast = forecast # spatial dimension (optional ?)

		self.op = nn.Linear(
			self.lag * (self.steps - self.forecast),
			self.forecast)
			# self.lag + self.forecast)

	def forward(self, inputs, hidden=None):
		inputs = torch.cat(inputs, dim=1)

		outputs = self.op(inputs)

		return outputs

	def params(self, lr=0.001):
		criterion = nn.MSELoss().cuda()
		opt = optim.SGD(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
		return criterion, opt, sch

	def format_batch(self, mat, ys, gpu=None):
		# raw   : batch x timelen x seqlen
		# needed: seqlen x batch x timelen

		steps = mat.shape[2] - self.forecast
		# withold steps for forecasting

		batch = []
		for si in range(steps):
			batch.append(torch.Tensor(mat[:, :, steps+si]).to(gpu))

		batch = list(reversed(batch))
		ys = ys[:, :5]
		ys = np.flip(ys, axis=1).copy()
		# sequence order is reversed, to infer traffic upstream

		ys = torch.from_numpy(ys).float().to(gpu)
		return batch, ys
