
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Conv(nn.Module):
	name = 'conv'
	def __init__(self, forecast=5, relu=False, deep=False):
		super(Conv, self).__init__()
		self.lag = 5    # temporal dimension
		self.steps = 10 # spatial dimension (optional ?)
		self.forecast = forecast

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
		self.dense = nn.Sequential(
			nn.Linear(128 * 2 * 5, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			# nn.Linear(256, self.steps),
			nn.Linear(256, self.forecast),
		)
		# self.conv_s = nn.Sequential(
		# 	nn.Conv1d(2, 64, 3, padding=1),
		# 	nn.ReLU(),
		# 	nn.Conv1d(64, 128, 3, padding=1),
		# 	nn.MaxPool1d(2),
		# 	nn.ReLU(),

		# 	nn.Conv1d(128, 256, 3, padding=1),
		# 	nn.ReLU(),
		# 	nn.Conv1d(256, 256, 3, padding=1),
		# 	nn.MaxPool1d(2),
		# 	nn.ReLU(),
		# )

	def forward(self, inputs, hidden=None):
		steps = len(inputs)

		t_out = []
		for ii in range(steps):
			out = self.conv_t(inputs[ii].unsqueeze(1))
			t_out.append(out.view(-1, 128 * 2))

		t_out = torch.stack(t_out, 2).view(-1, 2 * 5 * 128)

		s_out = self.dense(t_out)

		return s_out

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
			batch.append(torch.Tensor(mat[:, :, self.forecast+si]).to(gpu))

		batch = list(reversed(batch))
		ys = ys[:, :5] # forecasting earlier
		ys = np.flip(ys, axis=1).copy()
		# sequence order is reversed, since rnn is unrolled upstream

		ys = torch.from_numpy(ys).float().to(gpu)
		return batch, ys

if __name__ == '__main__':
	import os, sys
	sys.path.append('.')
	from dataset import Routes

	dset = Routes('train', 32, index_file='min-data.json')
	Xs, Ys = dset.next()

	model = Conv()
	Xs, Ys = model.format_batch(Xs, Ys)
	out = model(Xs)
	print(out.size())
	# model()
