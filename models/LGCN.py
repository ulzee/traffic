
import os, sys
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import numpy as np

class LGCN(nn.Module):
	def __init__(self,
				g,
				n_hidden,
				n_classes,
				n_layers,
				activation,
				dropout,
				in_feats=1,
				single_rnn=False):
		super().__init__()

		self.g = g
		self.single_rnn = single_rnn
		self.layers = nn.ModuleList()
		# input layer
		self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
		# hidden layers
		for i in range(n_layers - 1):
			self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
		# output layer
		self.layers.append(GraphConv(n_hidden, n_classes))
		self.dropout = nn.Dropout(p=dropout)

		if single_rnn:
			self.rnn = nn.LSTM(
				# lstm options
				input_size=in_feats,
				hidden_size=n_hidden,
			)
		else:
			self.rnns = nn.ModuleList()
			for _ in range(g.number_of_nodes()):
				self.rnns.append(nn.LSTM(
					# lstm options
					input_size=in_feats,
					hidden_size=n_hidden,
				))

	def forward(self, blob):

		sequence = [tens for tens in torch.split(blob, 1, dim=1)]
		if len(sequence[0].size()) == 3 and sequence[0].size()[1] == 1:
			sequence = [step.squeeze(1) for step in sequence]
		predictions = []
		hidden_list = [None for _ in range(self.g.number_of_nodes())]
		hidden = None
		for si, step in enumerate(sequence):
			if self.single_rnn:
				out, hidden = self.rnn(step.unsqueeze(0), hidden)
				h = out.squeeze(0)
			else:
				inputs = [tens for tens in torch.split(step, 1, dim=-2)]
				bynode = []
				for ii, inp in enumerate(inputs):
					hidden = hidden_list[ii]
					out, nexth = self.rnns[ii](inp.unsqueeze(0), hidden)
					hidden_list[ii] = nexth
					bynode.append(out)
				stepped = torch.cat(bynode, dim=1).squeeze(0)
				h = stepped

			for i, layer in enumerate(self.layers):
				if i != 0:
					h = self.dropout(h)
				h = layer(h, self.g)
			convolved = h.squeeze(-1)

			predictions.append(convolved)

		results = torch.stack(predictions, dim=-1)

		return results

	def get_predictions(self, valset, history=24, limit=None):
		losses = []
		predictions = []

		for dayind in range(len(valset.data)):
			if limit is not None and dayind > limit:
				continue

			calendar = []
			day = valset.data[dayind]
			t0, tf = valset.trange[dayind]

			midnight = t0.replace(hour=0, minute=0, second=0, microsecond=0)
			seconds_since = (t0 - midnight).total_seconds()

			for hi in range(day.shape[0]):
				seconds_enc = (seconds_since + hi * 60 * 10) / (24 * 60 * 60)
				time_encoding = t0.weekday() / 6 + seconds_enc * 0.1
				calendar.append(time_encoding)

			timerange = range(history, len(day)-1)

			day_preds = []
			day_ls = []
			for ti in timerange:
				series = day[ti-history:ti].copy()
				mask = 1-np.isnan(day[ti])
				series[np.isnan(series)] = -1
				# times = calendar[ti-history:ti]
				# times = np.tile(np.expand_dims(times, 1), (1, series.shape[1]))
				# inp = np.stack([series, times])
				# inp = np.expand_dims(inp, 0).transpose((0, 3, 1, 2))

				inp = torch.from_numpy(series.T).float()
				with torch.no_grad():
					out = self(inp)
				out = out.detach().cpu().numpy().T[-1]
				day_preds.append(out)

				# print(type(mask), type(out))
				day_ls.append(np.linalg.norm(
					out[mask] - day[ti][mask]
				))
			predictions.append(np.array(day_preds))
			losses.append(np.array(day_ls))

			sys.stdout.write('\r[%d/%d]' % (
				dayind+1,
				len(valset.data)))
		sys.stdout.write('\n')
		sys.stdout.flush()

		return predictions, losses