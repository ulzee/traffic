
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Template(nn.Module):
	def __init__(self):
		super(Template, self).__init__()

	# def to(self, device):
	# 	super(Template, self).to(device)
	# 	self.device = device
	# 	return self
