import torch
import torch.nn as nn

conv31 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
input = torch.rand((8, 512, 26))
ouput = conv31(input)