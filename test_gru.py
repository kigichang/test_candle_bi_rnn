import torch
import torch.nn as nn

rnn = nn.GRU(10, 20, 1, batch_first=True)
input = torch.randn(5, 3, 10)
output, hn = rnn(input)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output.contiguous()
state_dict['hn'] = hn
torch.save(state_dict, "gru_test.pt")