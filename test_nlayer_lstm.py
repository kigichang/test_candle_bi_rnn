import torch
import torch.nn as nn

rnn = nn.LSTM(10, 20, num_layers=3, batch_first=True)
input = torch.randn(5, 3, 10)
output, (hn, cn) = rnn(input)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output.contiguous()
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "lstm_nlayer_test.pt")

