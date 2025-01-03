import torch
import torch.nn as nn

# rnn = nn.LSTM(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = rnn(input, (h0, c0))

# print(input.shape)
# print(output.shape)

embedding = nn.Embedding(10, 20)

input = torch.LongTensor([[1,2], [0,3], [5,5]])

output= embedding(input)
print(output.shape)