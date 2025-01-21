import torch
message = torch.tensor([0.6,0.1])
message =  (message.gt(0.5).float() - 0.5).sign().float()
print(message)