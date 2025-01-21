import torch
# message = torch.tensor([0.6,0.1])
# message =  (message.gt(0.5).float() - 0.5).sign().float()
# print(message)
scale = 2 * 20
m = torch.tensor([[1.2,0.1,1.3], [2.3,-0.4,0.5]])
m = torch.softmax(m , 0)
print(m)