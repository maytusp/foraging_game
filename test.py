import torch
# G = 5
# P = G * G
# cells = torch.arange(P, dtype=torch.long)
# cy = (cells // G).view(P,1)
# cx = (cells % G).view(P,1)
# neigh4 = torch.tensor([[-1,0],[1,0],[0,-1],[0,1]], dtype=torch.long) # (4,2)
# ny = cy + neigh4[:,0].view(1,4)
# ny2 = cy + neigh4[:,0]
# print(ny == ny2)
# print(neigh4[:,0].shape)
# # print(f"cx {cx}")

# n=1
# Fd = 10
# y = torch.zeros(n, Fd, dtype=torch.long) # (n, Fd)
# y[:, 1::2] = G - 1

# print(y)

# t = torch.tensor([[1, 2, 3, 4], 
#                  [5,6,7,8], 
#                  [9,10,11,12]])


# print(torch.gather(t, 1, torch.tensor([[0], 
#                                       [1], 
#                                       [0]])))

# print(torch.gather(t, 0, torch.tensor([[0, 0]])))
# B = 6
# Fd = 2
# alive = torch.tensor([[True, True], [True, False],  [False, True],  [False, False]])
# bidx = torch.arange(B).unsqueeze(1).expand(B, Fd)  # (B, Fd)
# print(bidx)
# # print(bidx[alive])
# print(bidx[[0,1,2,3,3,3,3,3,3], [1,0,1,0,0,0,0,0,0]])
A = 2
B = 1
K = 3
print(torch.arange(A).shape)
channel_idx = torch.arange(A)
print(channel_idx)
channel_idx = channel_idx.view(1,A,1,1,1)
print(channel_idx)
channel_idx = channel_idx.expand(B, -1, A, K, K)
print(channel_idx.shape)
print(channel_idx)