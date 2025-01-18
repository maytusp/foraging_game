import torch
'''
Input: s_message with the size of (B, 1)
Output: r_message with the size of (B, 1)

Description:
s_message is in the form of [a1_message1, a2_message1, a1_message2, a2_message2, ...]
r_message is swap betweeen a1 and a2 in the way that
[a2_message1, a1_message1, a2_message1, a1_message1, ...]
'''
s_message = torch.Tensor([0.0,1.0,0.1,1.1,0.2,1.2,0.3,1.3,0.4,1.4]).unsqueeze(dim=1)
print(s_message.shape)
# Reshape s_message to (B/2, 2, 1) assuming B is even
# Note This is for only two agents.

s_message = s_message.view(-1, 2, 1)  # Each row contains [a1, a2]
print(s_message)
# Swap along dimension 1
r_message = s_message.flip(dims=[1])  # Flip [a1, a2] -> [a2, a1]
print(r_message)
# Reshape back to (B, 1)
r_message = r_message.view(-1, 1)
print(r_message)