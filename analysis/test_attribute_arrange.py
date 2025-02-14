import numpy as np
N_att = 8
attribute_list = np.arange(N_att)
selected_attributes = np.random.choice(attribute_list, size=N_att, replace=False)

mask_agent0 = np.zeros(N_att, dtype=int)
mask_agent1 = np.zeros(N_att, dtype=int)
mask_agent0[selected_attributes[:N_att//2]] = 1
mask_agent1[selected_attributes[N_att//2:]] = 1
attribute_mask = {0: mask_agent0, 1: mask_agent1}

print(attribute_list)
# print(attribute_mask)
# print(mask_agent0==mask_agent1)