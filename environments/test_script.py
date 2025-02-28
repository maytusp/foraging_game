import numpy as np
import random
# score_unit = 2
# start_steps = 0
# last_steps = 125
# score_list = [(i+1)*score_unit for i in range(start_steps, last_steps) if (i+1) % 5 != 0]
# print(score_list)


s = np.random.binomial(1, 0.5, 1)
print(s)
min_xy = (0,0)
max_xy = (7,7)
pos = (random.randint(min_xy[0], max_xy[0]), random.randint(min_xy[1], max_xy[1]))