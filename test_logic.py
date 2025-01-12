import numpy as np
num_steps = 16
num_envs = 8
envsperbatch = 2
update_epochs = 1
batch_size = num_steps * num_envs
# a = np.array(["e"+str(i)+"s"+str(j) for i in range(num_envs) for j in range(num_steps)])
a = np.array(["e"+str(i)+"s"+str(j) for j in range(num_steps) for i in range(num_envs)])
b = a.reshape(num_steps, num_envs)
print("a", a)
print("b", b.shape)
print(b)

c = b.reshape(-1)

flatinds = np.arange(batch_size).reshape(num_steps, num_envs)
print(f"flatinds {flatinds}")
clipfracs = []
envinds = np.arange(num_envs)
for epoch in range(0, update_epochs):
    # np.random.shuffle(envinds)
    for start in range(0, num_envs, envsperbatch):
        end = start + envsperbatch
        mbenvinds = envinds[start:end]
        mb_inds = flatinds[:, mbenvinds].ravel("F")  # be really careful about the index
        print(f"mb_inds {mb_inds}")

