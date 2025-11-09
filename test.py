import numpy as np

a = np.ones((2,2,3))

print(a[:, :, 0])
print(a[:, :, 1])
print(a[:, :, 2])


print('--------------')

padded = np.pad(a, ((1,1),(1,1),(0,0)), mode='constant')
print(padded.shape)

print(padded[:, :, 0])
print(padded[:, :, 1])
print(padded[:, :, 2])

print('--------------')
print(padded.strides)