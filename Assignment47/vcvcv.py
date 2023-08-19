import numpy as np
a = np.random.rand(24)
b = np.random.rand(24)
c =  np.random.rand(1)
print(a@b + c)

a.reshape((-1,1))
# print(a)
