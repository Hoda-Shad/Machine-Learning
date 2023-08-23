import numpy as np

y = [1 , 2 , 4 , 0 , 5 ]
y_pred = [2, 1 ,0,5,4]
y_pred = np.eye(np.max(y_pred)+1)[y_pred]
# print(y.shape)
# print(y_pred.shape)
print(np.sqrt(np.mean((y - y_pred)**2)))