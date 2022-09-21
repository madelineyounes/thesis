import numpy as np

var = [[-0.12421581, -0.04529671, -0.07717204,  0.186769],
       [-0.18687893, -0.05811683, -0.07163814,  0.14275934],
       [-0.19225629, -0.07723612, -0.05526555,  0.11532967],
       [-0.10553389,  0.0243341, -0.10081411,  0.10375907]]


y_pred = []
for predicts in var:
    p = predicts.index(np.amax(predicts))
    y_pred.append(p)
print(y_pred)