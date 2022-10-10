import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f1_path = ""
f2_path = ""
f3_path = ""
headers = ['Name', 'Age', 'Marks']

df = pd.read_csv('student.csv', names=headers)
x = np.arange(10)

plt.plot(x, x)
plt.plot(x, 2 * x)
plt.plot(x, 3 * x)
plt.plot(x, 4 * x)

plt.title("Amount of Training and Test Files")
plt.xlabel("X axis label")
plt.ylabel("Y axis label")

plt.legend(['y = x', 'y = 2x', 'y = 3x', 'y = 4x'], loc='upper left')

plt.show()
