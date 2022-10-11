import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f1_path = "/Users/myounes/Documents/Code/thesis/output/results_wav2vec-ADI17-50-files.csv"
f2_path = "/Users/myounes/Documents/Code/thesis/output/results_wav2vec-ADI17-100-files.csv"
headers = ['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss']

df1 = pd.read_csv(f1_path, names=headers)
df2 = pd.read_csv(f2_path, names=headers)
print(df1[['epoch', 'train_acc'][1::]])
x = np.arange(10)

axisrange = np.arange(0, 100, 1)
plt.figure()
x = df1['epoch']
y1 = df1['train_acc']
y2 = df1['val_acc']


plt.xticks(axisrange)
plt.yticks(axisrange)
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.plot(x, y1)

plt.title("Amount of Training and Test Files")
plt.legend(['50 files ', '100 files'], loc='upper left')
plt.show()
