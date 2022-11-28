from torchvision import transforms
import customTransform as T
from torch.utils.data import Dataset
import pickle
import pandas as pd
import torchaudio
import torch
import numpy as np
import tensorflow as tf
from matplotlib.ticker import PercentFormatter
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_true = []
y_pred = []
label_list = ['NOR', 'EGY', 'GLF', 'LEV']
reg_label_list = ['EGY', 'SDN', 'IRQ', 'KWT', 'ARE', 'QAT', 'OMN', 'SAU',
              'YEM', 'PSE', 'LBN', 'SYR', 'JOR', 'MRT', 'MAR', 'DZA', 'LBY']

label2id, id2label = dict(), dict()
for i, label in enumerate(label_list):
    label2id[label] = str(i)
    id2label[str(i)] = label

experiment_id = "test"
def plot_data(x_label, y_label, matrix):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap=plt.cm.Blues)

    fig.colorbar(cax)
    xaxis = np.arange(len(x_label))
    yaxis = np.arange(len(y_label))
    ax.set_xticks(xaxis)
    ax.set_yticks(yaxis)
    ax.set_xticklabels(x_label, rotation=45)
    ax.tick_params(labelbottom=True, labeltop=False )
    ax.set_yticklabels(y_label)
    plt.savefig("output/"+experiment_id+".png")


preds = tf.constant([[-0.7204,  0.5,  0.5,  0.2023],
                        [-0.9115, -0.0212,  0.7294,  0.2992],
        [-0.8858, -0.0124,  0.7047,  0.2535],
    [-0.8116, -0.0200,  0.6766,  0.2349]])
labels = tf.constant([2, 2, 3, 2])

print (preds)
print(labels)

for p in preds:
    y_pred.append(np.argmax(p))
print(y_pred)
for l in labels.numpy(): 
    y_true.append(l)
print(label2id)


reg_matrix = np.matrix('''54 12  1  3 43  4 25  3  7  5 11 35  4  6  5  3 29; 
        53 19  6  3 21 17  1  0  2 50  4  8 18  7  8  7 26;
        1  2  6  1  0  2  1  1  9  7 15  1  7  0  0 10  8;
        14  1  1  0  5  8  1  0  0  2  0  0 11  2 10  7  9;
        4  1  3  0 16  4  5  0  0  3  2  5  0  0  1  6 21;
        5  2  1  1  6  4  4  0  1  3  2  2  0  1  3 14 22;
        4  4  1  0  9  2 14  1  0  1  3 11  0  2  6  4  9;
        4  4  0  0  1  3  9  1  5  1  5  0  2  5 11  2 18;
        3  2  0  0  5  2  3  4  9  6  4  4  2  1  6  8 12;
        6 11  0  0  6  4  7  3  4 18  9 10  3  3  9 12 20;
        12  0  1  4 11  4  0  0  2  5 31  2  6  0  9 15 23;
        10  2  0  4 25  1  5  0  1  4  6  8  4  5 11 15 24;
        17  4  0  1  2  2  0  1  1  5 10  2 27  4  3 10 36;
        6 10  1 11  5  3  3  2  1  9 19  3  4  5 10 23 10;
        2  3  0  1  3  6  0  0  0  0  0  4  2  0 90  8  6;
        4  0  1  0  4  5  1  1  0  7  0  3  4  0 43 31 21;
        1  2  1  0  4  8  4  1  3  1  6  5 18  0  6  5 60''')


c_matrix = confusion_matrix(y_true, y_pred, normalize='all')
print("CONFUSION MATRIX")
print(c_matrix)
print("CLASSIFICATION REPORT")
print(classification_report(y_true, y_pred))
plot_data(reg_label_list, reg_label_list, reg_matrix)
print("done")
