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
    ax.set_xticklabels(x_label)
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

c_matrix = confusion_matrix(y_true, y_pred, normalize='all')
print("CONFUSION MATRIX")
print(c_matrix)
print("CLASSIFICATION REPORT")
print(classification_report(y_true, y_pred))
plot_data(label_list, label_list, c_matrix)
