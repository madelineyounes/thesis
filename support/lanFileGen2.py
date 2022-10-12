import os
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import numpy as np

"""
Program that generates training and test csvs for the VoxLingua107 language data
"""

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
source_dir = os.path.join(ROOT_DIR, 'data')
output_path = "../data/"
data_label_path = "/srv/scratch/z5208494/dataset/VoxLingua107/"


output_path = "../data/"

test_l_out_file = output_path +"test_lan_50f.csv"
train_l_out_file = output_path+ "train_lan_50f.csv"

lan = ['ar', 'en', 'fr', 'it']

num_files = 50 

firstline = "id,label\n"
ftest = open(test_l_out_file, 'w')
ftest.write(firstline)

ftrain = open(train_l_out_file, 'w')
ftrain.write(firstline)

for l in lan:
    directory = data_label_path + l
    trcount = 0
    tscount = 0
    for filename in os.scandir(directory):
        if filename.is_file() and trcount < num_files:
            f = filename.name.rstrip(".wav")
            print(f)
            ftrain.write(f + f",{l}\n")
            trcount += 1
        elif filename.is_file() and tscount < num_files:
            f = filename.name.rstrip(".wav")
            ftest.write(f + f",{l}\n")
            tscount += 1
        else:
            break

ftrain.close()
ftest.close()
print("done")

