import os
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import numpy as np

"""
Program that takes in original datafiles and limits them to an amount of files per dialect
"""

output_path = "../data/"

val_input_file = output_path + "adi17_official_dev_label.txt"
test_input_file = output_path + "adi17_official_test_label.txt"
train_input_file = output_path + "imported_r_train_files.csv"


train_u_out_file = output_path + "u_train_u_25f.csv"

umbrella_dialects = ['NOR', 'EGY', 'GLF', 'LEV']

dialect_dict = {
    "EGY": ['EGY', 'SDN'],
    "GLF": ['IRQ', 'KWT', 'ARE', 'QAT', 'OMN', 'SAU', 'YEM'],
    "LEV": ['PSE', 'LBN', 'SYR', 'JOR'],
    "NOR": ['MRT', 'MAR', 'DZA', 'LBY']
}

train_lines = tuple(open(train_input_file, 'r'))

firstline = "id,label\n"

ftrain = open(train_u_out_file, 'w')
ftrain.write(firstline)

for d in umbrella_dialects:
    numreg = len(dialect_dict[d])
    numf_train = np.floor(25 / numreg)
    for rd in dialect_dict[d]:
        dcount = 0
        for line in train_lines:
            lim = numf_train
            if rd in line.rstrip("\n") and dcount < lim:
                    filename = line.split(',')[0]
                    ftrain.write(filename + f",{d}\n")
                    dcount += 1
ftrain.close()
print("done")

