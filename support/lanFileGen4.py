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

test_u_out_file = output_path +"test_u_NOLEV.csv"
val_u_out_file = output_path+ "dev_u_NOLEV.csv"
train_u_out_file = output_path + "train_u_NOLEV.csv"

umbrella_dialects = ['NOR', 'EGY', 'GLF']

dialect_dict = {
    "EGY": ['EGY', 'SDN'],
    "GLF": ['IRQ', 'KWT', 'ARE', 'QAT', 'OMN', 'SAU', 'YEM'],
    "LEV": ['PSE', 'LBN', 'SYR', 'JOR'],
    "NOR": ['MRT', 'MAR', 'DZA', 'LBY']
}

num_files = 500 

train_lines = tuple(open(train_input_file, 'r'))
test_lines = tuple(open(test_input_file, 'r'))
val_lines = tuple(open(val_input_file, 'r'))

firstline = "id,label\n"
ftest = open(test_u_out_file, 'w')
ftest.write(firstline)

ftrain = open(train_u_out_file, 'w')
ftrain.write(firstline)

fval = open(val_u_out_file, 'w')
fval.write(firstline)

for d in umbrella_dialects:
    numreg = len(dialect_dict[d])
    numf = np.floor(num_files / numreg)
    for rd in dialect_dict[d]:
        dcount = 0
        for line in train_lines:
            if rd in line.rstrip("\n") and dcount < numf:
                    filename = line.split(',')[0]
                    ftrain.write(filename + f",{d}\n")
                    dcount += 1
        dcount = 0
        for line in test_lines:
            if rd in line.rstrip("\n") and dcount < numf:
                    filename = line.split(' ')[0]
                    ftest.write(filename + f",{d}\n")
                    dcount += 1

        dcount = 0
        for line in test_lines:
            if rd in line.rstrip("\n") and dcount < numf:
                    filename = line.split(' ')[0]
                    fval.write(filename + f",{d}\n")
                    dcount += 1

ftrain.close()
ftest.close()
fval.close()
print("done")

