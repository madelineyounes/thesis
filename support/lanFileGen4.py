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

test_u_out_file = output_path + "test_u_x2EGY.csv"
val_u_out_file = output_path+ "dev_u_x2EGY.csv"
train_u_out_file = output_path + "train_u_x2EGY.csv"

umbrella_dialects = ['NOR', 'EGY', 'GLF', 'LEV']

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
    numf_test = np.floor(100 / numreg)
    numf_train = np.floor(700 / numreg)
    numf_val = np.floor(200 / numreg)


    for rd in dialect_dict[d]:
        dcount = 0
        for line in train_lines:
            lim = numf_train
            if rd in dialect_dict["EGY"] and rd in line.rstrip("\n"):
                lim = 2*numf_train
            if rd in line.rstrip("\n") and dcount < lim:
                    filename = line.split(',')[0]
                    ftrain.write(filename + f",{d}\n")
                    dcount += 1
        print (f"Train Files:{d} {dcount}")
        dcount = 0
        for line in test_lines:
            lim = numf_test
            if rd in dialect_dict["EGY"] and rd in line.rstrip("\n"):
                lim = 2*numf_test
            if rd in line.rstrip("\n") and dcount < lim:
                    filename = line.split(' ')[0]
                    ftest.write(filename + f",{d}\n")
                    dcount += 1
        print(f"Test Files:{d} {dcount}")

        dcount = 0
        for line in val_lines:
            if rd in dialect_dict["EGY"] and rd in line.rstrip("\n"):
                lim = 2*numf_val
            if rd in line.rstrip("\n") and dcount < lim:
                    filename = line.split(' ')[0]
                    fval.write(filename + f",{d}\n")
                    dcount += 1
        print(f"Val Files:{d} {dcount}")

ftrain.close()
ftest.close()
fval.close()
print("done")

