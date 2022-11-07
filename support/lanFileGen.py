import os
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import numpy as np

"""
Program that takes in original datafiles and limits them to an amount of files per dialect
"""

output_path = "../data/"

train_input_file = output_path + "adi17_official_dev_label.txt"
test_input_file = output_path +"adi17_official_test_label.txt"

test_u_out_file = output_path +"test_u_NOLEV.csv"
train_u_out_file = output_path+ "dev_u_NOLEV.csv"

test_r_out_file = output_path + "test_r_NOLEV.csv"
train_r_out_file = output_path +"dev_r_NOLEV.csv"

dialect_dict = {
    "EGY": ['EGY', 'SDN'],
    "GLF": ['IRQ', 'KWT', 'ARE', 'QAT', 'OMN', 'SAU', 'YEM'],
    "LEV": ['PSE', 'LBN', 'SYR', 'JOR'],
    "NOR": ['MRT', 'MAR', 'DZA', 'LBY']
}

umbrella_dialects = ['NOR', 'EGY', 'GLF', 'LEV']

num_files = 200 

train_lines = tuple(open(train_input_file, 'r'))
test_lines = tuple(open(test_input_file, 'r'))

firstline = "id,label\n"
ftest = open(test_u_out_file, 'w')
ftest.write(firstline)

ftrain = open(train_u_out_file, 'w')
ftrain.write(firstline)

frtest = open(test_r_out_file, 'w')
frtest.write(firstline)

frtrain = open(train_r_out_file, 'w')
frtrain.write(firstline)


for d in umbrella_dialects:
    numreg = len(dialect_dict[d])
    numf = np.floor(num_files / numreg)
    for rd in dialect_dict[d]:
        dcount = 0
        rdcount = 0 
        for line in train_lines:
            if rd in line.rstrip("\n") and rd not in dialect_dict["LEV"]:
                if  dcount <= numf:
                    filename = line.split(' ')[0]
                    ftrain.write(filename + f",{d}\n")
                    dcount += 1
                if rdcount <= num_files:
                    filename = line.split(' ')[0]
                    frtrain.write(filename + f",{rd}\n")
                    rdcount += 1
        dcount = 0
        rdcount = 0
        for line in test_lines:
            if rd in line.rstrip("\n")and rd not in dialect_dict["LEV"]:
                if dcount <= numf:
                    filename = line.split(' ')[0]
                    ftest.write(filename + f",{d}\n")
                    dcount += 1
                if rdcount <= num_files:
                    filename = line.split(' ')[0]
                    frtest.write(filename + f",{rd}\n")
                    rdcount += 1

ftrain.close()
ftest.close()
frtrain.close()
frtest.close()
print("done")

