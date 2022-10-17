import os
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import numpy as np

"""
Program that takes in original datafiles and limits them to an amount of files per dialect
"""

output_path = "../data/"

input_file = output_path + "adi17_offical_train.txt"

out_u_file = output_path +"imported_u_train_files.csv"
out_r_file = output_path + "imported_r_train_files.csv"

dialect_dict = {
    "EGY": ['EGY', 'SDN'],
    "GLF": ['IRQ', 'KWT', 'ARE', 'QAT', 'OMN', 'SAU', 'YEM'],
    "LEV": ['PSE', 'LBN', 'SYR', 'JOR'],
    "NOR": ['MRT', 'MAR', 'DZA', 'LBY']
}

umbrella_dialects = ['NOR', 'EGY', 'GLF', 'LEV']

num_files = 300 


lines = tuple(open(input_file, 'r'))

firstline = "id,label\n"
fUout = open(out_u_file, 'w')
fUout.write(firstline)

fRout = open(out_r_file, 'w')
fRout.write(firstline)

for d in umbrella_dialects:
    numreg = len(dialect_dict[d])
    numf = np.floor(num_files/numreg)
    for rd in dialect_dict[d]:
        dcount = 0
        rdcount = 0
        for line in lines:
            if rd in line.rstrip("\n"):
                if dcount < num_files:
                    filename = line.split(' ')[0]
                    fUout.write(filename + f",{d}\n")
                    dcount += 1
                if rdcount < num_files:
                    filename = line.split(' ')[0]
                    fRout.write(filename + f",{rd}\n")
                    rdcount += 1

fRout.close()
fUout.close()
print("done")

