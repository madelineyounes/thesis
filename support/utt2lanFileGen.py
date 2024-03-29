import shutil
import os
import string
import mutagen
from mutagen.wave import WAVE
from regex import D
from wavinfo import WavInfoReader

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
source_dir = os.path.join(ROOT_DIR, 'data')
output_path = "../data/"
data_label_path = "/srv/scratch/z5208494/dataset/"
#data_label_path = "../data/"
#data_path = data_label_path+"dev_segments/"
data_path = "/srv/scratch/z5208494/dataset/test_segments/"
'''
Program to generate a csv file with all the file names to be used in trainging. 
The training data will be selected and using the 4 umbrella dialect categories (EGY, LEV, NOR, GLF) 
or the 17 specific regional dialects. The program will select enough data for a specified amount.
'''

regional_EGY_dialects = ['EGY', 'SDN']
regional_GLF_dialects = ['IRQ', 'KWT', 'ARE', 'QAT', 'OMN', 'SAU', 'YEM']
regional_LEV_dialects = ['PSE', 'LBN', 'SYR', 'JOR']
regional_NOR_dialects = ['MRT', 'MAR', 'DZA', 'LBY']

dialect_dict = {
    "EGY": ['EGY', 'SDN'],
    "GLF": ['IRQ', 'KWT', 'ARE', 'QAT', 'OMN', 'SAU', 'YEM'],
    "LEV": ['PSE', 'LBN', 'SYR', 'JOR'],
    "NOR": ['MRT', 'MAR', 'DZA', 'LBY']
}

regional_dialects = regional_NOR_dialects + regional_EGY_dialects + regional_GLF_dialects + regional_LEV_dialects
umbrella_dialects = ['NOR', 'EGY', 'GLF', 'LEV']

def start_prompt():
    start_messag = '''
    This program generates a csv file with a list of file names to be used as training data 
    for an arabic DID. The inputs are a selection of the dialect grouping system to be used (char)
    and the amount of data that needs to be collected for each of those dialects (float). 
    '''
    print(start_messag)
    dialect_group = input("Which dialect grouping do you want to use? The 4 umbrella dialects or the 17 regional dialects [u/r]?")
    total_time = float(input("How much data for each dialect in hrs?"))
    return dialect_group, total_time

def gen_csv(dialect_group, total_time):
    '''
    Function that generates a csv file which will contain a list of the files to be used as training data. 
    '''
    print("generating csv file...")
    counter = 0
    filename = output_path + "data_{d}_{t}_hrs_{count}_.csv".format(d=dialect_group, t=str(int(total_time)), count = counter)
    while os.path.isfile(filename.format(count=counter)):
        print(filename.format(count=counter))
        counter += 1
    filename = filename.format(count=counter)
    filename = output_path+"adi17_test_small.csv"
    f = open(filename, 'w')
    f.write("id,label\n")
    f.close()
    return filename

def populate_csv(file:string, dialect:string, total_time:float, dialect_group:string, time_counter: float):
    '''
    Function that takes in the csv file, the dialect abbrevation and 
    the total time of the training data. 
    '''
    print("populating doc ...")
    train_lines = tuple(
        open('../data/adi17_official_test_label.csv', 'r'))
    out_lines = tuple(open(file, 'r'))
    out_file = open(file, 'a+')
    info_time = 0
    for line in train_lines:
        if dialect in line.rstrip("\n") and time_counter < total_time:
            filename = line.split(' ')[0]
            filepath = data_path+line.split(' ')[0]
            try:
                info = WavInfoReader(filepath+".wav")
                if dialect_group == 'r':
                    out_file.write(line)
                else:
                    if dialect in regional_EGY_dialects:
                        out_file.write(filename + ",EGY\n")
                    elif dialect in regional_GLF_dialects:
                        out_file.write(filename + ",GLF\n")
                    elif dialect in regional_LEV_dialects:
                        out_file.write(filename + ",LEV\n")
                    elif dialect in regional_NOR_dialects:
                        out_file.write(filename + ",NOR\n")
                    else :
                        out_file.write(filename + ",NUL\n")
                # iterate time counter
                info_time = info.data.frame_count / info.fmt.sample_rate 
                time_counter += info_time
            except: 
                print("File not found: " + filepath)
    out_file.close()
    return time_counter

def main():
    dialect_group, total_time = start_prompt()
    total_time = 3600*total_time # convert hours to seconds
    csv_file = gen_csv(dialect_group, total_time/3600)
    print("Generated the file " + csv_file)

    if dialect_group == 'r':
        for dialect in regional_dialects:
            populate_csv(csv_file, dialect, total_time, dialect_group, 0)
    else:
        for dialect in umbrella_dialects: 
            time_counter = 0
            for d in regional_dialects: 
                if d in dialect_dict[dialect]:
                    time_counter += populate_csv(csv_file, d,
                                                 total_time, dialect_group, time_counter)

if __name__ == "__main__":
    main()
