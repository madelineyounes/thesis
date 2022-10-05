import shutil
import os
import string
import mutagen
import math
import fnmatch
from mutagen.wave import WAVE
from regex import D
from wavinfo import WavInfoReader

out_file_path = "/data/"
data_path = "/srv/scratch/z5208494/dataset/VoxLingua107/"

trainfile_path = out_file_path + "lang_data_train.csv"
testfile_path = out_file_path + "lang_data_test.csv"

trainfile = open(trainfile_path, 'w')
testfile = open(testfile_path, 'w')

lang = ['ar', 'en', 'fr', 'it']

def start_prompt():
    start_messag = '''
    This program generates a csv file with a list of file names to be used as training data 
    for an LID. 
    '''
    print(start_messag)
    split = float(input(
        "What perctage of the data do you want for training enter as percentage in decimal form eg. 0.30 for 30%?"))
    return split


split = start_prompt()
for l in lang:
    dir_path = data_path + lang + "/"
    count = len(fnmatch.filter(os.listdir(dir_path), '*.*'))

    numtrain = math.floor(count * split)
    numtest = math.floor(count * (1-split))

    




