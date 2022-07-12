from datasets import load_dataset, load_metric, ClassLabel
import datasets
from matplotlib.pyplot import xcorr
import torchaudio

data_file = "../data/data_1file.csv"
test_file = "../data/adi17_test_small.csv"

training_data_path = "/srv/scratch/z5208494/dataset/dev_segments/"
test_data_path = "/srv/scratch/z5208494/dataset/test_segments/"

lines = tuple(open(data_file, 'r'))
out_file = open(data_file, 'a+')

for line in lines:
    if line != ("id,label"):
        filename = line.split(' ')[0]
        filepath = training_data_path+filename
        try:
            audio_array = torchaudio.load(filepath)
            out_file.write(line)
        except:
            print("removed ", filename)
    else:
        out_file.write(line)
out_file.close() 

lines = tuple(open(test_file, 'r'))
out_file = open(test_file, 'a+')

for line in lines:
    if line != ("id,label"):
        filename = line.split(' ')[0]
        filepath = training_data_path+filename
        try:
            audio_array = torchaudio.load(filepath)
            out_file.write(line)
        except:
            print("removed ", filename)
    else:
        out_file.write(line)
out_file.close()
