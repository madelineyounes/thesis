from datasets import load_dataset, load_metric, ClassLabel
import datasets
from matplotlib.pyplot import xcorr
import torchaudio

data_file = "../data/data_1file.csv"
out_data_file = "../data/newdata_1file.csv"
test_file = "../data/adi17_test_small.csv"
out_test_file = "../data/new_adi17_test_small.csv"

training_data_path = "/srv/scratch/z5208494/dataset/dev_segments/"
test_data_path = "/srv/scratch/z5208494/dataset/test_segments/"

lines = tuple(open(data_file, 'r'))
out_file = open(out_data_file, 'w')

for line in lines:
    if lines.index(line) == 0:
        filename = line.split(',')[0]
        filepath = training_data_path+filename+'wav'
        try:
            audio_array = torchaudio.load(filepath)
            out_file.write(line)
        except:
            print("removed ", filename)
    else:
        out_file.write(line)
out_file.close() 
lines = tuple(open(test_file, 'r'))
out_file = open(out_test_file, 'w')

for line in lines:
    if lines.index(line)== 0:
        filename = line.split(',')[0]
        filepath = training_data_path+filename+'wav'
        try:
            audio_array = torchaudio.load(filepath)
            out_file.write(line)
        except:
            print("removed ", filename)
    else:
        print(line)
        out_file.write(line)
<<<<<<< HEAD
out_file.close()
=======
out_file.close()
>>>>>>> d796adf3be586c6845fa4081c9d46d11efa4f642
