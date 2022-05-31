import shutil
import os
import mutagen
from mutagen.wave import WAVE
from wavinfo import WavInfoReader

file = "/Users/myounes/Documents/Code/thesis/_FBO2f3kW5Q_009363-010989.wav"

info = WavInfoReader(file)
print(info.fmt.sample_rate, info.data.frame_count, info.fmt.bits_per_sample)
time = info.data.frame_count / info.fmt.sample_rate
print(time)

train_lines = tuple(open('train_label.txt', 'r'))
out_lines = tuple(open('txt_file.txt', 'r'))
out_file = open('txt_file.txt', 'a+')
dialect = 'ALG'
print(out_lines)
for line in train_lines:
    if dialect in line and line.rstrip() not in out_lines and line not in out_lines:
        filename = line.split(' ')[0]
        info = WavInfoReader(filename+".wav")
        out_file.write(line)
        # iterate time counter
        info_time = info.data.frame_count / info.fmt.sample_rate
        time_counter = time_counter + info_time
        break
out_file.close()

