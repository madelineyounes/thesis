import soundfile as sf

file = '../data/data_1file.csv'
datapath = "/srv/scratch/z5208494/dataset/dev_segments/"
outfile = '../data/data_1file2.csv'
lines = tuple(open(file, 'r'))
out_file = open(outfile, 'w+')

out_file.write('id,label,audio,sampling_rate')
for line in lines:
    filename = line.split(',')[0]
    filepath = datapath+filename+'.wav'
    try:
        audio_array, sampling_rate = sf.read(filepath)
        out_file.write('{name},{d},{a},{sr}\n'.format(
            name=filename, d=line.split(',')[1], a=audio_array, sr=sampling_rate))
    except:
        print('could not read file')
 
out_file.close()
