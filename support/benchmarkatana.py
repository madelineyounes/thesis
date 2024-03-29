import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier
language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

directory = '/srv/scratch/z5208494/dataset/test_segments/'
devFiles = tuple(open('data/adi17_official_test_label.txt', 'r'))

csvFile = open('speechbrainTestData.csv', 'w')
csvFile.write("file,dialect,prediction\n")
# iterate over files in
# that directory
correct = 0
total = 0
for line in devFiles:
    filename = line.split(' ')[0] + '.wav'
    dialect = line.split(' ')[1]
    f = os.path.join(directory, filename)
    print(f)
    # checking if it is a file
    if os.path.isfile(f):
        signal = language_id.load_audio(f)
        prediction = language_id.classify_batch(signal)
        csvFile.write(f"{filename},{dialect},{prediction[3]}\n")
        print(prediction[3])
        if (prediction[3] == ['ar: Arabic']):
            correct += 1
        total += 1
print(f"Accuracy {correct/total*100}%")
print(f"Out of {total} Files {correct} were identified as Arabic.")
