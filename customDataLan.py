import torch
import torchaudio
import pandas as pd
import pickle
from torch.utils.data import Dataset
import customTransform as T
from torchvision import transforms

label_list = ['ar', 'en', 'fr', 'it']
label2id, id2label = dict(), dict()
for i, label in enumerate(label_list):
    label2id[label] = str(i)
    id2label[str(i)] = label

def speech_file_to_array_fn(path, target_sampling_rate):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech
class CustomDataset(Dataset):
    def __init__(self, csv_fp, data_fp, labels, transform=None, sampling_rate=16000, model_name="facebook/wav2vec2-base", max_length=0.1):
        """
        Args:
        csv_fp (string): Path to csv with audio file ids and labels.
        data_fp (string): Path to audio files
        transform: Transform to be performed on audio file
        """
        if transform is None:
            self.transform = transforms.Compose([T.Extractor(model_name, sampling_rate, max_length)])
        else:
            self.transform = transform

        self.data_frame = pd.read_csv(csv_fp, delimiter=',')
        self.data_fp = data_fp
       
        self.sampling_rate = sampling_rate
        self.labels = labels

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audiopath = self.data_fp + \
            self.data_frame.iloc[idx, 1] + "\""+ \
            self.data_frame.iloc[idx, 0] + ".wav"
        speech = speech_file_to_array_fn(audiopath, self.sampling_rate)

        if self.transform:
            speech_features = self.transform(speech)[0]
            speech_mask = self.transform(speech)[1]

        label = int(label2id[self.data_frame.iloc[idx, 1]])

        speech_features = speech_features.float()
        speech_mask = speech_mask.long()
        item = {}
        item['input_values'] = speech_features
        item['attention_mask'] = speech_mask
        item['labels'] = torch.tensor(label)
        return item
