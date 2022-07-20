import torch
import torchaudio
import pandas as pd
import pickle
from torch.utils.data import Dataset

label_list = ['NOR', 'EGY', 'GLF', 'LEV']
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
    def __init__(self, csv_fp, data_fp, transform=None, sampling_rate=16000):
        """
        Args:
        csv_fp (string): Path to csv with audio file ids and labels.
        data_fp (string): Path to audio files
        transform: Transform to be performed on audio file
        """
        self.data_frame = pd.read_csv(csv_fp, delimiter=',')
        self.data_fp = data_fp
        self.transform = transform
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audiopath = self.data_fp + self.data_frame.iloc[idx, 0] + ".wav"
        speech = speech_file_to_array_fn(audiopath, self.sampling_rate)
        
        # speech_features, mask = feature_extractor(
        #     speech, sampling_rate=target_sampling_rate)

        if self.transform:
            speech_features = self.transform(speech)[0]
            speech_mask = self.transform(speech)[1]

        label = int(label2id[self.data_frame.iloc[idx, 1]])

        speech_features = speech_features.float()
        speech_mask = speech_mask.long()

        sample = {"input_values": speech_features, "attention_mask": speech_mask, "label": label}
        print("SAMPLE: ",sample)
        return sample
