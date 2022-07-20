import torch
import torchaudio
import pandas as pd
import pickle
from torch.utils.data import Dataset


def speech_file_to_array_fn(path, target_sampling_rate):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

class CustomDataset(Dataset):
    def __init__(self, csv_fp, data_fp, transform=None, target_transform=None, sampling_rate=16000):
        """
        Args:
        csv_fp (string): Path to csv with audio file ids and labels.
        data_fp (string): Path to audio files
        """
        self.data_frame = pd.read_csv(csv_fp, delimiter=',')
        self.data_fp = data_fp

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audiopath = self.data_fp + self.data_frame.iloc[idx, 0] + ".wav"
        speech = speech_file_to_array_fn(audiopath, self.sampling_rate)
        speech_features, mask = feature_extractor(
            speech, sampling_rate=target_sampling_rate)

        label = int(label2id[self.data_frame.iloc[idx, 1]])
        sample = {"input_values": speech_features, "label": label}
        return sample
