from transformers import (
    Trainer,
    TrainingArguments,
    AutoConfig,
    Wav2Vec2ForSequenceClassification
)
import torch
import librosa
from datasets import load_dataset
import torch.nn as nn
num_labels = 4

print("XLSR ARABIC")
model_path = "/srv/scratch/z5208494/ADI17-xlsr-arabic/pytorch_model.bin"
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "elgeish/wav2vec2-large-xlsr-53-arabic")


class GetLSTMOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out

model.classifier = nn.Sequential(
    nn.LSTM(256, 2, 1, batch_first=False, bidirectional=True),
    GetLSTMOutput(),
    nn.Dropout(p=0.5),
    nn.Softmax(dim=1),
    nn.Linear(14, num_labels, bias=True)
)

#model.load_state_dict(torch.load(model_path), strict=False)

try:
    print(model)
except:
    print("Cant print XLSR arabic")
