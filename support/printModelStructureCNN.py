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

pretrain_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "elgeish/wav2vec2-large-xlsr-53-arabic")


pretrain_model.classifier = nn.Linear(
    in_features=256, out_features=num_labels, bias=True)

pretrain_model.classifier = nn.Sequential(
    nn.Conv2d(in_channels=256, out_channels=128,  kernel_size=(5, 5)),
    nn.AvgPool2d(kernel_size=(5, 5)),
    nn.Conv2d(in_channels =64, out_channels =10, kernel_size=(3, 3)),
    nn.AvgPool2d(kernel_size=(3, 3)),
    nn.Flatten(),
    nn.Linear(10,120),
    nn.Linear(120, 84),
    nn.Linear(84, 17),
    nn.Linear(in_features=17, out_features=num_labels, bias=False)
)

try:
    print(pretrain_model)
except:
    print("Cant print XLSR arabic")
