#----------------------------------------------------------
# run_umbrellaDID.py
# Purpose: Uses xlsr to create a audio classifer that 
# identifies arabic dialects using the ADI17 dataset. 
# Based on source:
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb#scrollTo=pdcMxVGEA9Cd 
# & Code by: Renee Lu
# Author: Madeline Younes, 2022
#----------------------------------------------------------

# ------------------------------------------
#      Install packages if needed
# ------------------------------------------
#pip3 install datasets==1.8.0
#pip3 install transformers
#pip3 install torchaudio
#pip3 install pyarrow
#pip3 install numpy
#pip3 install random
#pip3 install dataclasses
#pip3 install torchvision
from transformers.optimization import Adafactor, AdafactorSchedule
import numpy as np
import pandas as pd
import random
import torch
from dataclasses import dataclass
from datasets import load_dataset, load_metric
from typing import Any, Dict, List, Optional, Union
import pyarrow.csv as csv
import pyarrow as pa
from sklearn.metrics import classification_report
from typing import Optional, Tuple, Any, Dict, Union
import customTransform as T
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchvision import transforms
import customTransform as T
from customData import CustomDataset
from transformers.file_utils import ModelOutput
from transformers import (
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor, 
    Wav2Vec2Processor, 
    Wav2Vec2FeatureExtractor, 
    AutoConfig,
    Wav2Vec2ForCTC, 
    Wav2Vec2CTCTokenizer
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from datetime import datetime
import os
print(
    "------------------------------------------------------------------------")
print("                         run_umbrellaDID.py                            ")
print("------------------------------------------------------------------------")
# ------------------------------------------
#       Import required packages
# ------------------------------------------
# For printing filepath
# ------------------------------------------
print('Running: ', os.path.abspath(__file__))
# ------------------------------------------
# For accessing date and time
now = datetime.now()
# Print out dd/mm/YY H:M:S
# ------------------------------------------
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Started:", dt_string)
# ------------------------------------------
print("\n------> IMPORTING PACKAGES.... ---------------------------------------\n")
# Import datasets and evaluation metric
# Convert pandas dataframe to DatasetDict
print("-->Importing datasets...")

# Generate random numbers
print("-->Importing random...")
# Manipulate dataframes and numbers
print("-->Importing pandas & numpy...")
# Read, Write, Open json files
print("-->Importing json...")
# Use models and tokenizers
print("-->Importing Wav2Vec transformers...")
# Loading audio files
print("-->Importing torchaudio...")
# For training
print("-->Importing torch, dataclasses & typing...")
print("-->Importing from transformers for training...")
print("-->Importing pyarrow for loading dataset...")
print("-->SUCCESS! All packages imported.")

# ------------------------------------------
#      Setting experiment arguments
# ------------------------------------------
print("\n------> EXPERIMENT ARGUMENTS ----------------------------------------- \n")

# Perform Training (True/False)
# If false, this will go straight to model evaluation
training = True
print("training:", training)

# Experiment ID
# For 
#     1) naming model output directory
#     2) naming results file
experiment_id = "xlsr-ADI17-initialtest/"
print("experiment_id:", experiment_id)

# DatasetDict Id
# For 1) naming cache directory and
#     2) saving the DatasetDict object
datasetdict_id = "myST-eval"
print("datasetdict_id:", datasetdict_id)

data_path = "/srv/scratch/z5208494/dataset/"
print("data path:", data_path)

training_data_path = "/srv/scratch/z5208494/dataset/dev_segments/"
print("training data path:", training_data_path)

test_data_path = "/srv/scratch/z5208494/dataset/test_segments/"
print("test data path:", test_data_path)
# Base filepath
# For setting the base filepath to direct output to
base_fp = "/srv/scratch/z5208494/output/"
print("base_fp:", base_fp)

# Base cache directory filepath
# For setting directory for cache files
base_cache_fp = "/srv/scratch/z5208494/cache/huggingface/datasets/"

# Training dataset name and filename
# Dataset name and filename of the csv file containing the training data
# For generating filepath to file location
train_name = "umbrella_alldevdata"
train_filename = "data_1file"
print("train_name:", train_name)
print("train_filename:", train_filename)

# Evaluation dataset name and filename
# Dataset name and filename of the csv file containing the evaluation data
# For generating filepath to file location
evaluation_filename = "adi17_test_small"
print("evaluation_filename:", evaluation_filename)

# Resume training from/ use checkpoint (True/False)
# Set to True for:
# 1) resuming from a saved checkpoint if training stopped midway through
# 2) for using an existing finetuned model for evaluation
# If 2), then must also set eval_pretrained = True
use_checkpoint = False
print("use_checkpoint:", use_checkpoint)
# Set checkpoint if resuming from/using checkpoint
checkpoint = "/srv/scratch/z5208494/checkpoint/20211018-base-intial-test"
if use_checkpoint:
    print("checkpoint:", checkpoint)

# Use pretrained model
model_name = "facebook/wav2vec2-base"
# try log0/wav2vec2-base-lang-id

# Use a pretrained tokenizer (True/False)
#     True: Use existing tokenizer (if custom dataset has same vocab)
#     False: Use custom tokenizer (if custom dataset has different vocab)
use_pretrained_tokenizer = True
print("use_pretrained_tokenizer:", use_pretrained_tokenizer)
# Set tokenizer
pretrained_tokenizer = model_name
if use_pretrained_tokenizer:
    print("pretrained_tokenizer:", pretrained_tokenizer)

# Evaluate existing model instead of newly trained model (True/False)
#     True: use the model in the filepath set by 'eval_model' for eval
#     False: use the model trained from this script for eval
eval_pretrained = False
print("eval_pretrained:", eval_pretrained)
# Set existing model to evaluate, if evaluating on existing model
eval_model = checkpoint
if eval_pretrained:
    print("eval_model:", eval_model)

print("\n------> MODEL ARGUMENTS... -------------------------------------------\n")
# For setting model = Wav2Vec2ForCTC.from_pretrained()

set_hidden_dropout = 0.1                    # Default = 0.1
print("hidden_dropout:", set_hidden_dropout)
set_activation_dropout = 0.1                # Default = 0.1
print("activation_dropout:", set_activation_dropout)
set_attention_dropout = 0.1                 # Default = 0.1
print("attention_dropoutput:", set_attention_dropout)
set_feat_proj_dropout = 0.0                 # Default = 0.1
print("feat_proj_dropout:", set_feat_proj_dropout)
set_layerdrop = 0.05                        # Default = 0.1
print("layerdrop:", set_layerdrop)
set_mask_time_prob = 0.065                  # Default = 0.05
print("mask_time_prob:", set_mask_time_prob)
set_mask_time_length = 10                   # Default = 10
print("mask_time_length:", set_mask_time_length)
set_ctc_loss_reduction = "mean"             # Default = "sum"
print("ctc_loss_reduction:", set_ctc_loss_reduction)
set_ctc_zero_infinity = False               # Default = False
print("ctc_zero_infinity:", set_ctc_zero_infinity)
set_gradient_checkpointing = False           # Default = False
print("gradient_checkpointing:", set_gradient_checkpointing)
set_pooling_mode = "mean"
print("pooling_mode:", set_pooling_mode)

print("\n------> TRAINING ARGUMENTS... ----------------------------------------\n")
# For setting training_args = TrainingArguments()

set_evaluation_strategy = "epoch"           # Default = "no"
print("evaluation strategy:", set_evaluation_strategy)
set_per_device_train_batch_size = 4         # Default = 8
print("per_device_train_batch_size:", set_per_device_train_batch_size)
set_gradient_accumulation_steps = 2         # Default = 4
print("gradient_accumulation_steps:", set_gradient_accumulation_steps)
set_learning_rate = 0.00004                 # Default = 0.00005
print("learning_rate:", set_learning_rate)
set_weight_decay = 0.01                     # Default = 0
print("weight_decay:", set_weight_decay)
set_adam_beta1 = 0.9                        # Default = 0.9
print("adam_beta1:", set_adam_beta1)
set_adam_beta2 = 0.98                       # Default = 0.999
print("adam_beta2:", set_adam_beta2)
set_adam_epsilon = 0.00000001               # Default = 0.00000001
print("adam_epsilon:", set_adam_epsilon)
set_num_train_epochs = 5                   # Default = 3.0
print("num_train_epochs:", set_num_train_epochs)
set_max_steps = 35000                       # Default = -1, overrides epochs
print("max_steps:", set_max_steps)
set_lr_scheduler_type = "linear"            # Default = "linear"
print("lr_scheduler_type:", set_lr_scheduler_type)
set_warmup_ratio = 0.1                      # Default = 0.0
print("warmup_ratio:", set_warmup_ratio)
set_logging_strategy = "steps"              # Default = "steps"
print("logging_strategy:", set_logging_strategy)
set_logging_steps = 10                      # Default = 500
print("logging_steps:", set_logging_steps)
set_save_strategy = "epoch"                 # Default = "steps"
print("save_strategy:", set_save_strategy)
set_save_steps = 1000                         # Default = 500
print("save_steps:", set_save_steps)
set_save_total_limit = 40                   # Optional
print("save_total_limit:", set_save_total_limit)
set_fp16 = False                             # Default = False
print("fp16:", set_fp16)
set_eval_steps = 1000                         # Optional
print("eval_steps:", set_eval_steps)
set_load_best_model_at_end = True           # Default = False
print("load_best_model_at_end:", set_load_best_model_at_end)
set_metric_for_best_model = "accuracy"           # Optional
print("metric_for_best_model:", set_metric_for_best_model)
set_greater_is_better = False               # Optional
print("greater_is_better:", set_greater_is_better)
set_group_by_length = True                  # Default = False
print("group_by_length:", set_group_by_length)
set_push_to_hub = False                      # Default = False
print("push_to_hub:", set_push_to_hub)

# ------------------------------------------
#        Generating file paths
# ------------------------------------------
print("\n------> GENERATING FILEPATHS... --------------------------------------\n")
# Path to dataframe csv for train dataset
data_base_fp = "data/"
data_train_fp = data_base_fp + train_filename + ".csv"
print("--> data_train_fp:", data_train_fp)
# Path to dataframe csv for test dataset
data_test_fp = data_base_fp + evaluation_filename + ".csv"
print("--> data_test_fp:", data_test_fp)

# Dataframe file
# |-----------|---------------------|----------|---------|
# | file path | transcription_clean | duration | spkr_id |
# |-----------|---------------------|----------|---------|
# |   ...     |      ...            |  ..secs  | ......  |
# |-----------|---------------------|----------|---------|
# NOTE: The spkr_id column may need to be removed beforehand if
#       there appears to be a mixture between numerical and string ID's
#       due to this issue: https://github.com/apache/arrow/issues/4168
#       when calling load_dataset()

# Path to datasets cache
data_cache_fp = base_cache_fp + datasetdict_id
print("--> data_cache_fp:", data_cache_fp)
# Path to save model output
model_fp = "../output/" + train_name + "_local/" + experiment_id
print("--> model_fp:", model_fp)
# Path to save results output
finetuned_results_fp = base_fp + train_name + \
    "_local/" + experiment_id + "_finetuned_results.csv"
print("--> finetuned_results_fp:", finetuned_results_fp)
# Pre-trained checkpoint model
# For 1) Fine-tuning or
#     2) resuming training from pre-trained model
# If 1) must set use_checkpoint = False
# If 2)must set use_checkpoint = True
# Default model to fine-tune is facebook's model
pretrained_mod = model_name
if use_checkpoint:
    pretrained_mod = checkpoint
print("--> pretrained_mod:", pretrained_mod)
# Path to pre-trained tokenizer
# If use_pretrained_tokenizer = True
if use_pretrained_tokenizer:
    print("--> pretrained_tokenizer:", pretrained_tokenizer)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_tokenizer)

# ------------------------------------------
#         Preparing dataset
# ------------------------------------------
# Run the following scripts to prepare data
# 1) Prepare data from kaldi file:
# /srv/scratch/z5160268/2020_TasteofResearch/kaldi/egs/renee_thesis/s5/wav2vec_exp/data_prep.py
# 3) [Optional] Limit the files to certain duration:
# /srv/scratch/z5160268/2020_TasteofResearch/kaldi/egs/renee_thesis/s5/wav2vec_projects/data_getShortWavs.py
# 2) Split data into train and test:
# /srv/scratch/z5160268/2020_TasteofResearch/kaldi/egs/renee_thesis/s5/wav2vec_projects/data_split.py

print("\n------> PREPARING DATASET... ------------------------------------\n")
# Read the existing csv saved dataframes and
# load as a DatasetDict
data = load_dataset('csv',
                    data_files={'train': data_train_fp,
                                'test': data_test_fp},
                    delimiter=",",
                    cache_dir=data_cache_fp)

label_list = ['NOR', 'EGY', 'GLF', 'LEV']
label2id, id2label = dict(), dict()
for i, label in enumerate(label_list):
    label2id[label] = str(i)
    id2label[str(i)] = label

print("--> dataset...")
print(data)
# Display some random samples of the dataset
print("--> Printing some random samples...")


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset), "Picking more elements than in dataset"
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    df = pd.DataFrame(dataset[picks])
    print(df)

show_random_elements(data["train"], num_examples=5)
show_random_elements(data["test"], num_examples=5)
print("SUCCESS: Prepared dataset.")
# ------------------------------------------
#       Processing transcription
# ------------------------------------------
# Extracting all distinct letters of train and test set

# ------------------------------------------
#    Create Wav2Vec2 Feature Extractor
# ------------------------------------------
print("\n------> CREATING WAV2VEC2 FEATURE EXTRACTOR... -----------------------\n")
# Instantiate a Wav2Vec2 feature extractor:
# - feature_size: set to 1 because model was trained on raw speech signal
# - sampling_rate: sampling rate the model is trained on
# - padding_value: for batched inference, shorter inputs are padded
# - do_normalize: whether input should be zero-mean-unit-variance
#   normalised or not. Usually, speech models perform better when true.
# - return_attention_mask: set to false for Wav2Vec2, but true for
#   fine-tuning large-lv60
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
#feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
# Feature extractor and tokenizer wrapped into a single
# Wav2Vec2Processor class so we only need a model and processor object
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
# Save to re-use the just created processor and the fine-tuned model
processor.save_pretrained(model_fp)
print("SUCCESS: Created feature extractor.")

# ------------------------------------------
#             Pre-process Data
# ------------------------------------------
print("\n------> PRE-PROCESSING DATA... ----------------------------------------- \n")

target_sampling_rate = feature_extractor.sampling_rate
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(
        sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

# Audio files are stored as .wav format
# We want to store both audio values and sampling rate
# in the dataset.
# We write a map(...) function accordingly.
max_duration = 0.10 
print ("Max Duration:",  max_duration, "s")
sampling_rate = feature_extractor.sampling_rate
print ("Sampling Rate:",  sampling_rate)
def audio_to_array_fn(batch):
    try:
        filepath = training_data_path + batch["id"] + ".wav"
        audio_array = speech_file_to_array_fn(filepath)
        inputs = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
        )
        inputs["labels"] = int(label2id[batch["label"]])
        return inputs
    except:
        try:
            filepath = test_data_path + batch["id"] + ".wav"
            audio_array = speech_file_to_array_fn(filepath)
            inputs = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
        )
            inputs["labels"] = int(label2id[batch["label"]])
            return inputs
        except: 
            pass

def preprocess_function(batch):
    speech_list = []
    target_list = []
    for i in range(0, len(batch["id"])):
        try:
            filepath = training_data_path + batch["id"][i] + ".wav"
            speech = speech_file_to_array_fn(filepath)
            speech_list.append(speech)
            target_list.append(int(label2id[batch["label"][i]]))
        except: 
            try:
                filepath = test_data_path + batch["id"][i] + ".wav"
                speech = speech_file_to_array_fn(filepath)
                speech_list.append(speech)
                target_list.append(int(label2id[batch["label"][i]]))
            except: 
                print("speech except")

    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)
    return result


training_data = data["train"]
test_data = data["test"]
encoded_data = data.map(audio_to_array_fn, remove_columns=["id"], num_proc=4)
en_training_data = training_data.map(preprocess_function, batched=True, batch_size=set_per_device_train_batch_size)
en_test_data = test_data.map(
    preprocess_function, batched=True, batch_size=set_per_device_train_batch_size)
#print(encoded_data)
#rint(en_training_data)
#print(en_test_data)
# Check a few rows of data to verify data properly loaded
print("--> Verifying data with a random sample...")

print(len(encoded_data["train"]))
if (len(encoded_data["train"]) > 0):
    rand_int = random.randint(0, len(encoded_data["train"])-1)
    print(rand_int)
    print("Dialect Label:", encoded_data["train"][rand_int]["label"])
    print("Input array shape:", np.asarray(
        encoded_data["train"][rand_int]["input_values"]).shape)

print(len(encoded_data["test"]))
if (len(encoded_data["test"]) > 0):
    rand_int = random.randint(0, len(encoded_data["train"])-1)
    print(rand_int)
    print("Dialect Label:", encoded_data["test"][rand_int]["label"])
    print("Input array shape:", np.asarray(
        encoded_data["test"][rand_int]["input_values"]).shape)

    idx = 0
    print(encoded_data["test"][idx]['labels'])
    print("Training labels", encoded_data["test"][idx]['labels'],encoded_data["test"][idx]['label'])


# create custom dataset class
print("Create a custom dataset ---> ")
# class CustomDataset(Dataset):
#     def __init__(self, csv_fp, data_fp):
#         """
#         Args:
#         csv_fp (string): Path to csv with audio file ids and labels.
#         data_fp (string): Path to audio files
#         """
#         self.data_frame = pd.read_csv(csv_fp, delimiter=',')
#         self.data_fp = data_fp

#     def __len__(self):
#         return len(self.data_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         audiopath = self.data_fp + self.data_frame.iloc[idx,0] + ".wav"
#         speech = speech_file_to_array_fn(audiopath)
#         speech_features, mask = feature_extractor(speech, sampling_rate=target_sampling_rate)

#         label = int(label2id[self.data_frame.iloc[idx, 1]])
#         sample = {"input_values": speech_features, "label": label}
#         return sample


random_transforms = transforms.Compose([T.Extractor(model_name, sampling_rate, 0.1)])

traincustomdata = CustomDataset(csv_fp=data_train_fp, data_fp=training_data_path,transform=random_transforms)
testcustomdata = CustomDataset(
    csv_fp=data_test_fp, data_fp=test_data_path, transform=random_transforms)

trainDataLoader = DataLoader(traincustomdata, batch_size=set_per_device_train_batch_size, shuffle=True, num_workers=0)
tr_itt = iter(trainDataLoader)

testDataLoader = DataLoader(testcustomdata, batch_size=set_per_device_train_batch_size, shuffle=True, num_workers=0)
tst_itt = iter(testDataLoader)

print("SUCCESS: Data ready for training and evaluation.")

# ------------------------------------------
#         Training & Evaluation
# ------------------------------------------
# Set up the training pipeline using HuggingFace's Trainer:
# 1) Define a data collator: Wav2Vec has much larger input
#    length than output length. Therefore, it is more
#    efficient to pad the training batches dynamically meaning
#    that all training samples should only be padded to the longest
#    sample in their batch and not the overall longest sample.
#    Therefore, fine-tuning Wav2Vec2 required a special
#    padding data collator, defined below.
# 2) Evaluation metric: we evaluate the model using accuracy 
#    We define a compute_metrics function accordingly.
# 3) Load a pre-trained checkpoint
# 4) Define the training configuration
print("\n------> PREPARING FOR TRAINING & EVALUATION... ----------------------- \n")

print("--> Defining pooling layer...")

num_labels = len(id2label)
print("Number of labels:", num_labels)

config = AutoConfig.from_pretrained(
    pretrained_mod,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
    problem_type="single_label_classification",
)
setattr(config, 'pooling_mode', set_pooling_mode)

print("--> Defining Classifer")
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(
            hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 1) Defining data collator
print("--> Defining data collator...")

class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature_extractor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        self.feature_extractor=feature_extractor
        self.padding = True

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)
        return batch

data_collator = DataCollatorCTCWithPadding()

print("SUCCESS: Data collator defined.")

# 2) Evaluation metric
#    Using Accuaracy 
print("--> Defining evaluation metric...")
# The model will return a sequence of logit vectors y.
# We are interested in the most likely prediction of the mode and
# thus take argmax(...) of the logits. We also transform the
# encoded label back to the original string by replacing -100
# with the pad_token_id and decoding the ids while making sure
# that consecutive tokens are not grouped to the same token in
# CTC style.
acc_metric = load_metric("accuracy")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    acc = acc_metric.compute(predictions=pred_str, references=label_str)

    return {"accuracy": acc}


print("SUCCESS: Defined Accuracy evaluation metric.")

# 3) Load pre-trained checkpoint
# Load pre-trained Wav2Vec2 checkpoint. The tokenizer's pad_token_id
# must be to define the model's pad_token_id or in the case of Wav2Vec2ForCTC
# also CTC's blank token. To save GPU memory, we enable PyTorch's gradient
# checkpointing and also set the loss reduction to "mean".
print("--> Loading pre-trained checkpoint...")


model = Wav2Vec2ForSpeechClassification.from_pretrained(
    pretrained_mod,
    config=config,
)

# 1) Define model
"""
model = AutoModelForAudioClassification.from_pretrained(
    pretrained_mod,
    label2id=label2id,
    id2label=id2label,
    num_labels=num_labels,
    hidden_dropout=set_hidden_dropout,
    activation_dropout=set_activation_dropout,
    attention_dropout=set_attention_dropout,
    feat_proj_dropout=set_feat_proj_dropout,
    layerdrop=set_layerdrop,
    mask_time_prob=set_mask_time_prob,
    mask_time_length=set_mask_time_length,
    ctc_loss_reduction=set_ctc_loss_reduction,
    ctc_zero_infinity=set_ctc_zero_infinity,
    gradient_checkpointing=set_gradient_checkpointing,
    pad_token_id=processor.tokenizer.pad_token_id
)
"""
# The first component of Wav2Vec2 consists of a stack of CNN layers
# that are used to extract acoustically meaningful - but contextually
# independent - features from the raw speech signal. This part of the
# model has already been sufficiently trained during pretrainind and
# as stated in the paper does not need to be fine-tuned anymore.
# Thus, we can set the requires_grad to False for all parameters of
# the feature extraction part.
print("SUCCESS: Pre-trained checkpoint loaded.")

print("--> Defining Custom Trainer Class...")
class myTrainer(Trainer):
    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            # train
            train_loss = self._train(train_loader)

            # validate
            val_loss = self._validate(val_loader)
            print(f"Epoch {epoch} Loss {train_loss} val {val_loss}")

    def _train(self, loader):
        # put model in train mode
        self.model.train()
        for i in range(len(loader)):
            # forward pass
            data = next(tr_itt)
            inputs = {}
            inputs['input_values'] = data["input_values"].float()
            inputs['attention_mask'] = data['attention_mask'].long()
            inputs['labels'] = data['label']
            out = self.model(**inputs)
            # loss
            loss = self._compute_loss(model, inputs)
            # remove gradient from previous passes
            self.optimizer.zero_grad()

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            # backprop
            loss.backward()
            # parameters update
            self.optimizer.step()
        return loss.item()

    def _validate(self, loader):
        # put model in evaluation mode
        print("in eval mode")
        self.model.eval()

        with torch.no_grad():
            for i in range(len(loader)):
                features, labels = next(tst_itt)
                out = self.model(features)
                loss = self._compute_loss(out, labels)
                print(loss)

        return loss.item()
        
    def _compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        lossfct = CrossEntropyLoss()
        loss = lossfct(logits, labels)
        return loss
# print("--> Defining CTC Trainer...")
# class CTCTrainer(Trainer):
#     def train(self, model_path: Optional[str] = None):        #train_dataloader = self.get_train_dataloader()
#         train_dataloader = trainDataLoader
#         #self.callback_handler.train_dataloader = train_dataloader
#         self.callback_handler.train_dataloader = train_dataloader
#         print("training...")

#     def predict(
#         self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval"
#     ):
#         #test_dataloader = self.get_test_dataloader(test_dataset)
#         test_dataloader = testDataLoader
#         #output = self.prediction_loop(
#         #    test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
#         #)

#         output = self.prediction_loop(
#             test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
#         )
#         output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))
#         return output

#     def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
#         """
#         Perform a training step on a batch of inputs.

#         Subclass and override to inject custom behavior.

#         Args:
#             model (:obj:`nn.Module`):
#                 The model to train.
#             inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
#                 The inputs and targets of the model.

#                 The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
#                 argument :obj:`labels`. Check your model's documentation for all accepted arguments.

#         Return:
#             :obj:`torch.Tensor`: The tensor with training loss on this batch.
#         """
#         print ("before model train")
#         model.train()
#         print ("before inputs train")
#         inputs = self._prepare_inputs(inputs)
#         print ("before input loss")

#         loss = self.compute_loss(model, inputs)

#         if self.args.gradient_accumulation_steps > 1:
#             print("before loss train")
#             loss = loss / self.args.gradient_accumulation_steps

#         if self.deepspeed:
#             print ("before backward train")
#             self.deepspeed.backward(loss)
#         else:
#             loss.backward()

#         return loss.detach()


# 4) Configure training parameters
#    - group_by_length: makes training more efficient by grouping
#      training samples of similar input length into one batch.
#      Reduces useless padding tokens passed through model.
#    - learning_rate and weight_decay: heuristically tuned until
#      fine-tuning has become stable. These paramteres strongly
#      depend on Timit dataset and might be suboptimal for this
#      dataset.
# For more info: https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer#trainingarguments
model.freeze_feature_extractor()

optimizer = Adafactor(model.parameters(), scale_parameter=True,
                      relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

training_args = TrainingArguments(
    output_dir=model_fp,
    evaluation_strategy=set_evaluation_strategy,
    per_device_train_batch_size=set_per_device_train_batch_size,
    gradient_accumulation_steps=set_gradient_accumulation_steps,
    learning_rate=set_learning_rate,
    weight_decay=set_weight_decay,
    adam_beta1=set_adam_beta1,
    adam_beta2=set_adam_beta2,
    adam_epsilon=set_adam_epsilon,
    num_train_epochs=set_num_train_epochs,
    max_steps=set_max_steps,
    lr_scheduler_type=set_lr_scheduler_type,
    warmup_ratio=set_warmup_ratio,
    logging_strategy=set_logging_strategy,
    logging_steps=set_logging_steps,
    save_strategy=set_save_strategy,
    save_steps=set_save_steps,
    save_total_limit=set_save_total_limit,
    fp16=set_fp16,
    eval_steps=set_eval_steps,
    load_best_model_at_end=set_load_best_model_at_end,
    metric_for_best_model=set_metric_for_best_model,
    greater_is_better=set_greater_is_better,
    group_by_length=set_group_by_length,
    hub_token='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',
    push_to_hub=set_push_to_hub,
    label_names=label2id, 
)
# All instances can be passed to Trainer and
# we are ready to start training!
# model.gradient_checkpointing_enable()
trainer = myTrainer(
    model=model,
    optimizers=(optimizer, lr_scheduler),
    args=training_args,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)

# ------------------------------------------
#               Training
# ------------------------------------------
# While the trained model yields a satisfying result on Timit's
# test data, it is by no means an optimally fine-tuned model
# for children's data.

if training:
    print("\n------> STARTING TRAINING... ----------------------------------------- \n")
    torch.cuda.empty_cache()
    # Train
    trainer.fit(trainDataLoader, testDataLoader, 10)
    if use_checkpoint:
        trainer.train(pretrained_mod)
    else:
        print("here")
        trainer.train()
    # Save the model
    model.save_pretrained(model_fp)

# ------------------------------------------
#            Evaluation
# ------------------------------------------
# Evaluate fine-tuned model on test set.
print("\n------> EVALUATING MODEL... ------------------------------------------ \n")

# Use avaliable GPUs 
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device =  ("cpu")

if eval_pretrained:
    config = AutoConfig.from_pretrained(eval_model)
    processor = Wav2Vec2Processor.from_pretrained(eval_model)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(eval_model).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(eval_model)
    sampling_rate = feature_extractor.sampling_rate
else:
    processor = Wav2Vec2Processor.from_pretrained(model_fp)
    model = Wav2Vec2ForCTC.from_pretrained(model_fp)
    config = AutoConfig.from_pretrained(model_fp)
    processor = Wav2Vec2Processor.from_pretrained(model_fp)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_fp).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_fp)
    sampling_rate = feature_extractor.sampling_rate

def predict(batch):
    features = feature_extractor(
        batch["input_values"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch


results = trainer.predict(testcustomdata)
y_true = [config.label2id[name] for name in results["label_ids"]]
y_pred = results["predictions"]

print(classification_report(y_true, y_pred, target_names=label_list))

# Save results to csv
results_df = results.to_pandas()
results_df.to_csv(finetuned_results_fp)
print("Saved results to:", finetuned_results_fp)

# Getting the Accuracy
print("--> Getting fine-tuned test results...")
print("Fine-tuned Test Accuracy: {:.3f}".format(acc_metric.compute(predictions=results["pred_str"],
      references=results["target_text"])))

# Deeper look into model: running the first test sample through the model,
# take the predicted ids and convert them to their corresponding tokens.
print("--> Taking a deeper look...")
model.to("cuda")
input_values = processor(data["test"][0]["audio"], sampling_rate=data["test"]
                         [0]["sampling_rate"], return_tensors="pt").input_values.to("cuda")

with torch.no_grad():
  logits = model(input_values).logits

pred_ids = torch.argmax(logits, dim=-1)

# convert ids to tokens
print(" ".join(processor.tokenizer.convert_ids_to_tokens(
    pred_ids[0].tolist())))

print("\n------> SUCCESSFULLY FINISHED ---------------------------------------- \n")
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Finished:", dt_string)
