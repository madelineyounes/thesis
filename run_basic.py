#----------------------------------------------------------
# run_umbrellaDID.py
# Purpose: Uses xlsr to create a audio classifer that
# identifies arabic dialects using the ADI17 dataset.
# Author: Madeline Younes, 2022
#----------------------------------------------------------

# ------------------------------------------
#      Install packages if needed
# ------------------------------------------
#pip3 install datasets
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
from datasets import load_metric, load_dataset
from typing import Any, Dict, List, Optional, Union
import pyarrow.csv as csv
import pyarrow as pa
from sklearn.metrics import classification_report
from typing import Optional, Tuple, Any, Dict, Union
import customTransform as T
import torch.nn as nn
import time
import math
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
import torchaudio
from torchvision import transforms
import customTransform as T
from customData import CustomDataset
from trainer_util import PredictionOutput, speed_metrics
from transformers.file_utils import ModelOutput
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    AutoConfig,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForSequenceClassification
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
experiment_id = "wav2vec-ADI17-initialtest"
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
train_name = "umbrella_100f_devdata"
train_filename = "train_u_100f"
print("train_name:", train_name)
print("train_filename:", train_filename)

# Evaluation dataset name and filename
# Dataset name and filename of the csv file containing the evaluation data
# For generating filepath to file location

#evaluation_filename = "adi17_test_umbrella_label"
evaluation_filename =  "test_u_50f"
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
# model_name = "elgeish/wav2vec2-large-xlsr-53-arabic"
# model_name = "facebook/wav2vec2-base"
# try log0/wav2vec2-base-lang-id

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
set_num_of_workers = 2  # equivilent to cpus*gpu 
print("number_of_worker:", set_num_of_workers)
set_hidden_dropout = 0.1                    # Default = 0.1
print("hidden_dropout:", set_hidden_dropout)
set_activation_dropout = 0.1                # Default = 0.1
print("activation_dropout:", set_activation_dropout)
set_attention_dropout = 0.1                 # Default = 0.1
print("attention_dropoutput:", set_attention_dropout)
set_feat_proj_dropout = 0.0                 # Default = 0.1
print("feat_proj_dropout:", set_feat_proj_dropout)
set_layerdrop = 0.1                        # Default = 0.1
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
set_evaluation_strategy = "no"           # Default = "no"
print("evaluation strategy:", set_evaluation_strategy)
set_per_device_train_batch_size = 10         # Default = 8
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
set_unfreezing_step = 10                   # Default = 3.0
print("unfreezing_step:", set_unfreezing_step)
set_num_train_epochs = 100                  # Default = 3.0
print("num_train_epochs:", set_num_train_epochs)
set_max_steps = -1                       # Default = -1, overrides epochs
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
set_save_steps = 500                         # Default = 500
print("save_steps:", set_save_steps)
set_save_total_limit = 40                   # Optional
print("save_total_limit:", set_save_total_limit)
set_fp16 = False                             # Default = False
print("fp16:", set_fp16)
set_eval_steps = 100                         # Optional
print("eval_steps:", set_eval_steps)
set_load_best_model_at_end = False           # Default = False
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
# Path to results csv 
output_csv_fp = "output/results_" + experiment_id + ".csv"
outcsv = open(output_csv_fp, 'w+')
outcsv.write("epoch,train_acc,val_acc,train_loss,val_loss\n")


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

# ------------------------------------------
#         Preparing dataset
# ------------------------------------------
# Run the following scripts to prepare data
# 1) Prepare data from kaldi file:
# 3) [Optional] Limit the files to certain duration:
# 2) Split data into train and test:

print("\n------> PREPARING DATASET LABELS... ------------------------------------\n")
# Read the existing csv saved dataframes and
# load as a DatasetDict
label_list = ['NOR', 'EGY', 'GLF', 'LEV']
label2id, id2label = dict(), dict()
for i, label in enumerate(label_list):
    label2id[label] = str(i)
    id2label[str(i)] = label

num_labels = len(id2label)

print("\n------> Creating blank confusion matrix ... -----------------------\n")
matrix = np.empty((0, num_labels), dtype=int)
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
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True,  return_tensors='pt').from_pretrained(model_name)
# Feature extractor and tokenizer wrapped into a single
# Wav2Vec2Processor class so we only need a model and processor object
#processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Save to re-use the just created processor and the fine-tuned model
#processor.save_pretrained(model_fp)
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
max_duration = 5
print("Max Duration:",max_duration, "s")
sampling_rate = feature_extractor.sampling_rate
print("Sampling Rate:",  sampling_rate)

# create custom dataset class
print("Create a custom dataset ---> ")
random_transforms = transforms.Compose(
    [T.Extractor(model_name, sampling_rate, max_duration)])

traincustomdata = CustomDataset(
    csv_fp=data_train_fp, data_fp=training_data_path, labels=label_list, transform=random_transforms, model_name=model_name, max_length=max_duration)
testcustomdata = CustomDataset(
    csv_fp=data_test_fp, data_fp=test_data_path, labels=label_list, transform=random_transforms, model_name=model_name, max_length=max_duration)

trainDataLoader = DataLoader(
    traincustomdata, batch_size=set_per_device_train_batch_size, shuffle=True, num_workers=set_num_of_workers)

testDataLoader = DataLoader(
    testcustomdata, batch_size=set_per_device_train_batch_size, shuffle=True, num_workers=set_num_of_workers)

print("Check data has been processed correctly... ")
print("Train Data Sample")
TrainData = next(iter(trainDataLoader))
print(TrainData)
print("Training DataCustom Files: "+ str(len(traincustomdata)))
print("Training Data Files: "+ str(len(trainDataLoader)))

print("Test Data Sample")
TestData = next(iter(testDataLoader))
print(TestData)
print("Test CustomData Files: " + str(len(testcustomdata)))
print("Test Data Files: " + str(len(testDataLoader)))

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
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [feature["labels"]
                          for feature in features]

        self.feature_extractor = feature_extractor
        self.padding = True

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )

        batch["labels"] = torch.stack(label_features)
        return batch

data_collator = DataCollatorCTCWithPadding()
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
# NOTE: CAN PROBS DELETE THIS SECTION
"""
def compute_metrics(pred):
    print("PRED", pred)
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    print("LABELS IDS", pred.label_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    print("LABELS STRING", label_str)
    print("PRED IDS", pred_str)
    acc = acc_metric.compute(predictions=pred_str, references=pred.label_ids)
    return {"accuracy": acc}
"""
print("SUCCESS: Defined Accuracy evaluation metric.")
# 3) Load pre-trained checkpoint
# Load pre-trained Wav2Vec2 checkpoint. The tokenizer's pad_token_id
# must be to define the model's pad_token_id or in the case of Wav2Vec2ForCTC
# also CTC's blank token. To save GPU memory, we enable PyTorch's gradient
# checkpointing and also set the loss reduction to "mean".

print("--> Loading pre-trained checkpoint...")
# NOTE: SWAPED Wav2Vec2ForSpeechClassification to Wav2Vec2ForSequenceClassification
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    pretrained_mod,
    config=config
)

model.classifier = nn.Linear(in_features=256, out_features=num_labels, bias=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
multi_gpu = False
if torch.cuda.device_count() > 1:
    print('GPUs Used : ', torch.cuda.device_count(), 'GPUs!')
    multi_gpu = True

model.to(device)

print("-------- Setting up Model --------")


for param in model.wav2vec2.feature_extractor.parameters():
    param.requires_grad = False

for param in model.wav2vec2.encoder.parameters():
    param.requires_grad = False

for param in model.wav2vec2.feature_projection.parameters():
    param.requires_grad = False

""""
trainable_transformers = 12
num_transformers = 12
if trainable_transformers > 0:
    for i in range(num_transformers-trainable_transformers, num_transformers, 1):
        for param in model.wav2vec2.encoder.layers[i].parameters():
            param.requires_grad = True
"""

# 1) Define model

# The first component of Wav2Vec2 consists of a stack of CNN layers
# that are used to extract acoustically meaningful - but contextually
# independent - features from the raw speech signal. This part of the
# model has already been sufficiently trained during pretrainind and
# as stated in the paper does not need to be fine-tuned anymore.
# Thus, we can set the requires_grad to False for all parameters of
# the feature extraction part.
print("SUCCESS: Pre-trained checkpoint loaded.")

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)
    return acc

print("--> Defining Custom Trainer Class...")


class myTrainer(Trainer):
    def fit(self, train_loader, val_loader, epochs):
        """
        for epoch in range(epochs):
            print("EPOCH unfeeze : " + str(epoch % set_unfreezing_step))
           
            if epoch != 0 and epoch % set_unfreezing_step == 0 :
                if epoch // set_unfreezing_step < (num_transformers-trainable_transformers):
                    if multi_gpu:
                        print("multi GPU used")
                        for param in model.module.wav2vec2.encoder.layers[num_transformers-(epoch//set_unfreezing_step) - trainable_transformers].parameters():
                            param.requires_grad = True
                    else:
                        for param in model.wav2vec2.encoder.layers[num_transformers-(epoch//set_unfreezing_step)-trainable_transformers].parameters():
                            print("grad change")
                            param.requires_grad = True
        """
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable Parameters : ' + str(params))
        
        loss_sum_tr = 0
        acc_sum_tr = 0
        loss_sum_val = 0
        acc_sum_val = 0
        tr_itt = iter(trainDataLoader)
        tst_itt = iter(testDataLoader)
        print("start train")
        # train
        train_loss, train_acc = self._train(train_loader, tr_itt, loss_sum_tr, acc_sum_tr)
        print("start validation")
        # validate
        val_loss, val_acc = self._validate(val_loader, tst_itt, loss_sum_val, acc_sum_val)

        print(f"Epoch {epoch} Train Acc {train_acc}% Val Acc {val_acc}% Train Loss {train_loss} Val Loss {val_loss}")
        outcsv.write(f"{epoch},{train_acc},{val_acc},{train_loss},{val_loss}\n")

        # on the last epoch generate a con

    def _train(self, loader, tr_itt, loss_sum_tr, acc_sum_tr):
        # put model in train mode
        self.model.train()
        for i in range(len(loader)):
            # forward pass
            try:
                data = next(tr_itt)
                inputs = {}
                inputs['input_values'] = data['input_values'].float().to(device).contiguous()
                inputs['attention_mask'] = data['attention_mask'].long().to(device).contiguous()
                labels = data['labels'].long().to(device).contiguous()
                # loss
                loss, acc = self._compute_loss(model, inputs, labels)
                # remove gradient from previous passes
                self.optimizer.zero_grad()

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                # parameters update
                self.optimizer.step()

                loss_sum_tr += loss.detach()
                acc_sum_tr += acc.detach()
            except StopIteration:
                break
        loss_tot_tr = loss_sum_tr/len(loader)
        acc_tot_tr = acc_sum_tr/len(loader)
        return loss_tot_tr, acc_tot_tr

    def _validate(self, loader, tst_itt, loss_sum_val, acc_sum_val):
        # put model in evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i in range(len(loader)):
                try:
                    data = next(tst_itt)
                    inputs = {}
                    inputs['input_values'] = data['input_values'].float(
                    ).to(device).contiguous()
                    inputs['attention_mask'] = data['attention_mask'].long(
                    ).to(device).contiguous()
                    labels = data['labels'].long().to(device).contiguous()
                    loss, acc = self._compute_loss(model, inputs, labels)
                    loss_sum_val += loss.detach()
                    acc_sum_val += acc.detach()
                except StopIteration:
                    break
        loss_tot_val = loss_sum_val/len(loader)
        acc_tot_val = acc_sum_val/len(loader)
        return loss_tot_val, acc_tot_val

    def _compute_loss(self, model, inputs, labels):
        prediction = model(**inputs).logits
        lossfct = CrossEntropyLoss()
        loss = lossfct(prediction, labels.reshape((labels.shape[0])).long().to(device).contiguous())
        acc = multi_acc(prediction, labels.reshape(
            (labels.shape[0])).long().to(device).contiguous())
        return loss, acc

    def _gen_prediction(self, loader, tst_itt):
        # put model in evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i in range(len(loader)):
                try:
                    data = next(tst_itt)
                    inputs = {}
                    inputs['input_values'] = data['input_values'].float(
                    ).to(device).contiguous()
                    inputs['attention_mask'] = data['attention_mask'].long(
                    ).to(device).contiguous()
                    labels = data['labels'].long().to(device).contiguous()
                    labels = labels.reshape(
                        (labels.shape[0])).long().to(device).contiguous()
                    prediction = model(**inputs).logits

                except StopIteration:
                    break

    def _predict(self, test_dataloader):
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
        <Tip>
        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(test_dataloader, description="Prediction")
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                split="test",
                start_time=start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)


# model.freeze_feature_extractor()
optimizer = Adafactor(model.parameters(), scale_parameter=True,
                      relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

training_args = TrainingArguments(
    output_dir=model_fp,
    evaluation_strategy=set_evaluation_strategy,
    per_device_train_batch_size=set_per_device_train_batch_size,
    gradient_accumulation_steps=set_gradient_accumulation_steps,
    gradient_checkpointing=set_gradient_checkpointing,
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
    eval_steps=myTrainer,
    load_best_model_at_end=set_load_best_model_at_end,
    metric_for_best_model=set_metric_for_best_model,
    greater_is_better=set_greater_is_better,
    group_by_length=set_group_by_length,
    hub_token='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',
    push_to_hub=set_push_to_hub,
)
# All instances can be passed to Trainer and
# we are ready to start training!
# model.gradient_checkpointing_enable()
trainer = myTrainer(
    model=model,
    optimizers=(optimizer, lr_scheduler),
    args=training_args,
    data_collator=data_collator,
)

# ------------------------------------------
#               Training
# ------------------------------------------
# While the trained model yields a satisfying result on Timit's
# test data, it is by no means an optimally fine-tuned model
# for children's data.

if training:
    print("\n------> STARTING TRAINING... ----------------------------------------- \n")
    # Use avaliable GPUs
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = ("cpu")
    # Train
    trainer.fit(trainDataLoader, testDataLoader, set_num_train_epochs)

    # Save the model
    model.save_pretrained(model_fp)

# ------------------------------------------
#            Evaluation
# ------------------------------------------
# Evaluate fine-tuned model on test set.
print("\n------> EVALUATING MODEL... ------------------------------------------ \n")


results = trainer._predict(testDataLoader)
y_true = results[1]
# create an array selecting the highest prediction value.
y_pred = []
for predicts in results[0]:
    p = np.where(predicts == np.amax(predicts))
    y_pred.append(p[0][0])


print("CLASSIFICATION REPORT")
print(classification_report(y_true, y_pred))

# Deeper look into model: running the first test sample through the model,
# take the predicted ids and convert them to their corresponding tokens.

print("\n------> SUCCESSFULLY FINISHED ---------------------------------------- \n")
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Finished:", dt_string)
