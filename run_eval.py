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
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from typing import Optional, Tuple, Any, Dict, Union
import customTransform as T
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torchaudio
from torchvision import transforms
import torch.distributed as dist
import customTransform as T
from customData import CustomDataset
from transformers.file_utils import ModelOutput
import gc
import tensorflow as tf 
from transformers import (
    Trainer,
    TrainingArguments,
    AutoConfig,
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
print("                         run_xlsr.py                            ")
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
print("\n------> EXPERIMENT ARGUMENTS ----------------------------------------- \n")
batch_size = 40        # Default = 8
print("batch_size:", batch_size)
# Experiment ID
# For
#     1) naming model output directory
#     2) naming results file
experiment_id = "ADI17-eval"
print("experiment_id:", experiment_id)

# DatasetDict Id
# For 1) naming cache directory and
#     2) saving the DatasetDict object
datasetdict_id = "myST-eval"
print("datasetdict_id:", datasetdict_id)

data_path = "/srv/scratch/z5208494/dataset/"
print("data path:", data_path)

train_data_path = "/srv/scratch/z5208494/dataset/train_segments/"
print("test data path:", train_data_path)

dev_data_path = "/srv/scratch/z5208494/dataset/dev_segments/"
print("training data path:", dev_data_path)

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
train_name = "u_train_700f"
train_filename = "u_train_700f"
print("train_name:", train_name)
print("train_filename:", train_filename)

validation_filename = "dev_u_200f"
print("validation_filename:", validation_filename)

# Evaluation dataset name and filename
# Dataset name and filename of the csv file containing the evaluation data
# For generating filepath to file location

evaluation_filename = "test_u_100f"
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
model_path = "model/"

#-----------------------------------------
#        Generating file paths
# ------------------------------------------
print("\n------> GENERATING FILEPATHS... --------------------------------------\n")
# Path to dataframe csv for train dataset
data_base_fp = "data/"
data_train_fp = data_base_fp + train_filename + ".csv"
print("--> data_train_fp:", data_train_fp)

data_val_fp = data_base_fp + validation_filename + ".csv"
print("--> data_test_fp:", data_val_fp)

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

print("\n------> LOAD SAVED MODEL ----------------------------------------- \n")
model = tf.keras.models.load_model(model_path, custom_objects=None, compile=True, options=None)


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

# ------------------------------------------
#             Pre-process Data
# ------------------------------------------
print("\n------> PRE-PROCESSING DATA... ----------------------------------------- \n")

def print_gpu_info():
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
    else:
        print('not using cuda')


max_duration = 10
print("Max Duration:", max_duration, "s")
sampling_rate = 16000
target_sampling_rate = 16000
print("Sampling Rate:",  sampling_rate)
print("Target Sampling Rate:",  target_sampling_rate)

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


# create custom dataset class
print("Create a custom dataset ---> ")
random_transforms = transforms.Compose([T.Extractor(model, sampling_rate, max_duration)])

traincustomdata = CustomDataset(
    csv_fp=data_train_fp, data_fp=train_data_path, labels=label_list, transform=random_transforms, model_name=model_name, max_length=max_duration)
valcustomdata = CustomDataset(
    csv_fp=data_val_fp, data_fp=dev_data_path, labels=label_list, transform=random_transforms, model_name=model_name, max_length=max_duration)
testcustomdata = CustomDataset(
    csv_fp=data_test_fp, data_fp=test_data_path, labels=label_list, transform=random_transforms, model_name=model_name, max_length=max_duration)


trainDataLoader = DataLoader(
    traincustomdata, batch_size=batch_size, shuffle=True, num_workers=set_num_of_workers)

valDataLoader = DataLoader(
    valcustomdata, batch_size=batch_size, shuffle=True, num_workers=set_num_of_workers)

testDataLoader = DataLoader(
    testcustomdata, batch_size=batch_size, shuffle=True, num_workers=set_num_of_workers)

print("Check data has been processed correctly... ")
print("Train Data Sample")
TrainData = next(iter(trainDataLoader))
print(TrainData)
print("Training DataCustom Files: "+ str(len(traincustomdata)))
print("Training Data Files: "+ str(len(trainDataLoader)))

print("Val Data Sample")
ValData = next(iter(valDataLoader))
print(ValData)
print("Test CustomData Files: " + str(len(valcustomdata)))
print("Test Data Files: " + str(len(valDataLoader)))

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

def plot_data(x_label, y_label, matrix, name):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap=plt.cm.Blues)

    fig.colorbar(cax)
    xaxis = np.arange(len(x_label))
    yaxis = np.arange(len(y_label))
    ax.set_xticks(xaxis)
    ax.set_yticks(yaxis)
    ax.tick_params(labelbottom=True, labeltop=False)
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label)
    plt.savefig("output/"+experiment_id+name+".png")

model.classifier = nn.Linear(in_features=256, out_features=num_labels, bias=True)

print("-------- Setting up Model --------")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
multi_gpu = False
if torch.cuda.device_count() > 1:
    print('GPUs Used : ', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
    multi_gpu = True

model.to(device)

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

    correct_pred = (y_pred_tags == y_test).float().to(device).contiguous()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

print("--> Defining Custom Trainer Class...")
def evaluate(self, loader, tst_itt):
    # put model in evaluation mode
    loss_sum = 0
    acc_sum = 0
    y_true = []
    y_pred = []
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
                labels = labels.reshape((labels.shape[0])).long().to(device).contiguous()
                predictions = model(**inputs).logits
                loss, acc = self._compute_loss(model, inputs, labels)
                loss_sum += loss.detach()
                acc_sum += acc.detach()
                for j in range(0, len(predictions)):
                    y_pred.append(np.argmax(predictions[j].cpu()).item())
                    y_true.append(labels[j].cpu().item())
            except StopIteration:
                break

    loss_tot = loss_sum/len(loader)
    acc_tot = acc_sum/len(loader)
    print(f"Final Test Acc:{acc_tot}% Loss:{loss_tot}")
    outcsv.write(f"Final Test,{acc_tot},{loss_tot}\n")

    c_matrix = confusion_matrix(y_true, y_pred)
    c_matrix_norm = confusion_matrix(y_true, y_pred, normalize='all')
    print("CONFUSION MATRIX")
    print(c_matrix)
    print("CONFUSION MATRIX NORMALISED")
    print(c_matrix_norm)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred))

    plot_data(label_list, label_list, c_matrix, "")
    plot_data(label_list, label_list, c_matrix_norm, "-norm")
# ------------------------------------------
#            Evaluation
# ------------------------------------------
# Evaluate fine-tuned model on test set.
print("\n------> EVALUATING MODEL... ------------------------------------------ \n")

tst_itt = iter(testDataLoader)
evaluate(model, testDataLoader, tst_itt)

print("\n------> SUCCESSFULLY FINISHED ---------------------------------------- \n")
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Finished:", dt_string)
