#!/usr/bin/env python3
"""
config.py

This file deteremines which device PyTorch will utilise.
"""

import torch

# Use a GPU if available, as it should be faster.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
