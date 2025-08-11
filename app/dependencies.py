# MultipleFiles/dependencies.py

# =============================================================================
# HATE SPEECH DETECTION PIPELINE - PART 1: MODULE DEPENDENCIES
# =============================================================================

import os
import sys
import logging # PENTING: Import logging di awal

# --- START: Set up logging for Jupyter and console ---
# Pastikan LOG_DIR dibuat sebelum digunakan oleh FileHandler
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'pipeline_log.log')

# Inisialisasi logger SEBELUM penggunaan pertamanya
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Hapus handler yang ada untuk menghindari duplikasi (penting untuk reload/multiple imports)
if logger.hasHandlers():
    logger.handlers.clear()

# Handler untuk konsol (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Handler untuk file log
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG) # DEBUG level untuk file log agar lebih detail
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
# --- END: Set up logging ---

# Sekarang, impor library lain setelah logger didefinisikan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Text Processing
import re
import string
from collections import Counter
from wordcloud import WordCloud
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from pathlib import Path
import threading

# Machine Learning & Deep Learning
import torch
import torch.nn as nn
import psutil
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoConfig,
)

# Importing SciPy for statistical calculations
try:
    from scipy.stats import norm
    # Sekarang logger sudah didefinisikan, jadi ini akan berfungsi
    logger.info("Successfully imported scipy for statistical calculations")
except ImportError:
    logger.warning("scipy not available, using numpy for statistical calculations")
    norm = None

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from IPython.display import display, clear_output

# Utilities
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import argparse
import time
import random

# Set random seeds for reproducibility
RANDOM_SEED = 86
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    logger.info("CUDA not available, using CPU")

