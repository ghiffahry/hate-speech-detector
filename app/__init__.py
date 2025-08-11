# app/__init__.py - Fixed circular import issue

# Import core components only
from .config import Config
from .dependencies import logger, device
from .predictor import HateSpeechPredictor
from .preprocessor import IndonesianTextPreprocessor
from .utils import (
    save_model_artifacts,
    load_model_artifacts,
    calculate_class_weights,
    get_model_summary
)

# Versi package
__version__ = "2.0.0"
__author__ = "Mafty Navue Erin"
__email__ = "ghiffaryankh@gmail.com"

# Ekspor komponen utama
__all__ = [
    'HateSpeechPredictor',
    'IndonesianTextPreprocessor',
    'Config',
    'logger',
    'device',
    'save_model_artifacts',
    'load_model_artifacts',
    'calculate_class_weights',
    'get_model_summary'
]

# Inisialisasi logger saat package dimuat
logger.info(f"Memuat package deteksi_ujaran_kebencian versi {__version__}")
logger.info(f"Device yang digunakan: {device}")
