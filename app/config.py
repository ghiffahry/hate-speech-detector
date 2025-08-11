# MultipleFiles/config.py

import os
import json
import logging # Tetap import logging di sini untuk fallback
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoTokenizer
from datetime import datetime

try:
    # Impor logger dari dependencies.py
    # Karena dependencies.py sudah dijamin mendefinisikan logger di awal, ini akan aman
    from .dependencies import logger
except ImportError:
    # Fallback logger jika dependencies.py tidak dapat diimpor atau logger tidak ditemukan
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("Could not import 'logger' from dependencies.py. Using fallback logging configuration.")


class Config:
    """
    Optimized configuration for hate speech detection model
    Addresses: performance, memory efficiency, training stability, and production readiness
    """

    def __init__(self,
                 MODEL_NAME: str = "indolem/indobert-base-uncased",
                 BATCH_SIZE: int = 16,
                 FP16: Optional[bool] = None,
                 GRADIENT_ACCUMULATION_STEPS: int = 2,
                 DATALOADER_NUM_WORKERS: int = 4,
                 USE_CUDA: Optional[bool] = None,
                 NUM_EPOCHS: int = 4,
                 MAX_LENGTH: int = 128,
                 LOGGING_STEPS: int = 50,
                 MAX_TRAINING_STEPS=1000,
                 SAVE_STEPS: int = 500,
                 ENABLE_MONITORING: bool = True,
                 API_VERSION: str = "2.0.0", # Added for app.py consistency
                 ENVIRONMENT: str = "development", # Added for app.py consistency
                 **kwargs):

        # ================== DATA CONFIGURATION ==================
        self.TRAIN_SIZE = kwargs.get('TRAIN_SIZE', 0.8)
        self.VAL_SIZE = kwargs.get('VAL_SIZE', 0.1)
        self.TEST_SIZE = kwargs.get('TEST_SIZE', 0.1)
        self.RANDOM_STATE = kwargs.get('RANDOM_STATE', 42)

        # Validate data split
        total_size = self.TRAIN_SIZE + self.VAL_SIZE + self.TEST_SIZE
        if abs(total_size - 1.0) > 0.001:
            # Using global logger
            logger.error(f"Data split sizes must sum to 1.0, got {total_size}")
            raise ValueError(f"Data split sizes must sum to 1.0, got {total_size}")

        # ================== MODEL CONFIGURATION ==================
        self.MODEL_NAME = MODEL_NAME
        self.NUM_LABELS = 2

        self.MAX_LENGTH = MAX_LENGTH
        self.ADAPTIVE_MAX_LENGTH = kwargs.get('ADAPTIVE_MAX_LENGTH', True)
        self.MAX_LENGTH_PERCENTILE = kwargs.get('MAX_LENGTH_PERCENTILE', 95)

        # Preprocessing flags - Ensure consistency with preprocessor.py and training
        self.USE_MINIMAL_PREPROCESSING = kwargs.get('USE_MINIMAL_PREPROCESSING', True)
        self.REMOVE_STOPWORDS = kwargs.get('REMOVE_STOPWORDS', False) # Default False for BERT
        self.USE_STEMMING = kwargs.get('USE_STEMMING', False)         # Default False for BERT
        self.NORMALIZE_SLANG = kwargs.get('NORMALIZE_SLANG', True)

        # ================== TRAINING OPTIMIZATION ==================
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS

        self.LEARNING_RATE = kwargs.get('LEARNING_RATE', 3e-5)
        self.WARMUP_RATIO = kwargs.get('WARMUP_RATIO', 0.1)
        self.LR_SCHEDULER_TYPE = kwargs.get('LR_SCHEDULER_TYPE', 'cosine')
        self.WEIGHT_DECAY = kwargs.get('WEIGHT_DECAY', 0.01)
        self.MAX_TRAINING_STEPS = MAX_TRAINING_STEPS

        self.GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS
        self.MAX_GRAD_NORM = kwargs.get('MAX_GRAD_NORM', 1.0)
        self.GRADIENT_CHECKPOINTING = kwargs.get('GRADIENT_CHECKPOINTING', True)

        # ================== HARDWARE OPTIMIZATION ==================
        self.USE_CUDA = USE_CUDA if USE_CUDA is not None else torch.cuda.is_available()
        self.FP16 = FP16 if FP16 is not None else (torch.cuda.is_available() and self._supports_fp16())

        cpu_count = os.cpu_count() or 4
        self.DATALOADER_NUM_WORKERS = min(DATALOADER_NUM_WORKERS, cpu_count // 2, 8)

        self.DATALOADER_PIN_MEMORY = kwargs.get('DATALOADER_PIN_MEMORY', self.USE_CUDA)
        self.EMPTY_CACHE_STEPS = kwargs.get('EMPTY_CACHE_STEPS', 100)

        self.AUTO_FIND_BATCH_SIZE = kwargs.get('AUTO_FIND_BATCH_SIZE', True)

        # ================== TRAINING STABILITY ==================
        self.EARLY_STOPPING_PATIENCE = kwargs.get('EARLY_STOPPING_PATIENCE', 3)
        self.EARLY_STOPPING_THRESHOLD = kwargs.get('EARLY_STOPPING_THRESHOLD', 0.001)
        self.METRIC_FOR_BEST_MODEL = kwargs.get('METRIC_FOR_BEST_MODEL', 'eval_f1')
        self.GREATER_IS_BETTER = kwargs.get('GREATER_IS_BETTER', True)

        self.DROPOUT_RATE = kwargs.get('DROPOUT_RATE', 0.1)
        self.ATTENTION_DROPOUT = kwargs.get('ATTENTION_DROPOUT', 0.1)
        self.LABEL_SMOOTHING = kwargs.get('LABEL_SMOOTHING', 0.0)

        # ================== EVALUATION & LOGGING ==================
        self.LOGGING_STEPS = LOGGING_STEPS
        self.EVAL_STEPS = kwargs.get('EVAL_STEPS', 250)
        self.EVALUATION_STRATEGY = kwargs.get('EVALUATION_STRATEGY', 'steps')
        self.SAVE_STEPS = SAVE_STEPS
        self.SAVE_STRATEGY = kwargs.get('SAVE_STRATEGY', 'steps')
        self.SAVE_TOTAL_LIMIT = kwargs.get('SAVE_TOTAL_LIMIT', 2)
        self.LOAD_BEST_MODEL_AT_END = kwargs.get('LOAD_BEST_MODEL_AT_END', True)

        # ================== DIRECTORIES ==================
        # 🔥 PATH VALIDATION: Ensure path is a valid string
        # Ensure this OUTPUT_DIR is consistent with MODEL_PATH in app.py
        self.OUTPUT_DIR = self._validate_path(kwargs.get('OUTPUT_DIR', './model/optimized'), 'OUTPUT_DIR')
        self.LOGS_DIR = self._validate_path(kwargs.get('LOGS_DIR', './logs'), 'LOGS_DIR')
        self.PLOTS_DIR = self._validate_path(kwargs.get('PLOTS_DIR', './plots'), 'PLOTS_DIR')
        self.CACHE_DIR = self._validate_path(kwargs.get('CACHE_DIR', './cache'), 'CACHE_DIR')

        # Create directories
        for dir_path in [self.OUTPUT_DIR, self.LOGS_DIR, self.PLOTS_DIR, self.CACHE_DIR]:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except OSError as e:
                # Using global logger
                logger.error(f"⚠️ Error creating directory {dir_path}: {e}")
                pass # Continue execution even if directory creation fails

        # ================== MONITORING & VISUALIZATION ==================
        # These parameters are more relevant for training, but kept in config
        self.ENABLE_MONITORING = ENABLE_MONITORING
        self.MONITOR_UPDATE_INTERVAL = kwargs.get('MONITOR_UPDATE_INTERVAL', 10.0)
        self.MONITOR_HISTORY_SIZE = kwargs.get('MONITOR_HISTORY_SIZE', 1000)
        self.SAVE_MONITORING_PLOTS = kwargs.get('SAVE_MONITORING_PLOTS', True)
        self.PLOT_SAVE_PATH = self._validate_path(kwargs.get('PLOT_SAVE_PATH', "./training_plots/"), 'PLOT_SAVE_PATH')
        try:
            os.makedirs(self.PLOT_SAVE_PATH, exist_ok=True)
        except OSError as e:
            # Using global logger
            logger.error(f"⚠️ Error creating directory {self.PLOT_SAVE_PATH}: {e}")
            pass

        # ================== INFERENCE OPTIMIZATION ==================
        self.INFERENCE_BATCH_SIZE = kwargs.get('INFERENCE_BATCH_SIZE', 32)
        self.ENABLE_TORCH_COMPILE = kwargs.get('ENABLE_TORCH_COMPILE', False)
        self.ENABLE_OPTIMUM_OPTIMIZATION = kwargs.get('ENABLE_OPTIMUM_OPTIMIZATION', False)

        # ================== API & ENVIRONMENT ==================
        self.API_VERSION = API_VERSION
        self.ENVIRONMENT = ENVIRONMENT

        # ================== DYNAMIC WARMUP STEPS ==================
        if 'WARMUP_STEPS' in kwargs:
            self.WARMUP_STEPS = kwargs['WARMUP_STEPS']
        else:
            self.WARMUP_STEPS = int(self.WARMUP_RATIO * self.MAX_TRAINING_STEPS)

        # Apply all optimizations
        self._apply_all_optimizations()

        # Print comprehensive summary
        self._print_optimization_summary()

    def _validate_path(self, path: Any, param_name: str) -> str:
        """Helper to validate and return a string path."""
        if not isinstance(path, str) or not path.strip():
            default_path = f'./default_{param_name.lower()}'
            # Using global logger
            logger.warning(f"⚠️ Invalid path for {param_name}: '{path}'. Using default: '{default_path}'")
            return default_path
        return path

    def _supports_fp16(self) -> bool:
        """Check if current GPU supports FP16"""
        if not torch.cuda.is_available():
            return False
        try:
            capability = torch.cuda.get_device_capability()
            return capability[0] >= 7
        except Exception as e:
            # Using global logger
            logger.warning(f"⚠️ Could not determine FP16 support: {str(e)}")
            return False

    def _apply_all_optimizations(self):
        """Apply all hardware and software optimizations"""
        if self.USE_CUDA and torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                torch.cuda.empty_cache()
                # Using global logger
                logger.info("⚡ CUDA optimizations enabled")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                # Using global logger
                logger.info(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            except Exception as e:
                # Using global logger
                logger.warning(f"⚠️ Some CUDA optimizations failed: {str(e)}")

        try:
            torch.backends.mkldnn.enabled = True
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(min(8, os.cpu_count() or 4))
            # Using global logger
            logger.info("⚡ CPU optimizations enabled")
        except Exception as e:
            # Using global logger
            logger.warning(f"⚠️ Some CPU optimizations failed: {str(e)}")

    def _print_optimization_summary(self):
        """Print comprehensive optimization summary"""
        # Using global logger initialized in dependencies.py
        logger.info("\n" + "="*60)
        logger.info("🚀 OPTIMIZED HATE SPEECH DETECTION CONFIG")
        logger.info("="*60)

        logger.info(f"🤖 Model: {self.MODEL_NAME}")
        logger.info(f"📏 Max Length: {self.MAX_LENGTH} (adaptive: {self.ADAPTIVE_MAX_LENGTH})")
        logger.info(f"📊 Batch Size: {self.BATCH_SIZE} (auto-find: {self.AUTO_FIND_BATCH_SIZE})")
        logger.info(f"🔄 Gradient Accumulation: {self.GRADIENT_ACCUMULATION_STEPS}")
        logger.info(f"📈 Effective Batch Size: {self.get_effective_batch_size()}")

        logger.info(f"\n📚 Training:")
        logger.info(f"   Epochs: {self.NUM_EPOCHS}")
        logger.info(f"   Learning Rate: {self.LEARNING_RATE}")
        logger.info(f"   LR Scheduler: {self.LR_SCHEDULER_TYPE}")
        logger.info(f"   Warmup Ratio: {self.WARMUP_RATIO}")
        logger.info(f"   Weight Decay: {self.WEIGHT_DECAY}")

        logger.info(f"\n⚡ Hardware:")
        logger.info(f"   CUDA: {self.USE_CUDA}")
        logger.info(f"   FP16: {self.FP16}")
        logger.info(f"   Workers: {self.DATALOADER_NUM_WORKERS}")
        logger.info(f"   Pin Memory: {self.DATALOADER_PIN_MEMORY}")
        logger.info(f"   Gradient Checkpointing: {self.GRADIENT_CHECKPOINTING}")

        logger.info(f"\n🎯 Optimization Features:")
        logger.info(f"   ✅ Adaptive max length")
        logger.info(f"   ✅ Memory management")
        logger.info(f"   ✅ Learning rate scheduling")
        logger.info(f"   ✅ Gradient clipping")
        logger.info(f"   ✅ Early stopping")
        logger.info(f"   ✅ Minimal preprocessing")

        logger.info("="*60)

    def optimize_for_dataset(self, texts: list, min_length: int = 32):
        """
        🔥 DYNAMIC OPTIMIZATION: Adapt config based on actual dataset
        """
        if not texts or not self.ADAPTIVE_MAX_LENGTH:
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            lengths = []

            sample_size = min(1000, len(texts))
            sample_texts = texts[:sample_size] if len(texts) > sample_size else texts

            for text in sample_texts:
                if text and isinstance(text, str):
                    tokens = tokenizer.encode(text, add_special_tokens=True)
                    lengths.append(len(tokens))

            if lengths:
                percentile_length = int(np.percentile(lengths, self.MAX_LENGTH_PERCENTILE))
                optimal_length = max(min_length, min(percentile_length, 512))

                if optimal_length != self.MAX_LENGTH:
                    # Using global logger
                    logger.info(f"📏 Optimizing max_length: {self.MAX_LENGTH} → {optimal_length}")
                    logger.info(f"   Dataset stats: min={min(lengths)}, max={max(lengths)}, "
                                     f"{self.MAX_LENGTH_PERCENTILE}th percentile={percentile_length}")
                    self.MAX_LENGTH = optimal_length

        except Exception as e:
            # Using global logger
            logger.warning(f"⚠️ Could not optimize max_length: {str(e)}")

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size"""
        return self.BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS

    def get_training_args(self) -> Dict[str, Any]:
        """
        🔥 Get optimized training arguments for Transformers Trainer
        """
        return {
            'output_dir': self.OUTPUT_DIR,
            'num_train_epochs': self.NUM_EPOCHS,
            'per_device_train_batch_size': self.BATCH_SIZE,
            'per_device_eval_batch_size': self.BATCH_SIZE,
            'gradient_accumulation_steps': self.GRADIENT_ACCUMULATION_STEPS,
            'learning_rate': self.LEARNING_RATE,
            'weight_decay': self.WEIGHT_DECAY,
            'warmup_ratio': self.WARMUP_RATIO,
            'lr_scheduler_type': self.LR_SCHEDULER_TYPE,
            'fp16': self.FP16,
            'gradient_checkpointing': self.GRADIENT_CHECKPOINTING,
            'max_grad_norm': self.MAX_GRAD_NORM,
            'logging_steps': self.LOGGING_STEPS,
            'eval_steps': self.EVAL_STEPS,
            'eval_strategy': self.EVALUATION_STRATEGY,
            'save_steps': self.SAVE_STEPS,
            'save_strategy': self.SAVE_STRATEGY,
            'save_total_limit': self.SAVE_TOTAL_LIMIT,
            'load_best_model_at_end': self.LOAD_BEST_MODEL_AT_END,
            'metric_for_best_model': self.METRIC_FOR_BEST_MODEL,
            'greater_is_better': self.GREATER_IS_BETTER,
            'dataloader_num_workers': self.DATALOADER_NUM_WORKERS,
            'dataloader_pin_memory': self.DATALOADER_PIN_MEMORY,
            'auto_find_batch_size': self.AUTO_FIND_BATCH_SIZE,
            'label_smoothing_factor': self.LABEL_SMOOTHING,
            'report_to': [],
            'run_name': f'hate_speech_{self.MODEL_NAME.split("/")[-1]}',
        }

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'num_labels': self.NUM_LABELS,
            'hidden_dropout_prob': self.DROPOUT_RATE,
            'attention_probs_dropout_prob': self.ATTENTION_DROPOUT,
        }

    def cleanup_memory(self):
        """🔥 Aggressive memory cleanup"""
        if self.USE_CUDA and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        # Using global logger
        logger.info("Memory cleanup performed.")

    def save_config(self, filepath: str = None):
        """Save optimized configuration"""
        if filepath is None:
            filepath = os.path.join(self.OUTPUT_DIR, 'optimized_config.json')

        # --- Validate filepath ---
        if not isinstance(filepath, str) or not filepath.strip():
            # Using global logger
            logger.error(f"Invalid filepath provided for saving config: '{filepath}'. Cannot save.")
            return None
        # --- End Validation ---

        config_dict = {}
        for key, value in self.__dict__.items():
            # Do not save logger if present, although it should not be
            if not key.startswith('_') and key != 'logger':
                try:
                    json.dumps(value)
                    config_dict[key] = value
                except (TypeError, ValueError):
                    # Convert non-serializable objects to string
                    config_dict[key] = str(value)

        config_dict['_optimization_version'] = '2.0'
        config_dict['_created_at'] = str(datetime.now().isoformat())

        try:
            with open(filepath, 'w', encoding='utf-8') as f: # Ensure UTF-8 encoding
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            # Using global logger
            logger.info(f"💾 Optimized config saved: {filepath}")
            return filepath
        except Exception as e:
            # Using global logger
            logger.error(f"❌ Error saving config to {filepath}: {str(e)}")
            return None

    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from file"""
        if not isinstance(filepath, str) or not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found or invalid path: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f: # Ensure UTF-8 encoding
            config_dict = json.load(f)

        config_dict.pop('_optimization_version', None)
        config_dict.pop('_created_at', None)
        config_dict.pop('logger', None) # Ensure logger is not reloaded

        return cls(**config_dict)

    def update_config(self, **kwargs):
        """Update configuration with validation"""
        updated_params = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                updated_params.append(f"{key}: {old_value} → {value}")
            else:
                # Using global logger
                logger.warning(f"⚠️ Unknown parameter: {key}")

        if updated_params:
            # Using global logger
            logger.info("✅ Updated parameters:")
            for param in updated_params:
                # Using global logger
                logger.info(f"   - {param}")

            hardware_params = ['USE_CUDA', 'FP16', 'DATALOADER_NUM_WORKERS']
            if any(param.split(':')[0].strip() in hardware_params for param in updated_params):
                self._apply_all_optimizations()


# 🔥 Factory function for different use cases
def create_config_for_use_case(use_case: str = "default", **kwargs) -> Config:
    """
    Factory function to create optimized configs for different scenarios

    Args:
        use_case: 'default', 'small_gpu', 'large_dataset', 'production', 'experimentation'
    """

    if use_case == "small_gpu":
        return Config(
            BATCH_SIZE=8,
            GRADIENT_ACCUMULATION_STEPS=4,
            MAX_LENGTH=96,
            FP16=True,
            GRADIENT_CHECKPOINTING=True,
            DATALOADER_NUM_WORKERS=2,
            **kwargs
        )

    elif use_case == "large_dataset":
        return Config(
            BATCH_SIZE=32,
            GRADIENT_ACCUMULATION_STEPS=1,
            NUM_EPOCHS=2,
            EVAL_STEPS=1000,
            SAVE_STEPS=1000,
            EARLY_STOPPING_PATIENCE=2,
            **kwargs
        )

    elif use_case == "production":
        return Config(
            BATCH_SIZE=16,
            NUM_EPOCHS=3,
            EARLY_STOPPING_PATIENCE=2,
            SAVE_TOTAL_LIMIT=1,
            ENABLE_TORCH_COMPILE=True,
            LABEL_SMOOTHING=0.1,
            ENVIRONMENT="production", # Set environment for production
            **kwargs
        )

    elif use_case == "experimentation":
        return Config(
            BATCH_SIZE=16,
            NUM_EPOCHS=2,
            MAX_LENGTH=96,
            LOGGING_STEPS=25,
            EVAL_STEPS=100,
            SAVE_STEPS=500,
            ENVIRONMENT="development", # Set environment for experimentation
            **kwargs
        )

    else:  # default
        return Config(**kwargs)