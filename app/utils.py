import os
import json
import torch
import numpy as np
import shutil
import glob
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import Counter # Import Counter

# --- ENSURE THESE IMPORTS ARE AT THE TOP OF THE FILE AND ARE ABSOLUTE ---
from .dependencies import logger, device # Ensure 'logger' and 'device' are imported from dependencies
from .config import Config # Required for save_model_artifacts
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
# --- END IMPORTS ---

def save_model_artifacts(model, tokenizer, config: Config, metrics: Dict, save_dir: str):
    """Save model, tokenizer, config, and metrics with error handling"""
    try:
        # --- Validate save_dir ---
        if not isinstance(save_dir, str) or not save_dir:
            logger.error(f"Invalid save_dir provided: {save_dir}. Must be a non-empty string.")
            return False
        # --- End Validation ---

        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"💾 Saving model artifacts to {save_dir}...")

        # Save model and tokenizer
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info("✅ Model and tokenizer saved")

        # Save config
        config_path = os.path.join(save_dir, 'optimized_config.json') # Consistent config file name
        config.save_config(config_path) # Assumes config has a save_config method
        logger.info("✅ Config saved")

        # Save metrics
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f: # Ensure UTF-8 encoding
            # Convert numpy types to native Python types for JSON serialization
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = value.item()
                elif isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                else:
                    serializable_metrics[key] = value

            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        logger.info("✅ Metrics saved")

        # Save model info
        model_info = {
            'model_name': config.MODEL_NAME,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'save_timestamp': datetime.now().isoformat(),
            'device': str(next(model.parameters()).device),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        }

        with open(os.path.join(save_dir, 'model_info.json'), 'w', encoding='utf-8') as f: # Ensure UTF-8 encoding
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        logger.info("✅ Model info saved")

        logger.info(f"Model artifacts saved to {save_dir}")
        return True

    except Exception as e:
        logger.error(f"Error saving model artifacts to {save_dir}: {str(e)}", exc_info=True)
        return False

def load_model_artifacts(load_dir: str):
    """Load model, tokenizer, and artifacts with error handling"""
    try:
        # --- Validate load_dir ---
        if not isinstance(load_dir, str) or not load_dir:
            error_msg = f"Invalid load_dir provided: {load_dir}. Must be a non-empty string."
            logger.error(error_msg)
            raise ValueError(error_msg)
        # --- End Validation ---

        if not os.path.exists(load_dir):
            error_msg = f"Directory not found: {load_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Attempting to load model artifacts from: {load_dir}")

        model = AutoModelForSequenceClassification.from_pretrained(load_dir)
        tokenizer = AutoTokenizer.from_pretrained(load_dir)
        logger.info("✅ Model and tokenizer loaded")

        metrics_path = os.path.join(load_dir, 'metrics.json')
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f: # Ensure UTF-8 encoding
                metrics = json.load(f)
            logger.info("✅ Metrics loaded")
        else:
            logger.warning(f"Metrics file not found at {metrics_path}. Skipping metrics load.")

        model_info_path = os.path.join(load_dir, 'model_info.json')
        model_info = {}
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r', encoding='utf-8') as f: # Ensure UTF-8 encoding
                model_info = json.load(f)
            logger.info("✅ Model info loaded")
        else:
            logger.warning(f"Model info file not found at {model_info_path}. Skipping model info load.")

        logger.info(f"Model artifacts loaded from {load_dir}")
        return model, tokenizer, metrics, model_info

    except Exception as e:
        logger.error(f"Error loading model artifacts from {load_dir}: {str(e)}", exc_info=True)
        raise

def calculate_class_weights(labels: List[int], method: str = 'balanced') -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets"""
    try:
        from sklearn.utils.class_weight import compute_class_weight

        unique_labels = np.unique(labels)

        if method == 'balanced':
            class_weights = compute_class_weight(
                'balanced',
                classes=unique_labels,
                y=labels
            )
        else:
            # Simple inverse frequency
            label_counts = Counter(labels)
            total_samples = len(labels)
            class_weights = [total_samples / (len(unique_labels) * label_counts[label])
                           for label in unique_labels]

        weight_dict = {int(label): float(weight) for label, weight in zip(unique_labels, class_weights)}

        logger.info(f"📊 Class weights calculated ({method}): {weight_dict}")

        return weight_dict

    except Exception as e:
        logger.error(f"Error calculating class weights: {str(e)}", exc_info=True)
        return {0: 1.0, 1: 1.0}  # Default equal weights

def get_model_summary(model) -> Dict[str, Any]:
    """Get comprehensive model summary"""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

        summary = {
            'model_class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'model_size_mb': model_size,
            'device': str(next(model.parameters()).device),
            'dtype': str(next(model.parameters()).dtype)
        }

        return summary

    except Exception as e:
        logger.error(f"Error getting model summary: {str(e)}", exc_info=True)
        return {}

def print_model_summary(model):
    """Print formatted model summary (for debugging/CLI, not for API)"""
    try:
        summary = get_model_summary(model)

        print("\n🤖 Model Summary:")
        print("="*50)
        print(f"📝 Model Class: {summary.get('model_class', 'Unknown')}")
        print(f"📊 Total Parameters: {summary.get('total_parameters', 0):,}")
        print(f"🎯 Trainable Parameters: {summary.get('trainable_parameters', 0):,}")
        print(f"❄️  Frozen Parameters: {summary.get('frozen_parameters', 0):,}")
        print(f"💾 Model Size: {summary.get('model_size_mb', 0):.2f} MB")
        print(f"🖥️  Device: {summary.get('device', 'Unknown')}")
        print(f"🔢 Data Type: {summary.get('dtype', 'Unknown')}")
        print("="*50)

    except Exception as e:
        logger.error(f"❌ Error printing model summary: {e}", exc_info=True)

def setup_optimizer_and_scheduler(model, config: Config, num_training_steps: int):
    """Setup optimizer and learning rate scheduler (for training, not for API)"""
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            eps=1e-8
        )

        scheduler = None
        if config.WARMUP_STEPS > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.WARMUP_STEPS,
                num_training_steps=num_training_steps
            )
            logger.info(f"✅ Scheduler setup: Linear with warmup (Warmup steps: {config.WARMUP_STEPS}, Total training steps: {num_training_steps})")

        logger.info(f"✅ Optimizer setup: AdamW (Learning rate: {config.LEARNING_RATE}, Weight decay: {config.WEIGHT_DECAY})")

        return optimizer, scheduler

    except Exception as e:
        logger.error(f"Error setting up optimizer and scheduler: {str(e)}", exc_info=True)
        raise

def validate_gpu_setup():
    """Validate GPU setup and print information (for debugging/CLI, not for API)"""
    print("\n🔍 GPU Setup Validation:")
    print("="*50)

    cuda_available = torch.cuda.is_available()
    print(f"🖥️  CUDA Available: {cuda_available}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"🔢 GPU Count: {gpu_count}")

        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        print(f"🎯 Current GPU: {current_gpu} ({gpu_name})")

        total_memory = torch.cuda.get_device_properties(current_gpu).total_memory
        allocated_memory = torch.cuda.memory_allocated(current_gpu)
        reserved_memory = torch.cuda.memory_reserved(current_gpu)

        print(f"💾 GPU Memory:")
        print(f"   - Total: {total_memory / 1024**3:.2f} GB")
        print(f"   - Allocated: {allocated_memory / 1024**2:.2f} MB")
        print(f"   - Reserved: {reserved_memory / 1024**2:.2f} MB")
        print(f"   - Free: {(total_memory - reserved_memory) / 1024**3:.2f} GB")

        major, minor = torch.cuda.get_device_capability(current_gpu)
        print(f"⚡ Compute Capability: {major}.{minor}")

        if major >= 7:
            print("🚀 Tensor Core Support: Available")
        else:
            print("⚠️ Tensor Core Support: Not available")

    else:
        print("❌ No CUDA GPUs available")
        print("💻 Will use CPU for training")

    print("="*50)

    return cuda_available

def create_training_directories(config: Config):
    """Create all necessary directories for training (for training, not for API)"""
    try:
        directories = [
            config.OUTPUT_DIR,
            config.LOGS_DIR,
            config.PLOTS_DIR,
            config.PLOT_SAVE_PATH,
            os.path.join(config.OUTPUT_DIR, 'checkpoints'),
            os.path.join(config.OUTPUT_DIR, 'best_model'),
            os.path.join(config.LOGS_DIR, 'tensorboard')
        ]

        for directory in directories:
            if not isinstance(directory, str) or not directory:
                logger.error(f"Invalid directory path encountered: {directory}. Skipping creation.")
                continue
            os.makedirs(directory, exist_ok=True)

        logger.info("📁 Training directories created:")
        for directory in directories:
            logger.info(f"   - {directory}")

        return True

    except Exception as e:
        logger.error(f"Error creating training directories: {str(e)}", exc_info=True)
        return False

def cleanup_training_artifacts(config: Config, keep_best_model: bool = True):
    """Clean up training artifacts (for training, not for API)"""
    try:
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, 'checkpoints')
        if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
            logger.warning(f"Invalid checkpoint_dir path: {checkpoint_dir}. Skipping cleanup.")
            return False

        if os.path.exists(checkpoint_dir):
            if keep_best_model:
                best_model_files = []
                for file in os.listdir(checkpoint_dir):
                    if 'best' in file.lower():
                        best_model_files.append(file)

                for file in os.listdir(checkpoint_dir):
                    if file not in best_model_files:
                        file_path = os.path.join(checkpoint_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
            else:
                shutil.rmtree(checkpoint_dir)

        temp_patterns = ['*.tmp', '*.temp', '*~']
        for pattern in temp_patterns:
            if not isinstance(config.OUTPUT_DIR, str) or not config.OUTPUT_DIR:
                logger.warning(f"Invalid config.OUTPUT_DIR path: {config.OUTPUT_DIR}. Skipping temporary file cleanup.")
                break
            for file in glob.glob(os.path.join(config.OUTPUT_DIR, pattern)):
                os.remove(file)

        logger.info("🧹 Training artifacts cleaned up")
        return True

    except Exception as e:
        logger.error(f"Error cleaning up training artifacts: {str(e)}", exc_info=True)
        return False