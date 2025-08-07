import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fsspec.registry import default
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, LSTM, \
    Bidirectional, TimeDistributed, Input, Concatenate, Lambda, Layer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.saving import register_keras_serializable
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import functools
import traceback
import sys
import time
import pickle
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from joblib import Parallel, delayed
from dataclasses import dataclass
import json
import os
import gc
import time
import numpy as np
import tensorflow as tf
from typing import Optional, List, Generator, Tuple
import psutil
import gc
from dataclasses import dataclass
from typing import Optional, List, Generator, Tuple

# Define constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
SEQUENCE_LENGTH = 16
BATCH_SIZE = 8
EPOCHS = 1
NUM_CLASSES = 4  # normal, moderate, dense, risky
data_path = r"C:\Users\rnidh\stampedeDetection\output\Stampede_detection_dataset"


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Stampede Detection')
    parser.add_argument('--test-only', action='store_true', help='Only test a video with a pre-trained model')
    parser.add_argument('--model-path', type=str, default='enhanced_stampede_detection_checkpoint.keras',
                        help='Path to pre-trained model or checkpoint')
    parser.add_argument('--video-path', type=str,
                        default=r"C:\Users\rnidh\Downloads\Untitled video - Made with Clipchamp (2).mp4",
                        help='Path to test video')
    parser.add_argument('--use-enhanced', action='store_true', default=True,
                        help='Use the enhanced model with additional features')
    parser.add_argument('--continue-training', action='store_true', default=True,
                        help='Continue training from checkpoint if model exists')
    return parser.parse_args()

@dataclass
class ModelConfig:
    # Model architecture
    image_height: int = 224
    image_width: int = 224
    sequence_length: int = 16
    num_classes: int = 4

    # Training parameters
    batch_size: int = 4  # Reduced from 8
    epochs: int = 1
    learning_rate: float = 0.001

    # Memory management
    gpu_memory_limit_mb: int = 4096
    max_memory_mb: int = 3072
    dynamic_batch_sizing: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 8

    # Streaming parameters
    chunk_size: int = 50
    max_sequences_predict: int = 15
    max_frames_video: int = 150

    # Memory monitoring
    memory_check_interval: int = 5
    max_memory_usage_percent: float = 80.0

    def get_available_memory_mb(self) -> float:
        """Get available system memory"""
        return psutil.virtual_memory().available / (1024 * 1024)

    def get_memory_usage_percent(self) -> float:
        """current memory usage percentage"""
        return psutil.virtual_memory().percent


class AdvancedMemoryManager:
    """Advanced memory management with real-time monitoring"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.memory_history = []
        self.setup_tensorflow()

    def setup_tensorflow(self):
        # Disable eager execution to reduce memory usage
        tf.config.run_functions_eagerly(False)

        # Configure GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=self.config.gpu_memory_limit_mb
                        )]
                    )
                print(f"GPU configured with {self.config.gpu_memory_limit_mb}MB limit")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")

        # Configure CPU parallelism
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    def get_memory_info(self) -> dict:
        """comprehensive memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_mb': memory.total / (1024 * 1024),
            'available_mb': memory.available / (1024 * 1024),
            'used_mb': memory.used / (1024 * 1024),
            'percent': memory.percent
        }

    def calculate_safe_batch_size(self, data_size: int, sample_shape: tuple) -> int:
        """Calculate safe batch size based on memory constraints"""
        memory_info = self.get_memory_info()
        available_mb = memory_info['available_mb']

        # Calculate memory needed per sample
        sample_size_mb = np.prod(sample_shape) * 4 / (1024 * 1024)  # 4 bytes per float32

        # Use 60% of available memory for safety
        safe_memory_mb = available_mb * 0.6

        # Calculate maximum batch size
        max_batch_size = int(safe_memory_mb / sample_size_mb)

        # Clamp to configured limits
        batch_size = max(
            self.config.min_batch_size,
            min(max_batch_size, self.config.max_batch_size)
        )

        print(f"Memory: {available_mb:.1f}MB available, "
              f"Sample: {sample_size_mb:.2f}MB, "
              f"Batch size: {batch_size}")

        return batch_size

    def aggressive_cleanup(self):
        # Clear TensorFlow session
        tf.keras.backend.clear_session()

        # Force garbage collection
        gc.collect()

        # Additional cleanup for numpy arrays
        if hasattr(np, 'clear_cache'):
            np.clear_cache()

        time.sleep(0.1)

    def monitor_memory_usage(self, operation_name: str = ""):
        """Monitor and log memory usage"""
        memory_info = self.get_memory_info()
        self.memory_history.append(memory_info)

        if memory_info['percent'] > self.config.max_memory_usage_percent:
            print(f"WARNING: High memory usage {memory_info['percent']:.1f}% during {operation_name}")
            self.aggressive_cleanup()

            # Check after cleanup
            new_memory_info = self.get_memory_info()
            print(f"After cleanup: {new_memory_info['percent']:.1f}% memory usage")


class GPUManager:
    """Manages GPU memory with 4GB limit and monitoring"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.setup_gpu()

    def setup_gpu(self):
        """Configure GPU with 4GB memory limit"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    # Set memory limit to 4GB
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=self.config.gpu_memory_limit_mb
                        )]
                    )
                print(f"GPU configured with {self.config.gpu_memory_limit_mb}MB memory limit")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU found, using CPU")

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / 1024 / 1024  # Convert to MB
        except:
            return 0.0

    def cleanup_memory(self):
        """Force cleanup"""
        tf.keras.backend.clear_session()
        gc.collect()


class DynamicBatchManager:
    """Manage batch sizes based on available memory"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.current_batch_size = config.batch_size
        self.memory_usage_history = []
        self.gpu_manager = GPUManager(config)

    def calculate_optimal_batch_size(self, data_size: int) -> int:
        """Calculate optimal batch size"""
        memory_usage = self.gpu_manager.get_memory_usage()
        memory_threshold = self.config.gpu_memory_limit_mb * 0.8

        if memory_usage > memory_threshold:
            # Reduce batch size if memory usage is high
            self.current_batch_size = max(
                self.config.min_batch_size,
                self.current_batch_size // 2
            )
        elif memory_usage < memory_threshold * 0.5:
            # Increase batch size if memory usage is low
            self.current_batch_size = min(
                self.config.max_batch_size,
                self.current_batch_size * 2
            )

        # Don't exceed data size
        self.current_batch_size = min(self.current_batch_size, data_size)

        print(f"Dynamic batch size: {self.current_batch_size} (Memory: {memory_usage:.1f}MB)")
        return self.current_batch_size

    def process_in_batches(self, data: np.ndarray, process_func) -> List:
        """dynamic batches"""
        results = []
        batch_size = self.calculate_optimal_batch_size(len(data))

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_result = process_func(batch)
            results.append(batch_result)
            self.gpu_manager.cleanup_memory()
            # Recalculate batch size based on current memory
            batch_size = self.calculate_optimal_batch_size(len(data) - i - batch_size)

        return results


class StreamingDataLoader:
    """Memory-efficient data loading with streaming and caching"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.cache = {}
        self.processing_times = []

    def load_sequences_streaming(self, base_dir: str, categories: List[str]) -> Generator:
        """Stream sequences without loading everything into memory"""

        for category_idx, category in enumerate(categories):
            category_path = os.path.join(base_dir, category)

            if not os.path.exists(category_path):
                continue
            frames = self._get_frame_list(category_path)
            chunk_size = self.config.chunk_size

            for chunk_start in range(0, len(frames), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(frames))
                chunk_frames = frames[chunk_start:chunk_end]

                for sequence_data in self._process_frame_chunk(
                        category_path, chunk_frames, category_idx
                ):
                    yield sequence_data
                gc.collect()

    def _process_frame_chunk(self, category_path: str, frames: List[str], category_idx: int) -> Generator:
        """Process a chunk of frames into sequences"""

        loaded_frames = []
        for frame_file in frames:
            frame_path = os.path.join(category_path, frame_file)
            img = cv2.imread(frame_path)
            if img is not None:
                loaded_frames.append(cv2.resize(img, (self.config.image_width, self.config.image_height)))

        #sliding window
        step_size = self.config.sequence_length // 2

        for i in range(0, len(loaded_frames) - self.config.sequence_length + 1, step_size):
            sequence_frames = loaded_frames[i:i + self.config.sequence_length]

            flow_sequence = self._frames_to_flow(sequence_frames)

            yield {
                'flow': np.array(flow_sequence),
                'original': np.array(sequence_frames),
                'label': category_idx
            }

    def _frames_to_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        flow_sequence = []

        for frame in frames:
            # Extract optical flow from BGR channels
            flow = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)
            flow[..., 0] = frame[..., 2] / 255.0 * 2 - 1  # R channel
            flow[..., 1] = frame[..., 1] / 255.0 * 2 - 1  # G channel
            flow_sequence.append(flow)

        return flow_sequence

    def _get_frame_list(self, category_path: str) -> List[str]:
        frames = []

        # Check for direct frames
        direct_frames = [f for f in os.listdir(category_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.tif'))]

        if direct_frames:
            frames.extend(sorted(direct_frames))
        else:
            # Check subfolders
            for subfolder in os.listdir(category_path):
                subfolder_path = os.path.join(category_path, subfolder)
                if os.path.isdir(subfolder_path):
                    sub_frames = [os.path.join(subfolder, f)
                                  for f in os.listdir(subfolder_path)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.tif'))]
                    frames.extend(sorted(sub_frames))

        return frames


class MemoryEfficientFeatureCalculator:

    def __init__(self, config: ModelConfig):
        self.config = config
        self.batch_manager = DynamicBatchManager(config)

    def calculate_features_streaming(self, sequence_generator: Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate features in streaming fashion"""

        flow_sequences = []
        original_sequences = []
        labels = []
        chunk_buffer = []

        for sequence_data in sequence_generator:
            chunk_buffer.append(sequence_data)

            if len(chunk_buffer) >= self.config.chunk_size:
                # Process chunk
                chunk_results = self._process_sequence_chunk(chunk_buffer)

                flow_sequences.extend(chunk_results['flow'])
                original_sequences.extend(chunk_results['original'])
                labels.extend(chunk_results['labels'])
                chunk_buffer = []
                gc.collect()


        if chunk_buffer:
            chunk_results = self._process_sequence_chunk(chunk_buffer)
            flow_sequences.extend(chunk_results['flow'])
            original_sequences.extend(chunk_results['original'])
            labels.extend(chunk_results['labels'])

        # Convert to arrays
        X = np.array(flow_sequences) if flow_sequences else np.array([])
        original_frames = np.array(original_sequences) if original_sequences else np.array([])
        y = np.array(labels) if labels else np.array([])

        # additional features
        if len(X) > 0:
            scalar_features = self._calculate_scalar_features_batched(X, original_frames)
            binary_features = self._quantize_features_batched(scalar_features)
            return X, binary_features, y

        return X, None, y

    def _process_sequence_chunk(self, chunk_buffer: List) -> dict:
        """Process a chunk of sequences"""
        results = {
            'flow': [],
            'original': [],
            'labels': []
        }

        for sequence_data in chunk_buffer:
            results['flow'].append(sequence_data['flow'])
            results['original'].append(sequence_data['original'])
            results['labels'].append(sequence_data['label'])

        return results

    def _calculate_scalar_features_batched(self, X: np.ndarray, original_frames: np.ndarray) -> np.ndarray:

        def calculate_batch_features(batch_X, batch_orig):
            # feature calculation in batches
            flow_acceleration = calculate_flow_acceleration(batch_X)
            flow_divergence = calculate_flow_divergence(batch_X)
            scene_changes = calculate_scene_changes(batch_orig) if batch_orig is not None else \
                np.zeros((len(batch_X), self.config.sequence_length))
            motion_entropy = calculate_motion_entropy(batch_X)
            motion_patterns = segment_motion_patterns(batch_X)

            # Calculate behavior features
            behavior_features = []
            for sequence in batch_X:
                seq_behaviors = []
                for flow in sequence:
                    behavior_hist = classify_region_behavior(flow)
                    seq_behaviors.append(behavior_hist)
                behavior_features.append(np.array(seq_behaviors))
            behavior_features = np.array(behavior_features)

            # Calculate density features
            density_features, heatmap_features = self._calculate_density_features_optimized(batch_X)

            # Combine features
            return np.concatenate([
                flow_acceleration[..., None],
                flow_divergence[..., None],
                scene_changes[..., None],
                motion_entropy[..., None],
                motion_patterns[..., None],
                behavior_features,
                density_features,
                heatmap_features
            ], axis=2)

        results = self.batch_manager.process_in_batches(
            np.arange(len(X)),
            lambda indices: calculate_batch_features(X[indices],
                                                     original_frames[indices] if original_frames is not None else None)
        )

        return np.vstack(results)

    def _calculate_density_features_optimized(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        density_features = []
        heatmap_features = []

        for sequence in sequences:
            seq_density = []
            seq_heatmaps = []

            for flow in sequence:
                # density calculation for speed
                density_score, cluster_count = self._fast_density_estimation(flow)
                seq_density.append([density_score, cluster_count])

                # heatmap calculation
                heatmap = self._fast_heatmap_generation(flow)
                seq_heatmaps.append(heatmap.flatten())

            density_features.append(np.array(seq_density))
            seq_heatmaps_array = np.array(seq_heatmaps)
            # Average heatmap across sequence
            avg_heatmap = np.mean(seq_heatmaps_array, axis=0)
            heatmap_features.append(
                avg_heatmap.repeat(self.config.sequence_length).reshape(self.config.sequence_length, -1))

        return np.array(density_features), np.array(heatmap_features)

    def _fast_density_estimation(self, flow_frame: np.ndarray) -> Tuple[float, int]:
        u = flow_frame[..., 0].flatten()
        v = flow_frame[..., 1].flatten()

        # Downsample aggressively for speed
        if len(u) > 500:
            idx = np.random.choice(len(u), 500, replace=False)
            u, v = u[idx], v[idx]

        magnitude = np.mean(np.sqrt(u ** 2 + v ** 2))

        density_score = min(magnitude * 2, 1.0)

        return density_score, 0

    def _fast_heatmap_generation(self, flow_frame: np.ndarray) -> np.ndarray:
        grid_size = 4
        height, width = flow_frame.shape[:2]
        cell_height = height // grid_size
        cell_width = width // grid_size

        heatmap = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_height
                y_end = (i + 1) * cell_height
                x_start = j * cell_width
                x_end = (j + 1) * cell_width

                cell_flow = flow_frame[y_start:y_end, x_start:x_end]
                density_score, _ = self._fast_density_estimation(cell_flow)
                heatmap[i, j] = density_score

        heatmap_resized = cv2.resize(heatmap, (8, 8), interpolation=cv2.INTER_LINEAR)
        return heatmap_resized

    def _quantize_features_batched(self, scalar_features: np.ndarray) -> np.ndarray:
        batch_size = self.config.batch_size * 4

        results = []
        for i in range(0, len(scalar_features), batch_size):
            batch = scalar_features[i:i + batch_size]
            batch_result = quantize_and_binarize_scalar_features(batch)
            results.append(batch_result)

            # Cleanup
            gc.collect()

        return np.vstack(results)


def load_optical_flow_data_optimized(base_dir: str, categories: List[str], config: ModelConfig) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:

    data_loader = StreamingDataLoader(config)
    feature_calculator = MemoryEfficientFeatureCalculator(config)

    print(f"Loading data from: {base_dir}")
    print("Using memory-efficient streaming approach...")

    # Create streaming generator
    sequence_generator = data_loader.load_sequences_streaming(base_dir, categories)

    X, binary_features, y = feature_calculator.calculate_features_streaming(sequence_generator)

    print(f"Loaded {len(X)} sequences with optimized memory usage")

    return X, binary_features, y


def train_enhanced_model_memory_optimized(X_flow: np.ndarray, X_scalar: np.ndarray, y: np.ndarray,
                                          config: ModelConfig,
                                          model_path: str = 'enhanced_stampede_detection_checkpoint.keras',
                                          continue_training: bool = True, initial_epoch: int = 0):
    # memory manager
    memory_manager = AdvancedMemoryManager(config)

    print(f"Starting training with {len(X_flow)} sequences")
    memory_manager.monitor_memory_usage("training_start")

    # safe batch size
    sample_shape = (config.sequence_length, config.image_height, config.image_width, 2)
    safe_batch_size = memory_manager.calculate_safe_batch_size(len(X_flow), sample_shape)

    # one-hot encoding
    y_onehot = to_categorical(y, num_classes=config.num_classes)

    # Split data
    indices = np.arange(len(X_flow))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)

    # indexing
    print(f"Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")

    # Load or create model
    if os.path.exists(model_path) and continue_training:
        model, loaded = load_model_safely(model_path, config)
        if not loaded:
            model = create_enhanced_cnn_lstm_model_optimized(config)
            initial_epoch = 0
    else:
        model = create_enhanced_cnn_lstm_model_optimized(config)
        initial_epoch = 0

    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    for epoch in range(initial_epoch, config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        memory_manager.monitor_memory_usage(f"epoch_{epoch}")

        # Training phase with batch processing
        train_loss = 0.0
        train_acc = 0.0
        num_train_batches = 0

        for i in range(0, len(train_idx), safe_batch_size):
            batch_indices = train_idx[i:i + safe_batch_size]

            # Get data
            batch_X_flow = X_flow[batch_indices]
            batch_X_scalar = X_scalar[batch_indices]
            batch_y = y_onehot[batch_indices]

            # Train
            batch_history = model.train_on_batch(
                [batch_X_flow, batch_X_scalar],
                batch_y
            )

            train_loss += batch_history[0]
            train_acc += batch_history[1]
            num_train_batches += 1

            del batch_X_flow, batch_X_scalar, batch_y
            if num_train_batches % config.memory_check_interval == 0:
                memory_manager.aggressive_cleanup()
                memory_manager.monitor_memory_usage(f"train_batch_{num_train_batches}")

        train_loss /= num_train_batches
        train_acc /= num_train_batches

        val_loss = 0.0
        val_acc = 0.0
        num_val_batches = 0

        for i in range(0, len(val_idx), safe_batch_size):
            batch_indices = val_idx[i:i + safe_batch_size]

            batch_X_flow = X_flow[batch_indices]
            batch_X_scalar = X_scalar[batch_indices]
            batch_y = y_onehot[batch_indices]

            # Validate
            batch_val_history = model.test_on_batch(
                [batch_X_flow, batch_X_scalar],
                batch_y
            )

            val_loss += batch_val_history[0]
            val_acc += batch_val_history[1]
            num_val_batches += 1

            del batch_X_flow, batch_X_scalar, batch_y

        val_loss /= num_val_batches
        val_acc /= num_val_batches

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"Loss: {train_loss:.4f} - Accuracy: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}")

        # CHECKPOINT SAVING WITH ERROR HANDLING
        max_save_attempts = 3
        save_successful = False

        for attempt in range(max_save_attempts):
            try:
                # directory exists
                checkpoint_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "."
                os.makedirs(checkpoint_dir, exist_ok=True)

                # temporary file first to avoid corruption
                temp_model_path = model_path + f".tmp_{epoch}_{attempt}"

                # Save to temporary file
                model.save(temp_model_path, overwrite=True, save_format='tf')

                # move to final location
                if os.path.exists(model_path):
                    backup_path = model_path + ".backup"
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(model_path, backup_path)

                os.rename(temp_model_path, model_path)
                print(f"Checkpoint saved successfully to {model_path}")
                save_successful = True
                break

            except Exception as e:
                print(f"Checkpoint save attempt {attempt + 1} failed: {e}")

                # Clean up temporary file
                if os.path.exists(temp_model_path):
                    try:
                        os.remove(temp_model_path)
                    except:
                        pass

                if attempt < max_save_attempts - 1:
                    print(f"Retrying checkpoint save in 2 seconds...")
                    time.sleep(2)
                    gc.collect()

        if not save_successful:
            print(f"Warning: Failed to save checkpoint after {max_save_attempts} attempts. Continuing training...")

        memory_manager.aggressive_cleanup()

    return model, history, config.epochs


def load_model_safely(model_path: str, config: ModelConfig):
    """
    Safely load a model with custom layers and multiple fallback attempts
    """
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            # Check if backup exists and main file is corrupted
            if attempt > 0 and os.path.exists(model_path + ".backup"):
                print(f"Attempt {attempt + 1}: Trying backup file...")
                load_path = model_path + ".backup"
            else:
                load_path = model_path

            custom_objects = {
                'FeatureWiseAttention': FeatureWiseAttention
            }

            model = tf.keras.models.load_model(load_path, custom_objects=custom_objects)

            # Recompile model
            model.compile(
                optimizer=Adam(learning_rate=config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            print(f"Model loaded successfully from {load_path}")
            return model, True

        except Exception as e:
            print(f"Model loading attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                print("Retrying model load...")
                time.sleep(1)
                gc.collect()

    print("All model loading attempts failed. Creating new model...")
    return None, False


def create_enhanced_cnn_lstm_model_optimized(config: ModelConfig):
    flow_input_shape = (config.sequence_length, config.image_height, config.image_width, 2)
    scalar_features_shape = (config.sequence_length, 624)

    # Model architecture
    flow_input = Input(shape=flow_input_shape, name='optical_flow_input')
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(flow_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)

    scalar_input = Input(shape=scalar_features_shape, name='scalar_features_input')
    attention_output = FeatureWiseAttention()(scalar_input)

    combined = Concatenate(axis=2)([x, attention_output])

    bilstm = Bidirectional(LSTM(256, return_sequences=True), merge_mode='concat')(combined)
    dropout1 = Dropout(0.3)(bilstm)
    bilstm2 = Bidirectional(LSTM(128), merge_mode='concat')(dropout1)
    dropout2 = Dropout(0.3)(bilstm2)

    dense1 = Dense(64, activation='relu')(dropout2)
    bn = BatchNormalization()(dense1)
    output = Dense(config.num_classes, activation='softmax')(bn)

    model = Model(inputs=[flow_input, scalar_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model





def handle_errors(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\nExecution interrupted by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()

    return wrapper


def calculate_flow_acceleration(flow_sequences):
    accelerations = []

    for sequence in flow_sequences:
        seq_acceleration = []
        for i in range(1, len(sequence)):
            accel = sequence[i] - sequence[i - 1]
            accel_magnitude = np.sqrt(np.sum(accel ** 2, axis=2))
            mean_accel = np.mean(accel_magnitude)
            seq_acceleration.append(mean_accel)

        # Pad the sequence to maintain sequence length
        seq_acceleration.insert(0, 0)
        accelerations.append(np.array(seq_acceleration))

    return np.array(accelerations)


def calculate_flow_divergence(flow_sequences):
    divergences = []

    for sequence in flow_sequences:
        seq_divergence = []
        for flow in sequence:
            dx = cv2.Sobel(flow[..., 0], cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(flow[..., 1], cv2.CV_64F, 0, 1, ksize=3)

            divergence = dx + dy

            mean_divergence = np.mean(np.abs(divergence))
            seq_divergence.append(mean_divergence)

        divergences.append(np.array(seq_divergence))

    return np.array(divergences)


def calculate_scene_changes(original_frames):

    ssim_scores = []

    for sequence in original_frames:
        seq_ssim = []
        for i in range(1, len(sequence)):

            if len(sequence[i].shape) == 3:
                frame1 = cv2.cvtColor(sequence[i - 1], cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(sequence[i], cv2.COLOR_BGR2GRAY)
            else:
                frame1 = sequence[i - 1]
                frame2 = sequence[i]

            # Calculate SSIM between consecutive frames
            score = ssim(frame1, frame2, data_range=frame2.max() - frame2.min())

            change_score = 1.0 - score
            seq_ssim.append(change_score)

        # Pad for first frame
        seq_ssim.insert(0, 0)
        ssim_scores.append(np.array(seq_ssim))

    return np.array(ssim_scores)


def calculate_motion_entropy(flow_sequences):

    entropies = []

    for sequence in flow_sequences:
        seq_entropy = []
        for flow in sequence:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            hist, _ = np.histogram(mag, bins=32, range=(0, np.max(mag) if np.max(mag) > 0 else 1))

            hist = hist / (np.sum(hist) if np.sum(hist) > 0 else 1)

            flow_entropy = entropy(hist, base=2)
            seq_entropy.append(flow_entropy)

        entropies.append(np.array(seq_entropy))

    return np.array(entropies)


def segment_motion_patterns(flow_sequences):
    behavior_features = []

    for sequence in flow_sequences:
        seq_features = []
        for flow in sequence:
            # grid of particles
            particles = np.stack(np.meshgrid(np.linspace(0, flow.shape[0] - 1, 20),
                                             np.linspace(0, flow.shape[1] - 1, 20)), axis=-1).reshape(-1, 2)

            # Advect particles using optical flow
            displacements = []
            for p in particles:
                x, y = int(p[0]), int(p[1])
                displacements.append(flow[x, y])
            displacements = np.array(displacements)

            # Cluster particles by motion similarity
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(displacements)
            labels = clustering.labels_

            # Count unique motion patterns
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            seq_features.append(n_clusters)
        behavior_features.append(np.array(seq_features))

    return np.array(behavior_features)


def classify_region_behavior(flow_frame):

    u = flow_frame[..., 0]
    v = flow_frame[..., 1]

    # spatial derivatives
    du_dx = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
    du_dy = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)
    dv_dx = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3)

    behavior_hist = np.zeros(7)  # 6 behaviors + unclassified
    height, width = u.shape

    for y in range(0, height, 10):
        for x in range(0, width, 10):
            # Jacobian matrix
            J = np.array([[np.mean(du_dx[y:y + 10, x:x + 10]), np.mean(du_dy[y:y + 10, x:x + 10])],
                          [np.mean(dv_dx[y:y + 10, x:x + 10]), np.mean(dv_dy[y:y + 10, x:x + 10])]])

            # eigenvalues
            eigvals = np.linalg.eigvals(J)
            λ1, λ2 = eigvals

            if np.iscomplex(λ1):
                behavior_hist[3] += 1  # Arch/Ring
            elif λ1 > 0 and λ2 > 0:
                behavior_hist[0] += 1  # Bottleneck
            elif λ1 < 0 and λ2 < 0:
                behavior_hist[1] += 1  # Fountainhead
            elif λ1 * λ2 < 0:
                behavior_hist[2] += 1  # Blocking
            elif abs(λ1 - λ2) < 0.1:
                behavior_hist[4] += 1  # Lane
            elif λ1 > λ2:
                behavior_hist[5] += 1  # Merging
            else:
                behavior_hist[6] += 1  # Splitting

    behavior_hist = np.nan_to_num(behavior_hist)
    if np.sum(behavior_hist) == 0:
        return np.zeros(7)
    return behavior_hist / np.sum(behavior_hist)


def estimate_density(flow_frame, density_threshold=0.5):

    start_time = time.time()
    # Extract motion vectors
    u = flow_frame[..., 0].flatten()
    v = flow_frame[..., 1].flatten()

    # Downsample vectors
    if len(u) > 1000:
        idx = np.random.choice(len(u), 1000, replace=False)
        u, v = u[idx], v[idx]

    vectors = np.vstack((u, v)).T

    magnitude = np.mean(np.sqrt(u ** 2 + v ** 2))

    if magnitude < density_threshold:
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=5, algorithm="ball_tree").fit(vectors)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        density_score = n_clusters / 100
    else:
        # KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(vectors)
        density_score = np.exp(kde.score_samples(vectors)).mean()
        n_clusters = 0

    calibrated_score = density_score * (1 + 0.2 * magnitude)

    calc_time = time.time() - start_time
    print(f"Density: {calibrated_score:.4f} | "
          f"Vectors: {len(vectors)} | "
          f"Time: {calc_time:.4f}s | "
          f"Method: {'DBSCAN' if magnitude < density_threshold else 'KDE'}")
    return min(calibrated_score, 1.0), n_clusters  # Cap at 1.0


def create_density_heatmap(flow_frame, grid_size=8):
    """8x8 density heatmap grid """
    start_time = time.time()
    height, width = flow_frame.shape[:2]
    cell_height = height // grid_size
    cell_width = width // grid_size
    heatmap = np.zeros((grid_size, grid_size))

    for i in range(0, grid_size, 2):
        for j in range(0, grid_size, 2):
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            cell_flow = flow_frame[y_start:y_end, x_start:x_end]
            density_score, _ = estimate_density(cell_flow)
            heatmap[i, j] = density_score
    heatmap = cv2.resize(heatmap, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)

    calc_time = time.time() - start_time
    print(f"  Heatmap: {calc_time:.4f}s | "
          f"Min: {np.min(heatmap):.4f} | "
          f"Max: {np.max(heatmap):.4f}")
    return heatmap


def quantize_and_binarize_scalar_features(scalar_features):

    # avoid modifying original
    features = np.copy(scalar_features)

    # Normalize motion_patterns (feature index 4) to [0,1]
    max_motion = 20.0
    features[..., 4] = np.clip(features[..., 4], 0, max_motion) / max_motion

    # Quantize first three features to [0, 255]
    features[..., :3] = np.clip(features[..., :3], 0, 1)
    features[..., :3] = np.round(features[..., :3] * 255).astype(np.uint8)
    # Quantize fourth feature to [0, 255] from [0, 4]
    features[..., 3] = np.clip(features[..., 3], 0, 4)
    features[..., 3] = np.round(features[..., 3] / 4 * 255).astype(np.uint8)

    # Quantize behavior features(0 - 1 range)
    features[..., 4:12] = np.clip(features[..., 4:12], 0, 1)
    features[..., 4:12] = np.round(features[..., 4:12] * 255).astype(np.uint8)

    # Density score (0-1)
    features[..., -66] = np.clip(features[..., -66], 0, 1)
    features[..., -66] = np.round(features[..., -66] * 255).astype(np.uint8)

    # Cluster count (0-100)
    features[..., -65] = np.clip(features[..., -65], 0, 100)
    features[..., -65] = np.round(features[..., -65] / 100 * 255).astype(np.uint8)

    # Heatmap features (0-1)
    features[..., -64:] = np.clip(features[..., -64:], 0, 1)
    features[..., -64:] = np.round(features[..., -64:] * 255).astype(np.uint8)

    bits = np.unpackbits(features.astype(np.uint8)[..., np.newaxis], axis=-1)
    bits = bits.reshape(features.shape[0], features.shape[1], 78 * 8)
    return bits.astype(np.float32)  # Model input should be float32


@tf.keras.saving.register_keras_serializable()
class FeatureWiseAttention(Layer):
    def __init__(self, **kwargs):
        super(FeatureWiseAttention, self).__init__(**kwargs)
        self.dense = None

    def build(self, input_shape):
        self.dense = Dense(input_shape[-1], activation=None)
        super(FeatureWiseAttention, self).build(input_shape)

    def call(self, inputs):
        # Compute attention scores for each feature
        attention_logits = self.dense(inputs)
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)
        return inputs * attention_weights

    def get_config(self):
        config = super(FeatureWiseAttention, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def load_model_safely(model_path: str, config: ModelConfig):
    """
    Safely load a model with custom layers
    """
    try:
        # Try loading with custom objects
        custom_objects = {
            'FeatureWiseAttention': FeatureWiseAttention
        }

        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        # Recompile the model
        model.compile(
            optimizer=Adam(learning_rate=config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"Model loaded successfully from {model_path}")
        return model, True

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model...")
        return None, False

def generate_optical_flow_and_features_from_video(video_path, output_dir, resize_dim=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                  max_frames=200, frame_skip=5):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate time intervals for frame sampling - aim for ~4 seconds of content
    target_duration = 4  # seconds
    target_frames = min(max_frames, int(target_duration * fps))

    # Calculate frame indices to capture
    if total_frames <= target_frames:
        frame_indices = list(range(0, total_frames, frame_skip))
    else:
        # Pick frames evenly distributed across the video
        frame_indices = [int(i * total_frames / target_frames) for i in range(target_frames)]

    print(f"Video: {total_frames} frames at {fps} FPS. Using {len(frame_indices)} frames for analysis.")

    # downsampling
    downsample_factor = 0.2

    # Process frames
    flow_frames = []
    original_frames = []
    processed_count = 0
    prev_r, prev_g, prev_b = None, None, None

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame at index {frame_idx}")
            continue

        # Save original frame for SSIM calculation
        resized_frame = cv2.resize(frame, resize_dim)
        original_frames.append(resized_frame)

        # Keep original frame before downsampling
        original_frame = frame.copy()  # <-- NEW

        # Downsample for processing
        frame = cv2.resize(frame, (0, 0), fx=downsample_factor, fy=downsample_factor)

        # Convert ORIGINAL frame to RGB
        frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        r = frame_rgb[..., 0]
        g = frame_rgb[..., 1]
        b = frame_rgb[..., 2]

        # Resize channels to target dimensions
        r = cv2.resize(r, resize_dim)
        g = cv2.resize(g, resize_dim)
        b = cv2.resize(b, resize_dim)

        # Initialize first frame
        if prev_r is None:
            prev_r, prev_g, prev_b = r, g, b
            continue

        # Calculate optical flow for each channel
        flow_r = cv2.calcOpticalFlowFarneback(
            prev_r, r, None,
            pyr_scale=0.5,
            levels=2,
            winsize=11,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flow_g = cv2.calcOpticalFlowFarneback(
            prev_g, g, None,
            pyr_scale=0.5,
            levels=2,
            winsize=11,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flow_b = cv2.calcOpticalFlowFarneback(
            prev_b, b, None,
            pyr_scale=0.5,
            levels=2,
            winsize=11,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Average flows from all channels
        combined_flow = (flow_r + flow_g + flow_b) / 3.0

        # Store averaged flow
        flow_frames.append(combined_flow)

        # Update previous frames
        prev_r, prev_g, prev_b = r, g, b
        processed_count += 1

        # Print progress
        if processed_count % 5 == 0:
            print(f"Processed {processed_count}/{len(frame_indices)} frames")

    cap.release()
    print(f"Generated {len(flow_frames)} optical flow frames")

    # Create dummy original frames array if empty
    if not original_frames and flow_frames:
        original_frames = [np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)] * len(flow_frames)

    return flow_frames, original_frames


def predict_with_enhanced_model(model, video_path, temp_dir="temp_optical_flow", timeout_seconds=300):

    def test_model_inference(model):
        """Test that model inference is working and measure its speed"""
        print("Testing model inference speed...")

        # Create dummy inputs
        dummy_flow_input = np.zeros((1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 2), dtype=np.float32)
        dummy_scalar_input = np.zeros((1, SEQUENCE_LENGTH, 624), dtype=np.float32)

        # Time the prediction
        start = time.time()
        dummy_pred = model.predict([dummy_flow_input, dummy_scalar_input], verbose=0)
        end = time.time()

        print(f"Model inference test: {end - start:.3f} seconds for a single sequence")
        print(f"Prediction shape: {dummy_pred.shape}, values: min={dummy_pred.min():.3f}, max={dummy_pred.max():.3f}")
        return end - start

    # Test model inference at the start
    inference_time = test_model_inference(model)
    start_time = time.time()


    def check_timeout():
        if time.time() - start_time > timeout_seconds:
            print(f"Processing timed out after {timeout_seconds} seconds!")
            return True
        return False

    # Generate optical flow frames and original frames
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Generating optical flow and features for video: {video_path}")

    flow_frames, original_frames = generate_optical_flow_and_features_from_video(
        video_path,
        temp_dir,
        max_frames=200,
        frame_skip=5
    )

    if check_timeout(): return "Timeout (processing took too long)", 0.0

    if len(flow_frames) < SEQUENCE_LENGTH:
        print(f"Warning: Video has only {len(flow_frames)} frames, which is less than required {SEQUENCE_LENGTH}")
        return "Unknown (not enough frames)", 0.0

    print(f"Creating sequences from {len(flow_frames)} frames...")

    # Create sequences for prediction
    flow_sequences = []
    orig_sequences = []
    max_sequences = 20

    if len(flow_frames) > SEQUENCE_LENGTH:
        step_size = max(1, (len(flow_frames) - SEQUENCE_LENGTH) // max_sequences)
        start_indices = range(0, len(flow_frames) - SEQUENCE_LENGTH, step_size)

        # Limit to max_sequences
        start_indices = list(start_indices)[:max_sequences]

        for i in start_indices:
            flow_seq = flow_frames[i:i + SEQUENCE_LENGTH]
            orig_seq = original_frames[i:i + SEQUENCE_LENGTH] if i < len(original_frames) else []

            flow_sequences.append(np.array(flow_seq))
            orig_sequences.append(np.array(orig_seq) if orig_seq else None)
            print(f"Created sequence starting at frame {i}/{len(flow_frames)}")

            if check_timeout(): return "Timeout (processing took too long)", 0.0

    if not flow_sequences:
        return "Unknown (could not create sequences)", 0.0

    # Calculate additional features for each sequence
    print("Calculating additional features for prediction...")

    # Flow acceleration
    flow_acceleration = calculate_flow_acceleration(flow_sequences)

    # Flow divergence
    flow_divergence = calculate_flow_divergence(flow_sequences)

    # Scene change detection (if we have original frames)
    scene_changes = calculate_scene_changes(orig_sequences) if all(seq is not None for seq in orig_sequences) else \
        np.zeros((len(flow_sequences), SEQUENCE_LENGTH))

    # Motion entropy
    motion_entropy = calculate_motion_entropy(flow_sequences)

    # Calculate motion patterns and behaviors
    motion_patterns = segment_motion_patterns(flow_sequences)
    behavior_features = []
    for seq in flow_sequences:
        seq_behaviors = []
        for flow in seq:
            behavior_hist = classify_region_behavior(flow)
            seq_behaviors.append(behavior_hist)
        behavior_features.append(np.array(seq_behaviors))
    behavior_features = np.array(behavior_features)

    # Add dimension check and correction
    if behavior_features.ndim == 2:
        # Reshape to (num_sequences, sequence_length, 1)
        behavior_features = behavior_features[..., np.newaxis]
    elif behavior_features.ndim == 3 and behavior_features.shape[-1] != 7:
        # Handle case where last dimension is missing
        behavior_features = behavior_features.reshape(
            behavior_features.shape[0],
            behavior_features.shape[1],
            -1
        )

    # Calculate density and heatmap features for each sequence
    density_features = []
    heatmap_features = []
    for seq in flow_sequences:
        seq_density = []
        seq_heatmaps = []
        for flow in seq:
            density_score, cluster_count = estimate_density(flow)
            seq_density.append([density_score, cluster_count])
            heatmap = create_density_heatmap(flow)
            seq_heatmaps.append(heatmap.flatten())
        density_features.append(np.array(seq_density))
        heatmap_features.append(np.array(seq_heatmaps))
    density_features = np.array(density_features)
    heatmap_features = np.array(heatmap_features)
    avg_heatmap_features = np.mean(heatmap_features, axis=1)
    # Stack all features
    scalar_features = np.concatenate([
        flow_acceleration[..., None],
        flow_divergence[..., None],
        scene_changes[..., None],
        motion_entropy[..., None],
        motion_patterns[..., None],
        behavior_features,
        density_features,
        avg_heatmap_features.repeat(SEQUENCE_LENGTH, axis=0).reshape(-1, SEQUENCE_LENGTH, 64)
    ], axis=2)
    # Shape: [num_sequences, sequence_length, 4]

    print(f"Prepared {len(flow_sequences)} sequences with additional features for prediction")

    # Display feature ranges
    feature_names = ["Flow Acceleration", "Flow Divergence", "Scene Changes", "Motion Entropy"]
    print("\n" + "=" * 60)
    print("PREDICTION DATA - FEATURE RANGES (DECIMAL VALUES)")
    print("=" * 60)

    for i, feature_name in enumerate(feature_names):
        feature_data = scalar_features[:, :, i]
        min_val = np.min(feature_data)
        max_val = np.max(feature_data)
        mean_val = np.mean(feature_data)
        std_val = np.std(feature_data)

        print(f"{feature_name}:")
        print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std Dev: {std_val:.6f}")
        print("-" * 40)

    print("=" * 60)

    # Quantize and binarize for prediction
    binary_scalar_features = quantize_and_binarize_scalar_features(scalar_features)

    # Convert sequences to numpy arrays
    X_flow_pred = np.array(flow_sequences)
    X_scalar_pred = binary_scalar_features

    # Make predictions in smaller batches to avoid memory issues
    batch_size = 8
    all_predictions = []

    print(f"Making predictions in batches of {batch_size}...")
    for i in range(0, len(X_flow_pred), batch_size):
        batch_flow = X_flow_pred[i:i + batch_size]
        batch_scalar = X_scalar_pred[i:i + batch_size]
        batch_preds = model.predict([batch_flow, batch_scalar], verbose=0)
        all_predictions.append(batch_preds)
        print(f"Processed batch {i // batch_size + 1}/{(len(X_flow_pred) + batch_size - 1) // batch_size}")

        print(f"batch_flow shape: {batch_flow.shape}")
        print(f"batch_scalar shape: {batch_scalar.shape}")
    # Combine all batch predictions
    predictions = np.vstack(all_predictions) if len(all_predictions) > 1 else all_predictions[0]

    # Calculate average prediction across all sequences
    avg_prediction = np.mean(predictions, axis=0)

    # Get class index with highest probability
    class_idx = np.argmax(avg_prediction)

    # Map class index to category
    categories = ["normal", "moderate", "dense", "risky"]
    category = categories[class_idx]

    # Output confidence score
    confidence = avg_prediction[class_idx]

    # Display more detailed results
    print("\nDetailed prediction results:")
    for i, cat in enumerate(categories):
        print(f"{cat}: {avg_prediction[i] * 100:.2f}%")

    return category, confidence


def analyze_features_by_category(X_flow, X_scalar_raw, y):
    """
    Analyze feature ranges by category to understand class-specific patterns
    """
    categories = ["normal", "moderate", "dense", "risky"]
    feature_names = ["Flow Acceleration", "Flow Divergence", "Scene Changes", "Motion Entropy"]

    print("\n" + "=" * 80)
    print("DETAILED FEATURE ANALYSIS BY CATEGORY")
    print("=" * 80)

    for cat_idx, category in enumerate(categories):
        # Get indices for this category
        cat_indices = np.where(y == cat_idx)[0]

        if len(cat_indices) == 0:
            print(f"\n{category.upper()}: No samples found")
            continue

        print(f"\n{category.upper()} ({len(cat_indices)} samples):")
        print("-" * 50)

        # Get features for this category
        cat_features = X_scalar_raw[cat_indices]  # Shape: [num_samples, seq_length, 4]

        for feat_idx, feature_name in enumerate(feature_names):
            feat_data = cat_features[:, :, feat_idx]

            print(f"  {feature_name}:")
            print(f"    Range: [{np.min(feat_data):.6f}, {np.max(feat_data):.6f}]")
            print(f"    Mean: {np.mean(feat_data):.6f} ± {np.std(feat_data):.6f}")
            print(f"    Median: {np.median(feat_data):.6f}")

            # Show percentiles
            p25, p75 = np.percentile(feat_data, [25, 75])
            print(f"    25th-75th percentile: [{p25:.6f}, {p75:.6f}]")
            print()

    print("=" * 80)


def test_enhanced_model(model_path, video_path):
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return

    try:
        # Load the pre-trained model with custom objects
        custom_objects = {
            'FeatureWiseAttention': FeatureWiseAttention
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Make prediction
    temp_dir = "temp_optical_flow_test"
    category, confidence = predict_with_enhanced_model(model, video_path, temp_dir)

    print("\n" + "=" * 50)
    print(f"Video Classification Result: {category.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("=" * 50)


def run_test_mode(args):
    print("Running in test-only mode")
    if args.use_enhanced:
        test_enhanced_model(args.model_path, args.video_path)
    else:
        test_video_only(args.model_path, args.video_path)


@handle_errors
def main():
    # Load configuration
    config = ModelConfig()
    # Initialize GPU management
    gpu_manager = GPUManager(config)
    # Parse arguments (keeping existing argument parsing)
    args = parse_arguments()

    if args.test_only:
        run_test_mode(args)
        return

    start_time = time.time()

    # Load data
    #data_path = r"C:\Users\rnidh\stampedeDetection\output\Stampede_detection_dataset"
    categories = ["normal", "moderate", "dense", "risky"]

    print("Loading data with memory-efficient streaming...")
    X_flow, X_scalar, y = load_optical_flow_data_optimized(data_path, categories, config)

    if len(X_flow) == 0:
        print("Error: No data loaded. Please check the data path and directory structure.")
        return

    # Train model
    print("Training model with memory optimization...")
    model, history, last_epoch = train_enhanced_model_memory_optimized(
        X_flow, X_scalar, y, config,
        model_path=args.model_path,
        continue_training=args.continue_training
    )

    # Test prediction
    if args.video_path:
        print("Testing prediction...")
        temp_dir = "temp_optical_flow_prediction"
        category, confidence = predict_with_enhanced_model(
            model, args.video_path, temp_dir
        )
        print(f"\nResult: {category.upper()} (Confidence: {confidence * 100:.2f}%)")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    gpu_manager.cleanup_memory()


def visualize_feature_importance(flow_sequences, scalar_features, labels):
    """
    Visualize the additional features to understand their contribution
    """
    categories = ["normal", "moderate", "dense", "risky"]
    feature_names = ["Flow Acceleration", "Flow Divergence", "Scene Changes", "Motion Entropy"]

    plt.figure(figsize=(15, 10))

    # Process each sequence
    for seq_idx in range(min(5, len(flow_sequences))):
        category = categories[labels[seq_idx]]

        # Plot the 4 scalar features
        plt.subplot(len(flow_sequences), 1, seq_idx + 1)

        for feature_idx in range(4):
            feature_values = scalar_features[seq_idx, :, feature_idx]
            plt.plot(feature_values, label=feature_names[feature_idx])

        plt.title(f"Sequence {seq_idx + 1} (Category: {category})")
        plt.xlabel("Frame")
        plt.ylabel("Feature Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_visualization.png')
    plt.show()


if __name__ == "__main__":
    # avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    main()
