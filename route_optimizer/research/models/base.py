import os
import json
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

class BaseForecaster(ABC):
    """
    Abstract Base Class for all Deep Learning AQI Forecasting Models.
    Provides standard interface for model training, prediction, parameter serialization,
    and save/load operations across different training paradigms (Centralized, Local, Federated).
    """

    def __init__(self, model_name: str, input_dim: int = 10, seq_len: int = 24, pred_len: int = 1):
        self.model_name = model_name
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.is_trained = False
        self.version = "1.0.0"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32, verbose: bool = False) -> Dict[str, Any]:
        """
        Train model on sequence input X (shape: [N, seq_len, input_dim]) and targets y (shape: [N, pred_len]).
        Returns dictionary containing loss history and metrics.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict future AQI values given past sequence inputs X.
        X shape: [N, seq_len, input_dim] or [seq_len, input_dim]
        Returns: np.ndarray of shape [N, pred_len]
        """
        pass

    @abstractmethod
    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Get model weight parameters as dictionary of NumPy arrays (used for Federated Aggregation).
        """
        pass

    @abstractmethod
    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Set model weight parameters from dictionary of NumPy arrays.
        """
        pass

    def save(self, filepath: str) -> None:
        """Save model state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "model_name": self.model_name,
            "version": self.version,
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "is_trained": self.is_trained,
            "weights": self.get_weights()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> None:
        """Load model state from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model_name = data["model_name"]
        self.version = data.get("version", "1.0.0")
        self.input_dim = data["input_dim"]
        self.seq_len = data["seq_len"]
        self.pred_len = data["pred_len"]
        self.is_trained = data.get("is_trained", True)
        self.set_weights(data["weights"])

    def create_sequences(self, data: np.ndarray, target_col_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Utility to create sliding window sequences (X, y) from multi-variate array.
        data: shape [total_steps, input_dim]
        """
        num_samples = len(data) - self.seq_len - self.pred_len + 1
        if num_samples <= 0:
            raise ValueError(f"Data length ({len(data)}) too short for seq_len={self.seq_len}")
        
        X, y = [], []
        for i in range(num_samples):
            X.append(data[i : i + self.seq_len])
            y.append(data[i + self.seq_len : i + self.seq_len + self.pred_len, target_col_idx])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
