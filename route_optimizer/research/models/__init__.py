from .base import BaseForecaster
from .lstm import LSTMForecaster
from .gru import GRUForecaster
from .attention_lstm import AttentionLSTMForecaster
from .temp_cnn import TemporalCNNForecaster
from .transformer import LightweightTransformerForecaster
from typing import Dict, Type

MODEL_REGISTRY: Dict[str, Type[BaseForecaster]] = {
    "LSTM": LSTMForecaster,
    "GRU": GRUForecaster,
    "Attention-LSTM": AttentionLSTMForecaster,
    "Temporal CNN": TemporalCNNForecaster,
    "Lightweight Transformer": LightweightTransformerForecaster,
}

def get_model(name: str, **kwargs) -> BaseForecaster:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in Model Registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)

__all__ = [
    "BaseForecaster",
    "LSTMForecaster",
    "GRUForecaster",
    "AttentionLSTMForecaster",
    "TemporalCNNForecaster",
    "LightweightTransformerForecaster",
    "MODEL_REGISTRY",
    "get_model",
]
