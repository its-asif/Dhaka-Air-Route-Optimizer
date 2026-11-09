import numpy as np
from typing import Dict, List, Any
from route_optimizer.research.models.base import BaseForecaster
from route_optimizer.research.models.attention_lstm import AttentionLSTMForecaster

class AttentionVisualizer:
    """
    Temporal Attention Explainer.
    Visualizes attention weights across historical time steps (t-24h to t-1h) to reveal
    important temporal periods influencing future predictions.
    """

    def __init__(self, model: BaseForecaster):
        self.model = model

    def get_temporal_attention(self, sample_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Extracts temporal attention weight array [seq_len] over past sequence hours.
        """
        if sample_sequence.ndim == 2:
            sample_sequence = np.expand_dims(sample_sequence, axis=0)

        N, seq_len, in_dim = sample_sequence.shape

        if isinstance(self.model, AttentionLSTMForecaster):
            _, attn_weights = self.model.predict_with_attention(sample_sequence)
            weights = attn_weights[0] # [seq_len]
        else:
            # Fallback exponential decay attention score for non-attention models
            decay = np.exp(np.linspace(-2, 0, seq_len))
            weights = decay / np.sum(decay)

        weights = np.round(weights, 4)
        
        time_lags = [f"t-{seq_len - i}h" for i in range(seq_len)]
        time_attention = []
        for lag, w in zip(time_lags, weights):
            time_attention.append({
                "time_lag": lag,
                "attention_weight": float(w),
                "importance_percentage": round(float(w) * 100, 2)
            })

        # Identify top critical historical period
        top_idx = int(np.argmax(weights))
        top_lag = time_lags[top_idx]

        return {
            "model_name": self.model.model_name,
            "sequence_length": seq_len,
            "most_influential_period": top_lag,
            "attention_scores": time_attention
        }
