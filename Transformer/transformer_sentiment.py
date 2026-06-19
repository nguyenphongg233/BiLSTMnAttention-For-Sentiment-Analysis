"""Pipeline Kaggle: Amazon Reviews 1-5 sao với Transformer encoder nhỏ."""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentiment_pipeline import base_config, env_int, run_pipeline


@tf.keras.utils.register_keras_serializable(package="sentiment")
class TokenAndPositionEmbedding(Layer):
    def __init__(self, max_len: int, vocab_size: int, embed_dim: int, embedding_matrix=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        if embedding_matrix is not None:
            self.token_embedding = Embedding(vocab_size, embed_dim, mask_zero=True, weights=[embedding_matrix], trainable=True)
        else:
            self.token_embedding = Embedding(vocab_size, embed_dim, mask_zero=True)
        self.position_embedding = Embedding(max_len, embed_dim)
        self.supports_masking = True

    def call(self, token_ids):
        positions = tf.keras.ops.arange(tf.keras.ops.shape(token_ids)[-1])
        return self.token_embedding(token_ids) + self.position_embedding(positions)

    def compute_mask(self, token_ids, mask=None):
        return token_ids != 0

    def get_config(self):
        return {
            **super().get_config(),
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        }


@tf.keras.utils.register_keras_serializable(package="sentiment")
class TransformerBlock(Layer):
    """Encoder block: self-attention, residual/norm, FFN, residual/norm."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float, **kwargs):
        super().__init__(**kwargs)
        if embed_dim % num_heads:
            raise ValueError("embedding_dim phải chia hết cho num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        # key_dim là chiều của từng head, không phải toàn bộ embedding.
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=rate,
        )
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="gelu"),
                Dropout(rate),
                Dense(embed_dim),
            ],
            name="feed_forward",
        )
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.supports_masking = True

    def call(self, inputs, training=None, mask=None):
        attention_mask = None
        if mask is not None:
            attention_mask = mask[:, tf.newaxis, :]
        attended = self.attention(
            inputs,
            inputs,
            attention_mask=attention_mask,
            training=training,
        )
        x = self.norm1(inputs + self.dropout1(attended, training=training))
        return self.norm2(x + self.dropout2(self.ffn(x, training=training), training=training))

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        return {
            **super().get_config(),
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        }


def build_model(vocab_size: int, config: dict) -> Model:
    inputs = Input(shape=(config["max_len"],), dtype="int32", name="tokens")
    x = TokenAndPositionEmbedding(
        config["max_len"],
        vocab_size,
        config["embedding_dim"],
        embedding_matrix=config.get("embedding_matrix"),
        name="token_position_embedding",
    )(inputs)
    for index in range(config["transformer_blocks"]):
        x = TransformerBlock(
            embed_dim=config["embedding_dim"],
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            rate=config["dropout_rate"],
            name=f"transformer_block_{index + 1}",
        )(x)
    # GlobalAveragePooling1D nhận mask truyền qua block và bỏ padding khỏi trung bình.
    x = GlobalAveragePooling1D(name="masked_average_pooling")(x)
    x = Dropout(config["dropout_rate"])(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(config["l2_rate"]))(x)
    x = Dropout(config["dropout_rate"])(x)
    outputs = Dense(
        config["num_classes"], 
        activation="softmax", 
        kernel_regularizer=l2(config["l2_rate"]),
        name="rating"
    )(x)
    model = Model(inputs, outputs, name="Transformer_Encoder_Classifier")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    config = base_config("transformer")
    config.update(
        {
            "num_heads": env_int("NUM_HEADS", 4),
            "ff_dim": env_int("FF_DIM", 256),
            "transformer_blocks": env_int("TRANSFORMER_BLOCKS", 1),
        }
    )
    run_pipeline(
        model_key="transformer",
        model_name="Transformer Encoder",
        model_filename="transformer.keras",
        build_model=build_model,
        config=config,
    )


if __name__ == "__main__":
    main()
