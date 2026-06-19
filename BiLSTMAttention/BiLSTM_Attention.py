"""Pipeline Kaggle: Amazon Reviews 1-5 sao với BiLSTM + additive attention."""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, Input, LSTM, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentiment_pipeline import base_config, env_int, run_pipeline


@tf.keras.utils.register_keras_serializable(package="sentiment")
class AttentionLayer(Layer):
    """Additive attention có mask để padding không nhận trọng số."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        features = int(input_shape[-1])
        self.weight_matrix = self.add_weight(
            name="attention_weight",
            shape=(features, features),
            initializer="glorot_uniform",
        )
        self.bias = self.add_weight(
            name="attention_bias", shape=(features,), initializer="zeros"
        )
        self.context_vector = self.add_weight(
            name="attention_context",
            shape=(features, 1),
            initializer="glorot_uniform",
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        scores = tf.squeeze(
            tf.matmul(tf.nn.tanh(tf.matmul(inputs, self.weight_matrix) + self.bias), self.context_vector),
            axis=-1,
        )
        if mask is not None:
            scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


def build_model(vocab_size: int, config: dict) -> Model:
    inputs = Input(shape=(config["max_len"],), dtype="int32", name="tokens")
    x = Embedding(
        vocab_size,
        config["embedding_dim"],
        mask_zero=True,
        weights=[config["embedding_matrix"]],
        trainable=True,
        name="token_embedding",
    )(inputs)
    x = Dropout(config["dropout_rate"], name="embedding_dropout")(x)
    x = Bidirectional(
        LSTM(config["lstm_units"], return_sequences=True),
        name="bidirectional_lstm",
    )(x)
    x = Dropout(config["dropout_rate"], name="bilstm_dropout")(x)
    x = AttentionLayer(name="additive_attention")(x)
    x = Dense(
        64, 
        activation="relu", 
        kernel_regularizer=l2(config["l2_rate"]),
        name="classifier_hidden"
    )(x)
    x = Dropout(config["dropout_rate"], name="classifier_dropout")(x)
    outputs = Dense(
        config["num_classes"], 
        activation="softmax", 
        kernel_regularizer=l2(config["l2_rate"]),
        name="rating"
    )(x)
    model = Model(inputs, outputs, name="BiLSTM_Attention")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    config = base_config("bilstm_attention")
    config["lstm_units"] = env_int("LSTM_UNITS", 128)
    run_pipeline(
        model_key="bilstm_attention",
        model_name="BiLSTM + Attention",
        model_filename="bilstm_attention.keras",
        build_model=build_model,
        config=config,
    )


if __name__ == "__main__":
    main()
