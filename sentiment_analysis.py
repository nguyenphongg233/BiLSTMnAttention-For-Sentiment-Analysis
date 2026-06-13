"""
BÁO CÁO NHẬP MÔN AI - Nhóm 5
Sentiment Analysis trên Amazon Reviews với BiLSTM + Attention

Thành viên:
    - Trần Lê Ngọc Tâm - 202416346
    - Nguyễn Phong - 202400066
    - Nguyễn Ngọc Tuấn Anh - 202400029

Dataset: https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset
Phân loại 5 lớp (1-5 sao)

Hướng dẫn chạy trên Kaggle:
    1. Tạo Notebook mới trên Kaggle
    2. Bật GPU: Settings → Accelerator → GPU T4 x2
    3. Add Dataset: + Add Data → tìm "dongrelaxman/amazon-reviews-dataset" → Add
    4. Copy toàn bộ code này vào 1 cell → Run All
"""

# ==============================================================================
# 0. IMPORT THƯ VIỆN
# ==============================================================================
# import sys

# if sys.version_info >= (3, 13):
#     raise RuntimeError(
#         "This project requires Python 3.11 or 3.12. "
#         "The current interpreter is too new for the TensorFlow/pandas stack used here."
#     )

import numpy as np
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense,
    Dropout, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import kagglehub

try:
    import pandas as pd
except Exception as exc:
    raise RuntimeError(
        "pandas could not be imported. Install the project dependencies with "
        "'pip install -r requirements.txt' inside a Python 3.11 or 3.12 environment."
    ) from exc

# Kiểm tra GPU trên Kaggle
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPU available:", gpus)
if not gpus:
    print("⚠ Không có GPU. Bật GPU: Settings → Accelerator → GPU T4 x2")

# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================

OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Tham số mô hình ----
MAX_WORDS = 50000       # Kích thước từ điển tối đa
MAX_LEN = 200           # Độ dài tối đa của mỗi câu (padding)
EMBEDDING_DIM = 128     # Chiều không gian embedding
LSTM_UNITS = 128        # Số unit trong mỗi lớp LSTM
DROPOUT_RATE = 0.3      # Tỷ lệ dropout
NUM_CLASSES = 5         # Số lớp phân loại (1-5 sao)
BATCH_SIZE = 128        # Batch size (lớn hơn nhờ GPU Kaggle)
EPOCHS = 10             # Số epoch tối đa
VALIDATION_SPLIT = 0.2  # Tỷ lệ chia validation
RANDOM_STATE = 42       # Seed để tái tạo kết quả

# ==============================================================================
# 2. ATTENTION LAYER (Custom Keras Layer)
# ==============================================================================

class AttentionLayer(Layer):
    """
    Attention Layer: Gán trọng số cho từng timestep của BiLSTM output,
    giúp mô hình tập trung vào các từ quan trọng trong câu.

    Công thức:
        e_t = tanh(W * h_t + b)
        a_t = softmax(v^T * e_t)
        context = sum(a_t * h_t)
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, features)
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.v = self.add_weight(
            name='attention_v',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, timesteps, features)
        # Tính attention score
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)  # (batch, timesteps, features)
        e = tf.matmul(e, self.v)                              # (batch, timesteps, 1)
        e = tf.squeeze(e, axis=-1)                             # (batch, timesteps)

        # Softmax để chuẩn hóa trọng số
        alpha = tf.nn.softmax(e, axis=-1)                      # (batch, timesteps)

        # Tính context vector = tổng trọng số của các hidden states
        alpha_expanded = tf.expand_dims(alpha, axis=-1)        # (batch, timesteps, 1)
        context = tf.reduce_sum(inputs * alpha_expanded, axis=1)  # (batch, features)

        return context

    def get_config(self):
        return super(AttentionLayer, self).get_config()


# ==============================================================================
# 3. TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================

def clean_text(text):
    """
    Chuẩn hóa văn bản:
    - Chuyển về lowercase
    - Loại bỏ HTML tags
    - Loại bỏ ký tự đặc biệt, giữ lại chữ cái và số
    - Loại bỏ khoảng trắng thừa
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)           # Xóa HTML tags
    text = re.sub(r'[^a-z0-9\s]', ' ', text)       # Giữ chữ và số
    text = re.sub(r'\s+', ' ', text).strip()        # Xóa khoảng trắng thừa
    return text


def load_and_preprocess_data():
    """
    Load Amazon_Reviews.csv từ Kaggle bằng kagglehub và tiền xử lý.
    Dataset có cột: Review Text, Rating (và các cột khác không dùng).
    """
    print("=" * 60)
    print("BƯỚC 1: LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    # ---- Download dataset rồi đọc thủ công (tránh lỗi malformed CSV) ----
    print(f"\n[1.1] Đang download dữ liệu từ Kaggle (dongrelaxman/amazon-reviews-dataset)...")
    dataset_path = kagglehub.dataset_download("dongrelaxman/amazon-reviews-dataset")
    csv_path = os.path.join(dataset_path, "Amazon_Reviews.csv")
    print(f"      Dataset path: {dataset_path}")

    df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
    print(f"      Kích thước ban đầu: {df.shape}")
    print(f"      Các cột: {list(df.columns)}")

    # ---- Lấy 2 cột cần thiết: 'Review Text' và 'Rating' ----
    df = df[['Review Text', 'Rating']].copy()
    df.columns = ['text', 'rating']
    df.dropna(inplace=True)

    # ---- Parse rating (có thể dạng "4/5", "4 out of 5", hoặc số) ----
    def parse_rating(val):
        val = str(val).strip()
        match = re.search(r'(\d)', val)
        if match:
            r = int(match.group(1))
            if 1 <= r <= 5:
                return r
        return None

    df['rating'] = df['rating'].apply(parse_rating)
    df.dropna(subset=['rating'], inplace=True)
    df['rating'] = df['rating'].astype(int)

    print(f"      Sau lọc: {df.shape}")
    print(f"\n      Phân bố rating:")
    print(df['rating'].value_counts().sort_index().to_string())

    # ----- Chuẩn hóa văn bản -----
    print(f"\n[1.2] Đang chuẩn hóa văn bản...")
    df['text_clean'] = df['text'].apply(clean_text)

    # Loại bỏ review rỗng sau khi clean
    df = df[df['text_clean'].str.len() > 0]
    print(f"      Kích thước sau chuẩn hóa: {df.shape}")

    # ----- Tokenization -----
    print(f"\n[1.3] Tokenization (MAX_WORDS={MAX_WORDS})...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['text_clean'])

    sequences = tokenizer.texts_to_sequences(df['text_clean'])
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
    print(f"      Kích thước từ điển: {vocab_size}")

    # ----- Padding -----
    print(f"\n[1.4] Padding sequences (MAX_LEN={MAX_LEN})...")
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # ----- Chuyển label -----
    # Rating 1-5 → Label 0-4 (cho categorical crossentropy)
    y = to_categorical(df['rating'].values - 1, num_classes=NUM_CLASSES)

    # ----- Train/Validation Split -----
    print(f"\n[1.5] Chia dữ liệu: Train ({1 - VALIDATION_SPLIT:.0%}) / Validation ({VALIDATION_SPLIT:.0%})")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=df['rating'].values  # Đảm bảo phân bố đều
    )

    print(f"      Train: {X_train.shape[0]:,} samples")
    print(f"      Validation: {X_val.shape[0]:,} samples")
    print(f"\n      ✓ Tiền xử lý hoàn tất!")

    return X_train, X_val, y_train, y_val, tokenizer, vocab_size


# ==============================================================================
# 4. XÂY DỰNG MÔ HÌNH BiLSTM + Attention
# ==============================================================================

def build_bilstm_attention_model(vocab_size=MAX_WORDS):
    """
    Xây dựng mô hình BiLSTM + Attention

    Kiến trúc:
        Input → Embedding → BiLSTM → Attention → Dense → Softmax

    Returns:
        Keras Model đã compile
    """
    print("\n" + "=" * 60)
    print("BƯỚC 2: XÂY DỰNG MÔ HÌNH BiLSTM + Attention")
    print("=" * 60)

    # ----- Input Layer -----
    inputs = Input(shape=(MAX_LEN,), name='input_layer')

    # ----- Embedding Layer -----
    # Biến các từ (index) thành vector không gian nhiều chiều
    x = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LEN,
        name='embedding_layer'
    )(inputs)

    # ----- Dropout sau Embedding -----
    x = Dropout(DROPOUT_RATE, name='embedding_dropout')(x)

    # ----- BiLSTM Layer -----
    # Đọc câu theo cả 2 chiều: trái→phải và phải→trái
    # return_sequences=True để trả về output tại mọi timestep (cần cho Attention)
    x = Bidirectional(
        LSTM(LSTM_UNITS, return_sequences=True, name='lstm_layer'),
        name='bilstm_layer'
    )(x)

    # ----- Dropout sau BiLSTM -----
    x = Dropout(DROPOUT_RATE, name='bilstm_dropout')(x)

    # ----- Attention Layer -----
    # Gán trọng số cho từng timestep, tập trung vào từ quan trọng
    x = AttentionLayer(name='attention_layer')(x)

    # ----- Dense Layers -----
    x = Dense(64, activation='relu', name='dense_hidden')(x)
    x = Dropout(DROPOUT_RATE, name='dense_dropout')(x)

    # ----- Output Layer (Softmax - 5 lớp) -----
    outputs = Dense(NUM_CLASSES, activation='softmax', name='output_layer')(x)

    # ----- Compile Model -----
    model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_Attention')

    model.compile(
        loss='categorical_crossentropy',    # Loss function cho bài toán multi-class
        optimizer='adam',                    # Optimizer Adam
        metrics=['accuracy']
    )

    print("\n[2.1] Kiến trúc mô hình:")
    model.summary()

    return model


# ==============================================================================
# 5. HUẤN LUYỆN MÔ HÌNH
# ==============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Huấn luyện mô hình với EarlyStopping và ReduceLROnPlateau

    Returns:
        history: Lịch sử training
    """
    print("\n" + "=" * 60)
    print("BƯỚC 3: HUẤN LUYỆN MÔ HÌNH")
    print("=" * 60)

    # ----- Callbacks -----
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    print(f"\n[3.1] Bắt đầu training...")
    print(f"      Batch size: {BATCH_SIZE}")
    print(f"      Epochs: {EPOCHS}")
    print(f"      EarlyStopping patience: 3")
    print(f"      Shuffle: True")

    # ----- Training -----
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        shuffle=True,     # Shuffle dữ liệu mỗi epoch
        verbose=1
    )

    print(f"\n      ✓ Huấn luyện hoàn tất!")
    return history


# ==============================================================================
# 6. ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================

def evaluate_model(model, X_val, y_val, history):
    """
    Đánh giá mô hình:
    - Accuracy
    - Precision / Recall / F1-score cho từng lớp
    - Confusion Matrix
    - Biểu đồ Loss/Accuracy
    """
    print("\n" + "=" * 60)
    print("BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)

    # ----- Dự đoán -----
    y_pred_probs = model.predict(X_val, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # ----- Accuracy -----
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[4.1] Overall Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    # ----- Classification Report -----
    print(f"\n[4.2] Classification Report (Precision / Recall / F1-score):")
    target_names = ['1 sao', '2 sao', '3 sao', '4 sao', '5 sao']
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    # ----- Confusion Matrix -----
    print(f"[4.3] Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # ----- Vẽ biểu đồ -----
    plot_training_history(history)
    plot_confusion_matrix(cm, target_names)

    return acc, report, cm


def plot_training_history(history):
    """Vẽ biểu đồ Loss và Accuracy qua các epoch"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Training & Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('BiLSTM + Attention - Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("      → Đã lưu biểu đồ: training_history.png")


def plot_confusion_matrix(cm, labels):
    """Vẽ Confusion Matrix dạng heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    plt.title('Confusion Matrix - BiLSTM + Attention', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("      → Đã lưu biểu đồ: confusion_matrix.png")


# ==============================================================================
# 7. DỰ ĐOÁN MỚI (INFERENCE)
# ==============================================================================

def predict_sentiment(model, tokenizer, text):
    """
    Dự đoán cảm xúc cho một review mới.

    Args:
        model: Mô hình đã train
        tokenizer: Tokenizer đã fit
        text: Chuỗi review cần phân tích

    Returns:
        predicted_rating: Rating dự đoán (1-5)
        probabilities: Xác suất từng lớp
    """
    # Tiền xử lý text
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    # Dự đoán
    probs = model.predict(padded, verbose=0)[0]
    predicted_class = np.argmax(probs)
    predicted_rating = predicted_class + 1  # Chuyển từ 0-4 về 1-5

    # In kết quả
    stars = '⭐' * predicted_rating
    print(f"\n{'─' * 50}")
    print(f"Review: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    print(f"Dự đoán: {predicted_rating} sao {stars}")
    print(f"Xác suất:")
    for i, p in enumerate(probs):
        bar = '█' * int(p * 30)
        print(f"  {i + 1} sao: {p:.4f} |{bar}")
    print(f"{'─' * 50}")

    return predicted_rating, probs


# ==============================================================================
# 8. LƯU / LOAD MÔ HÌNH
# ==============================================================================

def save_model(model, tokenizer):
    """Lưu mô hình và tokenizer vào /kaggle/working/"""
    save_dir = os.path.join(OUTPUT_DIR, 'saved_model')
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'bilstm_attention_model.keras')
    model.save(model_path)
    print(f"      ✓ Đã lưu mô hình: {model_path}")

    tokenizer_path = os.path.join(save_dir, 'tokenizer.json')
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    print(f"      ✓ Đã lưu tokenizer: {tokenizer_path}")


# ==============================================================================
# 9. CHẠY TOÀN BỘ PIPELINE (Kaggle: bấm Run All là chạy)
# ==============================================================================

print("╔" + "═" * 58 + "╗")
print("║   SENTIMENT ANALYSIS - BiLSTM + Attention                ║")
print("║   Amazon Reviews Dataset - 5 Classes (1-5 sao)           ║")
print("║   Nhóm 5 - Nhập Môn AI                                   ║")
print("╚" + "═" * 58 + "╝")

# ===== 1. LOAD & TIỀN XỬ LÝ =====
X_train, X_val, y_train, y_val, tokenizer, vocab_size = load_and_preprocess_data()

# ===== 2. XÂY DỰNG MÔ HÌNH =====
model = build_bilstm_attention_model(vocab_size=vocab_size)

# ===== 3. HUẤN LUYỆN =====
history = train_model(model, X_train, y_train, X_val, y_val)

# ===== 4. ĐÁNH GIÁ =====
acc, report, cm = evaluate_model(model, X_val, y_val, history)

# ===== 5. THỬ NGHIỆM DỰ ĐOÁN =====
print("\n" + "=" * 60)
print("BƯỚC 5: THỬ NGHIỆM DỰ ĐOÁN")
print("=" * 60)

test_reviews = [
    "This is the best product I have ever bought! Amazing quality!",
    "Decent product, works fine but nothing extraordinary.",
    "Terrible! Broke on the first day. Total waste of money.",
    "Good value for the price. I would recommend it.",
    "Not great, not terrible. Just average.",
]

for review in test_reviews:
    predict_sentiment(model, tokenizer, review)

# ===== 6. LƯU MÔ HÌNH =====
print("\n" + "=" * 60)
print("BƯỚC 6: LƯU MÔ HÌNH")
print("=" * 60)
save_model(model, tokenizer)

print("\n╔" + "═" * 58 + "╗")
print("║   ✓ PIPELINE HOÀN TẤT!                                   ║")
print("╚" + "═" * 58 + "╝")
