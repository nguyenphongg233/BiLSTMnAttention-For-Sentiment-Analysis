#!/bin/bash
set -e

# Đảm bảo bạn đang chạy trong đúng môi trường ảo (nếu có)
PYTHON_CMD="python"
if [ -d "venv" ]; then
    PYTHON_CMD="venv/bin/python"
fi

echo "=========================================="
echo "Bắt đầu chạy Pipeline Amazon Reviews"
echo "=========================================="

echo "[1/4] Đang huấn luyện SimpleRNN..."
$PYTHON_CMD "RNN/rnn_sentiment.py"

echo "[2/4] Đang huấn luyện BiLSTM + Attention..."
$PYTHON_CMD "BiLSTMAttention/BiLSTM_Attention.py"

echo "[3/4] Đang huấn luyện Transformer Encoder..."
$PYTHON_CMD "Transformer/transformer_sentiment.py"

echo "[4/4] So sánh và đánh giá các mô hình..."
$PYTHON_CMD "compare_models.py"

echo "=========================================="
echo "Hoàn tất! Các báo cáo và mô hình đã được lưu trong thư mục outputs/"
echo "=========================================="
