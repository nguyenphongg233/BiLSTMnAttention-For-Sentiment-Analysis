# Sentiment Analysis on Amazon Reviews: RNN vs BiLSTM + Attention vs Transformer

Dự án Mini Project môn học Nhập môn AI (IT3160).
Mục tiêu của dự án là xây dựng, đánh giá và so sánh **SimpleRNN**,
**BiLSTM kết hợp Attention** và **Transformer** cho bài toán phân loại đánh giá
Amazon Reviews thành 5 lớp (1-5 sao).

## 🧠 Kiến trúc Mô Hình

Dự án dùng chung một pipeline tiền xử lý và huấn luyện trong
`sentiment_pipeline.py`. Điều này bảo đảm ba mô hình được so sánh trên đúng cùng
một train/validation split.

### 1. SimpleRNN

- **File chạy**: `RNN/rnn_sentiment.py`
- Embedding có padding mask, spatial dropout, L2 regularization và gradient clipping.

### 2. BiLSTM + Attention

- **File chạy**: `BiLSTM +  Attention/BiLSTM_Attention.py`
- **Kiến trúc chi tiết**: Xem `BiLSTM +  Attention/Architecture.md`
- **Đặc điểm**:
  - Dữ liệu đi qua lớp Embedding.
  - Sử dụng mạng LSTM hai chiều (Bidirectional LSTM) để học ngữ cảnh từ cả quá khứ và tương lai của chuỗi văn bản.
  - Custom Attention bỏ qua token padding thay vì học nhầm trên phần đệm.

### 3. Transformer

- **File chạy**: `Transformer/transformer_sentiment.py`
- **Kiến trúc chi tiết**: Xem `Transformer/Transformer_Architecture.md`
- **Đặc điểm**:
  - Sử dụng kiến trúc cốt lõi của Transformer thay thế cho kiến trúc RNN truyền thống.
  - Áp dụng **Positional Embedding** song song với Word Embedding vì Transformer xử lý toàn bộ chuỗi cùng một lúc chứ không tuần tự.
  - Multi-Head Attention dùng 4 head với `key_dim = embedding_dim / 4` và
    attention mask đúng cho padding.

## ⚖️ So sánh ba mô hình

| Tiêu chí | SimpleRNN | BiLSTM + Attention | Transformer |
| :--- | :--- | :--- | :--- |
| **Xử lý** | Tuần tự một chiều | Tuần tự hai chiều | Song song |
| **Attention** | Không | Masked attention | Multi-head self-attention |
| **Ngữ cảnh dài** | Hạn chế | Khá | Tốt |
| **Độ phức tạp** | Thấp | Trung bình | Cao |

## Các biện pháp ổn định training

- Loại review trùng và review có cùng nội dung nhưng nhãn xung đột.
- Split có stratify trước, sau đó mới fit tokenizer trên tập train để tránh data leakage.
- Dùng class weight có giới hạn để giảm lệch lớp nhưng tránh gradient spike.
- Adam learning rate thấp hơn, gradient clipping, L2 và dropout.
- `ReduceLROnPlateau`, `EarlyStopping` và lưu checkpoint tốt nhất theo validation loss.
- Báo cáo thêm balanced accuracy và macro F1, không chỉ accuracy.

## 🛠 Cách chạy chương trình

Khuyến nghị Python 3.11 hoặc 3.12:

```bash
python -m pip install -r requirements.txt
```

Bạn có thể chạy riêng lẻ từng mô hình bằng lệnh:

```bash
# SimpleRNN
python RNN/rnn_sentiment.py

# BiLSTM + Attention
python "BiLSTM +  Attention/BiLSTM_Attention.py"

# Transformer
python Transformer/transformer_sentiment.py
```

Có thể truyền file CSV đã tải sẵn để không gọi Kaggle:

```bash
python sentiment_pipeline.py --model bilstm --csv /path/to/Amazon_Reviews.csv
```

Kết quả của từng model được lưu riêng trong `outputs/rnn`,
`outputs/bilstm` và `outputs/transformer`.
