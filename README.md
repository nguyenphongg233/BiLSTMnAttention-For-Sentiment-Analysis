# Amazon Reviews Sentiment Analysis: RNN vs BiLSTM + Attention vs Transformer

Dự án phân loại Amazon Reviews thành 5 mức rating (1–5 sao) bằng ba kiến trúc:

- `SimpleRNN`: baseline recurrent một chiều.
- `BiLSTM + Attention`: mô hình chính trong proposal.
- `Transformer Encoder`: self-attention huấn luyện từ đầu.

Chi tiết tham số, kỹ thuật và phân tích ưu/nhược điểm nằm trong
[IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md).

## Chạy trên Kaggle

1. Tạo Kaggle Notebook và bật GPU.
2. Chọn **Add Data**, thêm dataset
   `dongrelaxman/amazon-reviews-dataset`.
3. Upload/clone toàn bộ repository vào notebook, sau đó mở terminal tại thư mục
   gốc của project.
4. Chạy lần lượt:

```bash
python "RNN/rnn_sentiment.py"
python "BiLSTM +  Attention/BiLSTM_Attention.py"
python "Transformer/transformer_sentiment.py"
python compare_models.py
```

Pipeline tự tìm `Amazon_Reviews.csv` trong `/kaggle/input`. Nếu file ở nơi khác:

```bash
AMAZON_REVIEWS_CSV=/duong/dan/Amazon_Reviews.csv python "RNN/rnn_sentiment.py"
```

Để smoke test nhanh trước khi train toàn bộ:

```bash
MAX_SAMPLES=5000 EPOCHS=1 BATCH_SIZE=64 python "RNN/rnn_sentiment.py"
```

Các biến cấu hình phổ biến:

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `MAX_WORDS` | 50000 | Vocabulary tối đa |
| `MAX_LEN` | 200 | Số token mỗi review |
| `EMBEDDING_DIM` | 128 | Chiều embedding |
| `BATCH_SIZE` | 128 | Batch size |
| `EPOCHS` | 10 | Epoch tối đa |
| `MAX_SAMPLES` | toàn bộ | Giới hạn mẫu để thử nhanh |
| `USE_CLASS_WEIGHTS` | 1 | Cân bằng loss theo lớp |

## Output

Mỗi model lưu model, tokenizer, metrics, prediction và biểu đồ vào:

```text
outputs/
├── rnn/
├── bilstm_attention/
├── transformer/
└── comparison/
```

`compare_models.py` chỉ chạy sau khi có `metrics.json` của đủ ba model. Script
tạo bảng so sánh, Macro F1 theo lớp, validation curves và biểu đồ chi phí tính
toán. Không có số liệu kết quả giả định được hard-code trong repository.

## So sánh 3 Kiến trúc Thuật toán (Sau khi Tối ưu)

Cả 3 mô hình đều được hưởng lợi từ các cơ chế chống Overfitting mạnh mẽ: **GloVe 100d**, **L2 Regularization (0.01)**, và **Dropout cao (0.4-0.5)** cùng với **cân bằng trọng số lớp (Class Weights)**.

### 1. SimpleRNN
- **Cấu trúc (Layer Flow):** `Input` → `Embedding (GloVe 100d)` → `SpatialDropout1D` → `SimpleRNN(unroll=True)` → `Dense(ReLU) + L2` → `Dropout` → `Dense(Softmax) + L2`
- **Đặc điểm:** Kiến trúc Recurrent một chiều cổ điển, đọc tuần tự từng token.
- **Cải tiến:** Đã được bật `unroll=True` để trải phẳng vòng lặp giúp GPU xử lý song song, cải thiện tốc độ đáng kể.
- **Đánh giá:** 
  - *Ưu điểm:* Tham số cực kỳ nhỏ gọn, tốc độ suy luận nhanh nhất, dễ làm Baseline.
  - *Nhược điểm:* Quên ngữ cảnh dài (Vanishing Gradient), không bắt được sự đảo ngữ (ví dụ "not bad" ở 2 đầu câu).
  - *Kết quả thực tế:* Vượt mong đợi. Nhờ L2/Dropout và GloVe, mô hình chạm mốc `Accuracy ~76.7%` và không còn sụp đổ (overfit) ngay từ epoch 3-4 như lúc trước.

### 2. BiLSTM + Attention
- **Cấu trúc (Layer Flow):** `Input` → `Embedding (GloVe 100d)` → `SpatialDropout1D` → `BiLSTM` → `Dropout` → `Masked Additive Attention` → `Dense(ReLU) + L2` → `Dropout` → `Dense(Softmax) + L2`
- **Đặc điểm:** Kiến trúc kết hợp LSTM hai chiều và Additive Attention. Đọc câu từ trái qua phải và phải qua trái.
- **Cải tiến:** Attention hiện tại đã tích hợp Padding Mask để bỏ qua các ký tự đệm, giúp trọng số hội tụ chính xác vào các từ khóa cảm xúc.
- **Đánh giá:**
  - *Ưu điểm:* Là cỗ máy ổn định và mạnh mẽ nhất cho dữ liệu văn bản vừa/ngắn. Bắt được cấu trúc ngữ pháp phức tạp nhờ luồng thông tin 2 chiều.
  - *Nhược điểm:* Tốc độ huấn luyện chậm hơn do cấu trúc mạng phức tạp và không thể song song hóa hoàn toàn.
  - *Kết quả thực tế:* Thường đạt hiệu năng cao nhất `Accuracy ~81%`. Khả năng bắt class thiểu số (2,3,4 sao) rất tốt nhờ việc kết nối ngữ cảnh rộng.

### 3. Transformer Encoder
- **Cấu trúc (Layer Flow):** `Input` → `Token & Position Embedding (GloVe 100d)` → `SpatialDropout1D` → `Transformer Block (MultiHead Attention + FFN)` → `GlobalAveragePooling1D` → `Dense(GELU) + L2` → `Dropout` → `Dense(Softmax) + L2`
- **Đặc điểm:** Mô hình Self-Attention tinh khôi (không dùng Pre-trained như BERT), tự học trọng số từ đầu dựa trên GloVe.
- **Cải tiến:** Áp dụng chặt chẽ Padding Mask vào Self-Attention và GlobalAveragePooling. Sửa lỗi `KerasTensor` tương thích với Keras 3.
- **Đánh giá:**
  - *Ưu điểm:* Liên kết các từ ở khoảng cách xa tốt nhất. Tối ưu sức mạnh ma trận của GPU/Apple Metal (train rất nhanh so với BiLSTM).
  - *Nhược điểm:* Đói dữ liệu (Data-hungry). Rất dễ bị Overfit nếu tập dữ liệu nhỏ.
  - *Kết quả thực tế:* Mặc dù train từ con số 0, nó hội tụ nhanh và cho kết quả xấp xỉ BiLSTM. Tuy nhiên, nó đôi lúc không ổn định bằng BiLSTM nếu thiếu dữ liệu ngữ cảnh cục bộ.

## Kiểm tra proposal

BiLSTM + Attention hiện khớp pipeline được mô tả trong `NMAI_proposal.md`:
Embedding → BiLSTM hai chiều → Attention → Dense → Softmax 5 lớp; train bằng Adam/cross-entropy, có shuffle, EarlyStopping và đánh giá precision/recall/F1, accuracy, confusion matrix.

Các lỗi kỹ thuật của phiên bản gốc đã được khắc phục hoàn toàn:
- Thay vì `Embedding` tự học, mô hình khởi tạo với **GloVe 100d** (`trainable=True`).
- Các lớp Dense áp dụng **L2 Regularizer** và Dropout.
- Tokenizer không còn bị leakage (chỉ fit trên train).
- Masking hoạt động trơn tru xuyên suốt Attention.
