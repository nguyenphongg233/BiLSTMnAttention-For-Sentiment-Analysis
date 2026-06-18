# Chi tiết implementation: RNN, BiLSTM + Attention và Transformer

## 1. Mục tiêu và thiết lập chung

Ba mô hình giải cùng một bài toán phân loại Amazon Reviews thành 5 lớp tương ứng
rating 1–5 sao. Để phép so sánh có ý nghĩa, cả ba dùng chung:

- Hàm làm sạch: lowercase, bỏ HTML, ký tự ngoài chữ/số và khoảng trắng thừa.
- Split phân tầng 80% train, 20% validation với seed 42.
- Tokenizer tối đa 50.000 từ, có token `<OOV>`, chỉ fit trên tập train để tránh
  data leakage.
- Chuỗi dài tối đa 200 token, post-padding và post-truncation.
- Embedding học từ đầu, 128 chiều; token `0` là padding và được mask.
- Output 5 nút Softmax; loss `sparse_categorical_crossentropy`; optimizer Adam.
- Batch size 128, tối đa 10 epoch, EarlyStopping patience 3,
  ReduceLROnPlateau patience 2.
- Class weight cân bằng mặc định do phân bố rating thường lệch. Có thể tắt bằng
  `USE_CLASS_WEIGHTS=0`.
- Metrics: accuracy, macro precision/recall/F1, weighted F1, rating MAE,
  confusion matrix, F1 từng lớp, số tham số và thời gian chạy.

Các giá trị có thể đổi bằng biến môi trường như `MAX_WORDS`, `MAX_LEN`,
`BATCH_SIZE`, `EPOCHS`, `MAX_SAMPLES` và `OUTPUT_DIR`.

## 2. SimpleRNN

Luồng: `Embedding → Dropout → SimpleRNN(128) → Dense(64) → Dropout → Softmax(5)`.

SimpleRNN là baseline tuần tự và một chiều. Hidden state được cập nhật theo từng
token, nên kiến trúc gọn nhưng dễ gặp vanishing gradient khi phụ thuộc ngữ cảnh
dài. Padding được bỏ qua nhờ mask từ Embedding.

Tham số riêng mặc định:

| Tham số | Giá trị |
|---|---:|
| RNN units | 128 |
| Dense hidden | 64 |
| Dropout | 0.3 |

## 3. BiLSTM + Attention

Luồng:
`Embedding → Dropout → Bidirectional LSTM(128 mỗi chiều) → Dropout → additive attention → Dense(64) → Dropout → Softmax(5)`.

BiLSTM tạo hidden state từ cả chiều thuận và nghịch. Additive attention học một
score cho mỗi timestep, softmax các score rồi lấy tổng có trọng số. Mask được áp
vào score trước softmax, vì vậy các vị trí padding không đóng góp vào context
vector. Đây là phần đã được chỉnh so với implementation ban đầu.

Kiến trúc này phù hợp với proposal: có Embedding, BiLSTM, attention tổng hợp ngữ
cảnh, Dense và Softmax 5 lớp; đồng thời có shuffle, EarlyStopping và các metrics
được yêu cầu.

| Tham số | Giá trị |
|---|---:|
| LSTM units mỗi chiều | 128 |
| Attention | Additive, trainable |
| Dense hidden | 64 |
| Dropout | 0.3 |

## 4. Transformer encoder

Luồng:
`Token + learned positional embedding → Transformer encoder block → masked global average pooling → Dense(64) → Softmax(5)`.

Encoder block gồm multi-head self-attention, residual connection + layer
normalization, feed-forward network, residual connection + layer normalization.
Mặc định có 4 head; với embedding 128, `key_dim=32` cho mỗi head. Implementation
ban đầu đặt `key_dim=128`, tức mỗi head rộng bằng toàn embedding và làm projection
lớn hơn dự kiến. Phiên bản hiện tại cũng truyền padding mask vào self-attention
và pooling, tránh để các token `0` làm nhiễu biểu diễn câu.

| Tham số | Giá trị |
|---|---:|
| Encoder blocks | 1 |
| Attention heads | 4 |
| Dimension mỗi head | 32 |
| Feed-forward dimension | 256 |
| Dense hidden | 64 |
| Dropout | 0.3 |

Đây là Transformer encoder huấn luyện từ đầu, không phải mô hình pretrained như
BERT. Nó là cách áp dụng Transformer hợp lệ cho bài toán classification, nhưng
thường cần nhiều dữ liệu hơn recurrent model để học embedding/ngữ cảnh tốt.

## 5. So sánh định tính

| Tiêu chí | SimpleRNN | BiLSTM + Attention | Transformer |
|---|---|---|---|
| Điểm mạnh | Nhỏ, dễ hiểu, baseline tốt | Ngữ cảnh hai chiều; attention dễ diễn giải | Song song tốt trên GPU; quan hệ xa trực tiếp |
| Điểm yếu | Quên ngữ cảnh dài; train tuần tự | Vẫn tuần tự; chậm hơn khi chuỗi dài | Self-attention tốn bộ nhớ theo bình phương độ dài |
| Dữ liệu nhỏ/vừa | Khá phù hợp làm baseline | Thường ổn định nhất trong ba mô hình train từ đầu | Dễ overfit nếu dữ liệu ít |
| Văn bản dài | Yếu | Tốt hơn RNN | Mạnh về dependency xa nhưng chi phí cao |
| Khả năng giải thích | Thấp | Có thể quan sát attention weights | Có nhiều attention map nhưng diễn giải phức tạp |
| Song song hóa | Thấp | Thấp | Cao |

Không nên khẳng định trước mô hình nào tốt nhất. Kết luận thực nghiệm phải dựa
trên `Macro F1`, confusion matrix, độ lệch giữa các lớp và chi phí tính toán từ
cùng một lần chạy.

## 6. Artifact và báo cáo tự động

Mỗi pipeline ghi vào `outputs/<model>/`:

- `metrics.json`, `history.csv`, `classification_report.csv`;
- `confusion_matrix.csv`, `predictions.csv`;
- biểu đồ training/confusion matrix;
- model `.keras` và `tokenizer.json`.

Sau khi train đủ ba model, `compare_models.py` tạo:

- bảng CSV và báo cáo Markdown;
- biểu đồ quality metrics;
- F1 theo từng rating;
- so sánh tham số, thời gian train và inference;
- validation curves của cả ba model.

Thời gian chỉ so sánh được khi cả ba model chạy trên cùng loại accelerator và
cùng điều kiện Kaggle.
