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
python demo_models.py
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

Sau mỗi epoch, pipeline còn tính và lưu các validation metric: Macro Precision,
Macro Recall, Macro F1, Weighted F1 và Rating MAE. Mỗi thư mục model có:

- `history.csv`: toàn bộ chỉ số theo epoch;
- `training_history.png` và `epoch_metrics.png`: graph chỉ số theo epoch;
- `confusion_matrix.png`: confusion matrix sau khi train.

`compare_models.py` tạo thêm `epoch_metrics_comparison.png` để đặt đường metric
theo epoch của cả ba model trên cùng biểu đồ.

## Demo dự đoán

Sau khi đã train đủ ba model, chạy nhóm review mẫu mặc định:

```bash
python demo_models.py
```

Hoặc truyền review riêng (có thể lặp lại `--text`):

```bash
python demo_models.py \
  --text "This product is excellent and works perfectly." \
  --text "It stopped working after two days."
```

Script in validation metrics đã lưu của từng model, giá trị rating thô, rating
1–5 sao sau khi làm tròn, và sai số tuyệt đối đối với các sample mặc định.

## Kiểm tra proposal

BiLSTM + Attention hiện khớp pipeline được mô tả trong `NMAI_proposal.md`:
Embedding → BiLSTM hai chiều → Attention → Dense → Softmax 5 lớp; train bằng
Adam/cross-entropy, có shuffle, EarlyStopping và đánh giá precision/recall/F1,
accuracy, confusion matrix.

Hai lỗi kỹ thuật của phiên bản cũ đã được sửa:

- tokenizer không còn fit trên validation;
- attention bỏ qua padding thay vì gán trọng số cho token đệm.

Transformer cũng đã được bổ sung padding mask cho self-attention/pooling và dùng
dimension theo từng attention head đúng với embedding tổng.
