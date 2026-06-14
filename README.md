# Sentiment Analysis on Amazon Reviews: BiLSTM + Attention vs Transformer

Dự án Mini Project môn học Nhập môn AI (IT3160).
Mục tiêu của dự án là xây dựng, đánh giá và so sánh hai kiến trúc Deep Learning phổ biến trong xử lý ngôn ngữ tự nhiên (NLP) là **BiLSTM kết hợp Attention** và **Transformer** cho bài toán phân tích cảm xúc (Sentiment Analysis) trên tập dữ liệu Amazon Reviews (phân loại 1-5 sao).

## 🧠 Kiến trúc Mô Hình

Dự án triển khai hai mô hình độc lập:

### 1. BiLSTM + Attention
- **File chạy**: `sentiment_analysis.py`
- **Kiến trúc chi tiết**: Xem [Architecture.md](Architecture.md)
- **Đặc điểm**:
  - Dữ liệu đi qua lớp Embedding.
  - Sử dụng mạng LSTM hai chiều (Bidirectional LSTM) để học ngữ cảnh từ cả quá khứ và tương lai của chuỗi văn bản.
  - Cơ chế **Custom Attention** được áp dụng để gán trọng số lớn hơn cho những từ đóng vai trò quan trọng trong việc quyết định cảm xúc của câu (ví dụ: "tuyệt vời", "tệ hại").

### 2. Transformer
- **File chạy**: `transformer_sentiment.py`
- **Kiến trúc chi tiết**: Xem [Transformer_Architecture.md](Transformer_Architecture.md)
- **Đặc điểm**:
  - Sử dụng kiến trúc cốt lõi của Transformer thay thế cho kiến trúc RNN truyền thống.
  - Áp dụng **Positional Embedding** song song với Word Embedding vì Transformer xử lý toàn bộ chuỗi cùng một lúc chứ không tuần tự.
  - Sử dụng khối **Transformer Block** với **Multi-Head Attention** (4 heads), giúp mô hình tập trung vào nhiều phần khác nhau của câu đồng thời.

## ⚖️ So sánh: BiLSTM+Attention vs Transformer

| Tiêu chí | BiLSTM + Attention | Transformer |
| :--- | :--- | :--- |
| **Cách xử lý dữ liệu** | Tuần tự (Sequential). Đọc từng từ một từ trái qua phải và ngược lại. | Song song (Parallel). Xử lý toàn bộ các từ trong câu cùng lúc. |
| **Cơ chế Attention** | Dùng Custom Attention 1 chiều để tóm tắt ngữ cảnh. | Dùng Multi-Head Self-Attention để tự học mối quan hệ của tất cả các cặp từ. |
| **Khả năng nắm bắt văn bản dài** | Gặp khó khăn khi chuỗi quá dài (dù có LSTM và Attention nhưng vẫn có hiện tượng thắt nút cổ chai). | Rất tốt trong việc nắm bắt ngữ cảnh ở xa nhau nhờ Self-Attention. |
| **Tốc độ huấn luyện** | Chậm hơn (do xử lý tuần tự nên khó tối ưu hóa tính toán song song phần cứng). | Nhanh hơn (đặc biệt khi có GPU) vì tính toán hoàn toàn song song (Matrix Multiplication). |
| **Số lượng tham số** | Thường ít tham số hơn. Dễ hội tụ trên dataset nhỏ. | Khá nhiều tham số. Cần lượng dữ liệu lớn và Dropout mạnh tay để tránh Overfitting. |
| **Sự phụ thuộc vị trí** | Có sẵn thông qua cách đọc tuần tự của RNN. | Phải thêm thủ công bằng Positional Embedding. |

## 🛠 Cách chạy chương trình

Bạn có thể chạy riêng lẻ từng mô hình bằng lệnh:

```bash
# Chạy mô hình BiLSTM + Attention
python sentiment_analysis.py

# Chạy mô hình Transformer
python transformer_sentiment.py
```

Sau khi chạy xong, mô hình đã huấn luyện, tokenizer và các biểu đồ đánh giá (Accuracy, Loss, Confusion Matrix) sẽ lưu trong thư mục `outputs/`.
