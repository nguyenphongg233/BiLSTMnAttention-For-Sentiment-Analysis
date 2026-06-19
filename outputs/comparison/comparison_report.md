# Kết quả so sánh thực nghiệm

| Model               |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Rating MAE |   Parameters |   Epochs |   Training seconds |   Inference ms/sample |
|:--------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------:|-------------:|---------:|-------------------:|----------------------:|
| SimpleRNN           |     0.7493 |            0.2847 |         0.3252 |     0.3029 |        0.6818 |       0.6288 |    2,709,393 |       10 |              235.4 |                1.4516 |
| BiLSTM + Attention  |     0.8138 |            0.4722 |         0.3946 |     0.3727 |        0.7549 |       0.3328 |    2,988,817 |       10 |              308.5 |                0.8103 |
| Transformer Encoder |     0.7988 |            0.3026 |         0.3735 |     0.3336 |        0.7328 |       0.4009 |    2,790,645 |        6 |               56.4 |                0.1859 |

- Macro F1 cao nhất: **BiLSTM + Attention**.
- Inference nhanh nhất trong lần chạy này: **Transformer Encoder**.
- Ít tham số nhất: **SimpleRNN**.

> Các kết luận trên chỉ áp dụng cho cùng split, cấu hình và phần cứng của lần chạy này.
