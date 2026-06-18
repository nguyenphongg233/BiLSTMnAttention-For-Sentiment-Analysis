# Kết quả so sánh thực nghiệm

| Model               |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Rating MAE |   Rating MSE | Parameters   |   Epochs |   Training seconds |   Inference ms/sample |
|:--------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------:|-------------:|:-------------|---------:|-------------------:|----------------------:|
| SimpleRNN           |     0.3394 |            0.4254 |         0.391  |     0.2718 |        0.4143 |       0.815  |       1.0555 | 3,460,737    |       10 |              441.3 |                0.1833 |
| BiLSTM + Attention  |     0.3092 |            0.4279 |         0.4017 |     0.2881 |        0.3869 |       0.8552 |       1.1224 | 3,765,249    |        5 |               45.9 |                0.6309 |
| Transformer Encoder |     0.458  |            0.4325 |         0.4322 |     0.3536 |        0.5489 |       0.6765 |       0.9011 | 3,585,921    |        7 |               35.3 |                0.1335 |

- Macro F1 cao nhất: **Transformer Encoder**.
- Inference nhanh nhất trong lần chạy này: **Transformer Encoder**.
- Ít tham số nhất: **SimpleRNN**.

> Các kết luận trên chỉ áp dụng cho cùng split, cấu hình và phần cứng của lần chạy này.
