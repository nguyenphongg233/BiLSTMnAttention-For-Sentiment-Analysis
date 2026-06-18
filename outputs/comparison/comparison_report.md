# Kết quả so sánh thực nghiệm

| Model               |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Rating MAE | Parameters   |   Epochs |   Training seconds |   Inference ms/sample |
|:--------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------:|:-------------|---------:|-------------------:|----------------------:|
| SimpleRNN           |     0.1566 |            0.3071 |         0.2329 |     0.1189 |        0.1839 |       1.2838 | 3,460,997    |        4 |              133.8 |                0.3691 |
| BiLSTM + Attention  |     0.6871 |            0.4566 |         0.4727 |     0.4546 |        0.7166 |       0.4751 | 3,765,509    |       10 |               59.5 |                0.3309 |
| Transformer Encoder |     0.6895 |            0.4738 |         0.4873 |     0.4635 |        0.7195 |       0.4732 | 3,586,181    |        6 |               26.2 |                0.0987 |

- Macro F1 cao nhất: **Transformer Encoder**.
- Inference nhanh nhất trong lần chạy này: **Transformer Encoder**.
- Ít tham số nhất: **SimpleRNN**.

> Các kết luận trên chỉ áp dụng cho cùng split, cấu hình và phần cứng của lần chạy này.
