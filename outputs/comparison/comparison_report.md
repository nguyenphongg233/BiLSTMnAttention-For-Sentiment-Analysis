# Kết quả so sánh thực nghiệm

| Model               |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Rating MAE |   Parameters |   Epochs |   Training seconds |   Inference ms/sample |
|:--------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------:|-------------:|---------:|-------------------:|----------------------:|
| SimpleRNN           |     0.4764 |            0.2487 |         0.3074 |     0.2303 |        0.5179 |       1.245  |    2,709,393 |       10 |              183.2 |                1.1592 |
| BiLSTM + Attention  |     0.6714 |            0.4702 |         0.5169 |     0.4719 |        0.7176 |       0.4533 |    2,988,817 |       10 |              219.4 |                0.6065 |
| Transformer Encoder |     0.6158 |            0.4106 |         0.4473 |     0.4009 |        0.6633 |       0.5942 |    2,790,645 |        7 |               47.1 |                0.1467 |

- Macro F1 cao nhất: **BiLSTM + Attention**.
- Inference nhanh nhất trong lần chạy này: **Transformer Encoder**.
- Ít tham số nhất: **SimpleRNN**.

> Các kết luận trên chỉ áp dụng cho cùng split, cấu hình và phần cứng của lần chạy này.
