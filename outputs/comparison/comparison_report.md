# Kết quả so sánh thực nghiệm

| Model               |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Rating MAE |   Parameters |   Epochs |   Training seconds |   Inference ms/sample |
|:--------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------:|-------------:|---------:|-------------------:|----------------------:|
| SimpleRNN           |     0.6481 |            0.2492 |         0.2145 |     0.1845 |        0.5274 |       1.0831 |    3,460,997 |       10 |              174.2 |                0.9883 |
| BiLSTM + Attention  |     0.8155 |            0.4793 |         0.4015 |     0.3908 |        0.7583 |       0.3399 |    3,765,509 |        5 |              102.8 |                0.5403 |
| Transformer Encoder |     0.7991 |            0.478  |         0.3816 |     0.346  |        0.7364 |       0.3999 |    3,586,181 |        4 |               27.3 |                0.1372 |

- Macro F1 cao nhất: **BiLSTM + Attention**.
- Inference nhanh nhất trong lần chạy này: **Transformer Encoder**.
- Ít tham số nhất: **SimpleRNN**.

> Các kết luận trên chỉ áp dụng cho cùng split, cấu hình và phần cứng của lần chạy này.
