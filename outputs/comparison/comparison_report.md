# Kết quả so sánh thực nghiệm

| Model               |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Rating MAE |   Rating MSE |   Parameters |   Epochs |   Training seconds |   Inference ms/sample |
|:--------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------:|-------------:|-------------:|---------:|-------------------:|----------------------:|
| SimpleRNN           |     0.1566 |            0.3071 |         0.2329 |     0.1189 |        0.1839 |       1.2838 |       0      |    3,460,997 |        4 |              133.8 |                0.3691 |
| BiLSTM + Attention  |     0.4876 |            0.437  |         0.4263 |     0.3478 |        0.5682 |       0.6525 |       0.9816 |    3,765,249 |        7 |              148.5 |                0.5718 |
| Transformer Encoder |     0.1701 |            0.3959 |         0.3408 |     0.2064 |        0.1657 |       1.2031 |       1.9747 |    3,585,921 |        4 |               26.4 |                0.1489 |

- Macro F1 cao nhất: **BiLSTM + Attention**.
- Inference nhanh nhất trong lần chạy này: **Transformer Encoder**.
- Ít tham số nhất: **SimpleRNN**.

> Các kết luận trên chỉ áp dụng cho cùng split, cấu hình và phần cứng của lần chạy này.
