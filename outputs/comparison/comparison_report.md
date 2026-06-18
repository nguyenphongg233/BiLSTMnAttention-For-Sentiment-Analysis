# Kết quả so sánh thực nghiệm

| Model               |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   Rating MAE |   Parameters |   Epochs |   Training seconds |   Inference ms/sample |
|:--------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------:|-------------:|---------:|-------------------:|----------------------:|
| SimpleRNN           |     0.6021 |            0.3774 |         0.3758 |     0.363  |        0.6462 |       0.5859 |    3,460,997 |        9 |              209.8 |                1.4968 |
| BiLSTM + Attention  |     0.6712 |            0.474  |         0.497  |     0.4643 |        0.7153 |       0.4506 |    3,765,509 |        5 |              157.7 |                0.7941 |
| Transformer Encoder |     0.7439 |            0.4754 |         0.4439 |     0.4375 |        0.7447 |       0.4075 |    3,586,181 |        5 |               50.8 |                0.2257 |

- Macro F1 cao nhất: **BiLSTM + Attention**.
- Inference nhanh nhất trong lần chạy này: **Transformer Encoder**.
- Ít tham số nhất: **SimpleRNN**.

> Các kết luận trên chỉ áp dụng cho cùng split, cấu hình và phần cứng của lần chạy này.
