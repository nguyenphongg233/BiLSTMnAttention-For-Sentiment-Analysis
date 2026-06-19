# Kiến trúc SimpleRNN

```mermaid
flowchart LR
    A[Review text] --> B[Tokenizer + padding, 200 tokens]
    B --> C[Embedding GloVe 100d]
    C --> D[SpatialDropout1D, 0.4-0.5]
    D --> E[SimpleRNN, 128 units, unroll=True]
    E --> F[Dense, 48, ReLU, L2 Reg]
    F --> G[Dropout, 0.4-0.5]
    G --> H[Dense, 5, Softmax, L2 Reg]
```

SimpleRNN là baseline tuần tự cơ bản. `mask_zero=True` giúp lớp RNN bỏ qua token
padding. Mô hình ít phức tạp hơn BiLSTM + Attention và Transformer, nhưng thường
khó giữ thông tin dài hạn do vanishing gradient. Gần đây, mô hình đã được bổ sung `unroll=True` để giải phóng khả năng tính toán song song của GPU cho vòng lặp, đồng thời áp dụng L2, Dropout và GloVe 100d để tránh Overfitting mạnh mẽ.
