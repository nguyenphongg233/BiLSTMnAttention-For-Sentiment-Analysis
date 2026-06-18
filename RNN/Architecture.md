# Kiến trúc SimpleRNN

```mermaid
flowchart LR
    A[Review text] --> B[Tokenizer + padding, 200 tokens]
    B --> C[Embedding, 128]
    C --> D[Dropout, 0.3]
    D --> E[SimpleRNN, 128 units]
    E --> F[Dense, 64, ReLU]
    F --> G[Dropout, 0.3]
    G --> H[Dense, 5, Softmax]
```

SimpleRNN là baseline tuần tự cơ bản. `mask_zero=True` giúp lớp RNN bỏ qua token
padding. Mô hình ít phức tạp hơn BiLSTM + Attention và Transformer, nhưng thường
khó giữ thông tin dài hạn do vanishing gradient.
