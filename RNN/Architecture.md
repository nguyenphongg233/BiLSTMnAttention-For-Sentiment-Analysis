# Kiến trúc SimpleRNN

```mermaid
flowchart LR
    A[Review text] --> B[Tokenizer + padding, 200 tokens]
    B --> C[Embedding, 128]
    C --> D[Dropout, 0.3]
    D --> E[SimpleRNN, 128 units]
    E --> F[Dense, 64, ReLU]
    F --> G[Dropout, 0.3]
    G --> H[Dense, 1, Linear]
```

SimpleRNN là baseline regression tuần tự cơ bản. `mask_zero=True` giúp lớp RNN
bỏ qua token padding. Đầu ra liên tục được tối ưu bằng weighted MSE, sau đó làm
tròn và chặn về 1–5 khi tính metric theo lớp.
