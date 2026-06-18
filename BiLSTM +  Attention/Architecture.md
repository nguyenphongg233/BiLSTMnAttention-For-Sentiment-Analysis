# Kiến Trúc Mô Hình: BiLSTM kết hợp Attention

Dưới đây là sơ đồ kiến trúc chi tiết mô phỏng lại luồng đi của dữ liệu từ khi nhập câu bình luận (Review) cho đến khi đưa ra kết quả dự đoán (1-5 sao).

```mermaid
graph TD
    %% Định nghĩa các Style để biểu đồ đẹp hơn
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000
    classDef processing fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef core fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000

    %% Các khối chính
    A("Input: Câu Review <br>(Shape: MAX_LEN = 200)")
    
    subgraph tien_xu_ly [Tiền Xử Lý]
        B("Tokenizer & Padding<br>(Chuyển chữ thành số)")
    end
    
    subgraph neural_network [Kiến trúc Mạng Neural Keras]
        C("Embedding Layer<br>(Kích thước: 50000 x 128)")
        Drop1("Dropout (Rate = 0.3)")
        
        D("Bidirectional LSTM Layer<br>(128 Units x 2 chiều)")
        Drop2("Dropout (Rate = 0.3)")
        
        E{"Masked Custom Attention Layer<br>(Trích xuất Context Vector)"}
        
        F("Dense Layer (Hidden)<br>(64 Units, ReLU)")
        Drop3("Dropout (Rate = 0.3)")
        
        G("Regression Output<br>(1 Unit, Linear)")
    end
    
    H("Kết quả Dự Đoán<br>(Rating liên tục, làm tròn về 1-5 khi đánh giá)")

    %% Luồng dữ liệu
    A --> B
    B --> C
    C --> Drop1
    Drop1 --> D
    D --> Drop2
    Drop2 --> E
    E --> F
    F --> Drop3
    Drop3 --> G
    G --> H

    %% Gán Style
    class A input
    class B processing
    class C,Drop1,D,Drop2,E,F,Drop3 core
    class G,H output
```

## Chú giải các thành phần trong sơ đồ:

1. **Input (Đầu vào)**: Đoạn văn bản bình luận của khách hàng. Sẽ được xử lý cắt gọt / đệm (padding) để đảm bảo chuỗi luôn dài đúng 200 từ (`MAX_LEN`).
2. **Embedding**: Lớp giúp chuyển đổi từng con số rời rạc (từ vựng) thành một không gian vector đa chiều (128 chiều). Giúp mô hình hiểu được ngữ nghĩa (semantics) và mối liên hệ giữa các từ.
3. **BiLSTM (Bidirectional LSTM)**: Đây là lớp LSTM đọc 2 chiều. Nó sẽ đọc câu văn từ trái sang phải, và đồng thời đọc ngược từ phải sang trái để thấu hiểu toàn bộ ngữ cảnh của câu. Nó cung cấp Output cho từng từ trong câu.
4. **Attention Mechanism (Cơ chế tập trung)**: Attention gán trọng số cho từng từ rồi nén chuỗi thành context vector. Padding mask được áp dụng trước softmax, nên các token đệm không nhận trọng số attention.
5. **Dense (Lớp kết nối đầy đủ)**: Học các đặc trưng phức tạp từ vector do Attention tạo ra thông qua hàm kích hoạt ReLU.
6. **Output**: Một nơ-ron tuyến tính dự đoán rating liên tục. Mô hình tối ưu weighted MSE; khi đánh giá classification metrics, kết quả được làm tròn và chặn về 1–5.
7. **Dropout**: Được chèn vào giữa các lớp với tỷ lệ 30% (0.3) để tắt ngẫu nhiên các nơ-ron trong quá trình huấn luyện, nhằm tránh hiện tượng mô hình học vẹt (Overfitting).
