# Kiến Trúc Mô Hình: Transformer

Dưới đây là sơ đồ kiến trúc chi tiết mô phỏng lại luồng đi của dữ liệu từ khi nhập câu bình luận (Review) cho đến khi đưa ra kết quả dự đoán (1-5 sao) sử dụng mô hình Transformer.

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
        C("Token & Position Embedding Layer<br>(Kích thước: 50000 x 128)")
        
        subgraph transformer_block [Transformer Block]
            D("Masked Multi-Head Attention<br>(4 Heads, Key Dim: 32)")
            E("Add & Layer Normalization")
            F("Feed Forward Network<br>(Dense 128 -> Dense 128)")
            G("Add & Layer Normalization")
        end
        
        H("Masked Global Average Pooling 1D")
        Drop1("Dropout (Rate = 0.3)")
        
        I("Dense Layer (Hidden)<br>(64 Units, ReLU)")
        Drop2("Dropout (Rate = 0.3)")
        
        J("Output Layer<br>(5 Units, Softmax)")
    end
    
    K("Kết quả Dự Đoán<br>(Xác suất từ 1 sao đến 5 sao)")

    %% Luồng dữ liệu
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> Drop1
    Drop1 --> I
    I --> Drop2
    Drop2 --> J
    J --> K

    %% Gán Style
    class A input
    class B processing
    class C,D,E,F,G,H,Drop1,I,Drop2 core
    class J,K output
```

## Chú giải các thành phần trong sơ đồ:

1. **Input (Đầu vào)**: Đoạn văn bản bình luận của khách hàng. Sẽ được xử lý cắt gọt / đệm (padding) để đảm bảo chuỗi luôn dài đúng 200 từ (`MAX_LEN`).
2. **Token & Position Embedding**: Kết hợp Word Embedding (nhúng từ vựng thành vector 128 chiều) và Positional Embedding (lưu trữ thông tin vị trí của từ trong câu). Do Transformer xử lý dữ liệu song song (không tuần tự như RNN/LSTM), nó cần thông tin vị trí để hiểu thứ tự của các từ.
3. **Transformer Block**: Khối xử lý cốt lõi của Transformer:
   - **Multi-Head Attention**: Gồm 4 "đầu" (heads) hoạt động song song, mỗi head có chiều 32 để tổng chiều biểu diễn là 128. Padding mask được truyền vào attention để token đệm không ảnh hưởng kết quả.
   - **Feed Forward Network (FFN)**: Mạng truyền thẳng giúp trích xuất các đặc trưng phi tuyến tính sâu hơn từ kết quả của quá trình Attention.
   - **Add & Layer Normalization**: Kỹ thuật chuẩn hóa và cộng kết nối thặng dư (residual connection) giúp mô hình huấn luyện ổn định và tránh được hiện tượng triệt tiêu đạo hàm (vanishing gradient).
4. **Global Average Pooling 1D**: Gom đặc trưng chuỗi thành một vector duy nhất và bỏ qua các vị trí padding thông qua mask.
5. **Dense (Lớp kết nối đầy đủ)**: Học các đặc trưng phức tạp cuối cùng từ vector trước khi phân loại thông qua hàm kích hoạt ReLU.
6. **Output**: Gồm 5 Nơ-ron tương ứng với số sao từ 1 đến 5. Hàm kích hoạt `Softmax` đảm bảo đầu ra là một tỷ lệ phần trăm (%), phân loại có xác suất cao nhất chính là kết quả dự đoán của mô hình. 
7. **Dropout**: Được chèn vào giữa các lớp với tỷ lệ 30% (0.3) để tắt ngẫu nhiên các nơ-ron trong quá trình huấn luyện, nhằm tránh hiện tượng mô hình học vẹt (Overfitting).
