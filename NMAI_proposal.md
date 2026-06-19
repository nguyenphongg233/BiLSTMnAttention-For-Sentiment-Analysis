## **BÁO CÁO NH** Ậ **P MÔN AI** 

**Nhóm 5: Tr** ầ **n Lê Ng** ọ **c Tâm - 202416346 Nguy** ễ **n Phong - 202400066 Nguy** ễ **n Ng** ọ **c Tu** ấ **n Anh - 202400029** 

**Sentiment Analysis trên Amazon Reviews Dataset (Kaggle) v** ớ **i mô hình BiLSTM + Attention** 

## **1. Gi** ớ **i thi** ệ **u** 

Phân tích c ả m xúc (Sentiment Analysis) là bài toán xác đ ị nh thái đ ộ c ủ a ng ườ i dùng t ừ văn b ả n. Trong báo cáo này, chúng tôi s ử d ụ ng Amazon Reviews Dataset trên Kaggle đ ể xây d ự ng mô hình phân lo ạ i c ả m xúc d ự a trên review s ả n ph ẩ m. 

Khác v ớ i các bài toán đ ơ n gi ả n, d ữ li ệ u đ ượ c gi ữ nguyên thang đánh giá 1–5 sao, t ươ ng ứ ng v ớ i bài toán phân lo ạ i đa l ớ p (5 l ớ p), bao g ồ m c ả m ứ c trung l ậ p. 

## **2. Dataset** 

**Ngu** ồ **n:** Amazon Reviews Dataset (Kaggle) 

**Thành ph** ầ **n:** 

Review text 

Rating (1–5 sao) 

## ổ ể **3. Pipeline t ng th** 

Quy trình x ử lý g ồ m các b ướ c đ ượ c th ự c hi ệ n tu ầ n t ự đ ể đ ả m b ả o tính nh ấ t quán c ủ a d ữ li ệ u đ ầ u vào và hi ệ u qu ả c ủ a mô hình đ ầ u ra. 

## **4. Ti** ề **n x** ử **lý d** ữ **li** ệ **u** 

M ụ c tiêu chính c ủ a b ướ c này là chuy ể n văn b ả n t ừ d ạ ng thô thành d ạ ng s ố đ ể mô hình máy h ọ c có th ể x ử lý đ ượ c. 

## **Các b** ướ **c chính:** 

Chu ẩ n hóa văn b ả n (lowercase) 

- Lo ạ i b ỏ ký t ự đ ặ c bi ệ t 

- Tokenization (tách t ừ ) 

- Chuy ể n thành sequence s ố 

- Padding v ề cùng đ ộ dài 

**K** ế **t qu** ả **:** Chuy ể n đ ổ i thành công d ữ li ệ u Text → vector s ố . 

## **5. Mô hình BiLSTM + Attention** 

## **5.1. Ý t** ưở **ng** 

## **BiLSTM (Bidirectional LSTM):** 

- Đ ọ c câu theo c ả 2 chi ề u (t ừ trái sang ph ả i và ng ượ c l ạ i). 

- N ắ m b ắ t ng ữ c ả nh t ố t h ơ n so v ớ i mô hình LSTM thông th ườ ng. 

## **Attention:** 

Gán tr ọ ng s ố cho t ừ ng t ừ trong câu. 

- T ậ p trung vào các t ừ mang tính then ch ố t (các t ừ quan tr ọ ng). 

## **S** ự **k** ế **t h** ợ **p này giúp mô hình:** 

- Hi ể u toàn b ộ câu (global context). 

- Nh ấ n m ạ nh các t ừ quan tr ọ ng (key information). 

## **5.2. Ki** ế **n trúc mô hình** 

Mô hình đ ượ c xây d ự ng theo c ấ u trúc l ớ p ch ồ ng l ớ p đ ể trích xu ấ t đ ặ c tr ư ng t ừ văn b ả n m ộ t cách hi ệ u qu ả nh ấ t. 

## **5.3. Gi** ả **i thích các thành ph** ầ **n** 

**Embedding Layer:** Bi ế n các t ừ thành vector không gian nhi ề u chi ề u. 

**BiLSTM:** X ử lý chu ỗ i theo 2 chi ề u, t ạ o bi ể u di ễ n ng ữ c ả nh phong phú. 

**Attention Layer:** Tính tr ọ ng s ố cho m ỗ i timestep, t ổ ng h ợ p thành vector quan tr ọ ng nh ấ t. 

**Dense + Softmax:** D ự đoán xác su ấ t thu ộ c 1 trong 5 l ớ p c ả m xúc. 

## **6. Hu** ấ **n luy** ệ **n trên Kaggle** 

## **6.1. Môi tr** ườ **ng** 

- Kaggle Notebook 

- S ử d ụ ng GPU mi ễ n phí (T4 / P100) đ ể t ố i ư u th ờ i gian hu ấ n luy ệ n. 

## **6.2. Quy trình train** 

1. Load dataset tr ự c ti ế p t ừ Kaggle. 

2. Ti ề n x ử lý d ữ li ệ u theo các b ướ c đã nêu. 

3. Chia d ữ li ệ u thành các t ậ p: Train / Validation. 

4. Compile model v ớ i: 

   - Loss function: CrossEntropy 

   - Optimizer: Adam 

5. Th ự c hi ệ n Train: 

   - Batch size: 32–128 

   - Epoch: 5–10 

## **6.3. L** ư **u ý** 

- S ử d ụ ng **EarlyStopping** đ ể tránh hi ệ n t ượ ng overfitting (quá kh ớ p). 

- Th ự c hi ệ n **Shuffle** d ữ li ệ u đ ể đ ả m b ả o tính khách quan. 

- T ậ n d ụ ng GPU đ ể tăng t ố c đ ộ training. 

## **7. Metrics đánh giá** 

Vì bài toán là phân lo ạ i 5 l ớ p, chúng tôi s ử d ụ ng các h ệ s ố đo l ườ ng sau: 

**Accuracy:** T ỷ l ệ d ự đoán đúng trên t ổ ng th ể . 

**Precision / Recall / F1-score:** Đánh giá chi ti ế t cho t ừ ng l ớ p c ụ th ể (t ừ 1 sao đ ế n 5 sao). 

**F1-score:** Đ ặ c bi ệ t quan tr ọ ng khi t ậ p d ữ li ệ u không cân b ằ ng. 

**Confusion Matrix:** Phân tích các l ỗ i nh ầ m l ẫ n gi ữ a các l ớ p v ớ i nhau. 

## **8. Đánh giá mô hình** 

Mô hình có kh ả năng hi ể u ng ữ c ả nh r ấ t t ố t nh ờ vào ki ế n trúc BiLSTM. 

C ơ ch ế Attention giúp t ậ p trung chính xác vào các t ừ mang tính bi ể u đ ạ t c ả m xúc m ạ nh (t ừ quan tr ọ ng). 

Có hi ệ u qu ả v ượ t tr ộ i khi so sánh v ớ i các ki ế n trúc đ ơ n gi ả n h ơ n nh ư Hybrid CNN-LSTM cho cùng m ộ t bài toán. 

## **9. K** ế **t lu** ậ **n** 

Mô hình BiLSTM + Attention là m ộ t c ả i ti ế n h ợ p lý và hi ệ u qu ả cho bài toán Sentiment Analysis trên Amazon Reviews. So v ớ i các mô hình c ơ b ả n, mô hình này có kh ả năng n ắ m b ắ t ng ữ c ả nh t ố t h ơ n và t ậ p trung vào các thành ph ầ n quan tr ọ ng trong câu, t ừ đó c ả i thi ệ n đáng k ể hi ệ u qu ả phân lo ạ i đa l ớ p. 

