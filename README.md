# train-ai
# Hướng Dẫn Sử Dụng - Bộ Train AI  Dự Đoán Tài Xỉu

## Giới Thiệu
Ảnh chụp màn hình:

client

![Client](https://i.ibb.co/wZnrV51x/Screenshot-348.png)

admin 

![Admin](https://i.ibb.co/ZpwP8ZhZ/Screenshot-347.png)


Bộ ứng dụng bao gồm:
- **sv.py**: Server Flask xử lý API, huấn luyện mô hình, lưu dữ liệu vào SQLite.
- **client.py**: Ứng dụng giao diện người dùng giúp nhập dữ liệu, gửi yêu cầu dự đoán và phản hồi kết quả.
- **quanly.py**: Ứng dụng quản lý server, kiểm soát quá trình huấn luyện, xóa dữ liệu và xem thống kê chi tiết.
- **index.js**: Ưngs dụng để train Ai.

## Cài Đặt
### Yêu Cầu Hệ Thống
- Python 3.8+
- Các thư viện cần thiết:
  ```bash
  update-pip.bat
  cai-module-python.bat
  ```

### Khởi Chạy Server
```bash
run-ai.bat
run-api-train.bat
```
Mặc định, server chạy trên `http://0.0.0.0:5000`.

### Khởi Chạy Client
```bash
python client.py
```
Ứng dụng giao diện giúp gửi dữ liệu và nhận dự đoán từ server, đồng thời cung cấp giao diện phản hồi kết quả thực tế.

### Khởi Chạy Quản Lý Server
```bash
python quanly.py
```
Ứng dụng giúp kiểm soát server, huấn luyện mô hình và xem thống kê kết quả gần đây.

## API Endpoints (sv.py)
### 1. Dự đoán kết quả
- **Endpoint:** `/predict`
- **Method:** `POST`
- **Body:**
  ```json
  {
    "session": [3, 8, 12, 14, 10, 6, 9, 13, 15, 7]
  }
  ```
- **Response:**
  ```json
  {
    "prediction": "Tài",
    "confidence": 0.85,
    "stats": {
      "total_records": 1000,
      "average_total": 10.5,
      "tai_ratio": 0.55,
      "streak": 3
    }
  }
  ```

### 2. Điều khiển server
- **Endpoint:** `/control`
- **Method:** `POST`
- **Actions:**
  - `start`: Bắt đầu huấn luyện mô hình.
  - `stop`: Dừng huấn luyện.
  - `stats`: Lấy thông tin thống kê.

### 3. Gửi phản hồi kết quả thực tế
- **Endpoint:** `/feedback`
- **Method:** `POST`
- **Body:**
  ```json
  { "actual_result": 12 }
  ```
- **Response:**
  ```json
  { "message": "Feedback recorded" }
  ```

## Chức Năng Client (client.py)
- Nhập dữ liệu gần nhất của trò chơi tài xỉu.
- Gửi yêu cầu dự đoán đến server Flask.
- Hiển thị kết quả dự đoán, độ chính xác và thống kê lịch sử.
- Gửi phản hồi thực tế để cải thiện mô hình.
- Cập nhật biểu đồ xu hướng tỷ lệ tài/xỉu trong 50 lần gần nhất.

## Chức Năng Quản Lý (quanly.py)
- Bật/tắt quá trình huấn luyện mô hình AI.
- Xóa dữ liệu lịch sử trong SQLite.
- Lấy và hiển thị thống kê tổng số lượt chơi, trung bình tổng điểm, tỷ lệ tài và chuỗi tài/xỉu dài nhất.
- Cung cấp biểu đồ trực quan giúp theo dõi xu hướng kết quả gần đây.

## Cách Hoạt Động
1. **Server (`sv.py`)**:
   - Lấy dữ liệu từ API hoặc SQLite.
   - Xử lý, huấn luyện mô hình bằng Gradient Boosting, XGBoost, LightGBM, LSTM.
   - Cung cấp API dự đoán và thống kê dữ liệu.
2. **Client (`client.py`)**:
   - Người dùng nhập dữ liệu, gửi yêu cầu đến server.
   - Nhận phản hồi dự đoán và gửi phản hồi thực tế.
3. **Quản lý (`quanly.py`)**:
   - Giúp admin theo dõi quá trình server, bật/tắt huấn luyện, quản lý dữ liệu và xem thống kê.
4. **Api lấy data (`index.js`)**:
   - Train ra data về tính ngẫu nhiên.

## Ghi Chú
- Nếu server không khởi động được, kiểm tra xem cổng 5000 có đang bị chiếm dụng không.
- Dữ liệu SQLite có thể mở bằng công cụ như DB Browser for SQLite để kiểm tra.
- Khi huấn luyện mô hình, cần đảm bảo có đủ dữ liệu đầu vào.

## Liên Hệ
Nếu gặp lỗi, vui lòng kiểm tra log hoặc liên hệ nhóm phát triển.

