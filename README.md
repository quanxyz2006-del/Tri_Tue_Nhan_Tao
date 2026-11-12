# Dịch Anh–Việt bằng biLSTM

Ứng dụng đề tài 23: biLSTM Encoder → Linear + tanh → LSTM Decoder, dùng PyTorch và CustomTkinter. Bản chia sẻ có sẵn checkpoint và tự nạp khi mở, không cần huấn luyện lại để dịch.

## Chạy nhanh trên Windows

1. Cài **Python 64-bit**, nên dùng Python 3.12 từ https://www.python.org/downloads/. Bật **Add Python to PATH**, giữ thành phần Tcl/Tk.
2. Giải nén toàn bộ ZIP vào thư mục có quyền ghi; không chạy ngay bên trong ZIP.
3. Nhấp đúp [START_WINDOWS.bat](START_WINDOWS.bat).
4. Lần đầu cần Internet để tải thư viện. Đợi cửa sổ ứng dụng và trạng thái **Hoàn tất.**
5. Nhập câu tiếng Anh ngắn rồi nhấn **Dịch Anh → Việt**.

Lần sau chỉ cần nhấp đúp tệp trên. Khi đã cài đủ thư viện, dịch chạy tại máy, không cần Internet, tài khoản API hoặc GPU NVIDIA.

## macOS / Linux

Cài Python có Tcl/Tk, mở terminal trong thư mục giải nén và chạy:

```sh
sh start_macos_linux.sh
```

Hoặc `python3 run.py`. Ubuntu/Debian nếu thiếu Tk/venv:

```sh
sudo apt install python3 python3-venv python3-tk
```

macOS có thể dùng bộ cài Python từ python.org có Tcl/Tk. Linux cần desktop/display để mở GUI. Cần Python từ 3.11 và hệ điều hành/CPU được PyTorch hỗ trợ; không cam kết mọi thiết bị đều chạy được. Python 3.12 64-bit là lựa chọn đề xuất cho máy mới. Đã kiểm thử thực tế Windows/Python 3.14.4; chưa kiểm thử trực tiếp macOS/Linux.

## Checkpoint có sẵn

Lần huấn luyện cuối cùng ngày 12/11/2026: 6.391 cặp train, 806 validation, 802 test;
86 epoch trong 724,95 giây, khôi phục epoch 7 (validation loss 3,2238).
BLEU-1/2/3/4 trên test lần lượt **37,99 / 21,24 / 12,51 / 7,74** (thang 0–100).
Các ví dụ kiểm thử còn sai nghĩa đáng kể; kết quả này chưa chứng minh chất lượng
dịch tổng quát tốt và chưa có phép so sánh đối chứng với mô hình cũ trên cùng tập.

- Mô hình tự nạp: [bilstm.pt](artifacts/basic_15min/bilstm.pt).
- BLEU và câu dịch kiểm thử: [evaluation.json](artifacts/basic_15min/evaluation.json).
- Lịch sử loss/accuracy: [bilstm_metrics.json](artifacts/basic_15min/bilstm_metrics.json).

Nếu không tự nạp, chọn **Nạp checkpoint** rồi chọn tệp trên. Checkpoint chứa cấu hình, trọng số và từ điển; không cần pickle riêng. Giữ nguyên cấu trúc thư mục khi chuyển máy.

Đây là mô hình thử nghiệm trên khoảng 8.000 cặp câu ngắn, chưa phải công cụ dịch tổng quát. Có thể dịch sai, lặp từ hoặc xuất `<UNK>`. Accuracy khi huấn luyện không phải tỷ lệ câu dịch đúng; xem ví dụ thực tế trong báo cáo.

## Cài thủ công và xử lý lỗi

Windows, PowerShell tại thư mục dự án:

```powershell
py -3 -m venv .venv-run
.\.venv-run\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-run\Scripts\python.exe app_gui.py
```

macOS/Linux:

```sh
python3 -m venv .venv-run
.venv-run/bin/python -m pip install -r requirements.txt
.venv-run/bin/python app_gui.py
```

- Không tìm thấy Python: cài Python rồi mở lại terminal.
- Lỗi tải thư viện: kiểm tra mạng/proxy rồi chạy lại.
- Lỗi `tkinter`: cài Tcl/Tk hoặc sửa bộ cài Python.
- Không có bản PyTorch phù hợp: dùng Python 3.12 64-bit trên hệ điều hành hỗ trợ.
- Đã chép môi trường từ máy cũ: xóa riêng `.venv-run`, chạy lại bộ khởi động; giữ thư mục `artifacts`.

Không chuyển môi trường ảo giữa các máy. [run.py](run.py) tạo môi trường mới tại máy nhận. ZIP không chứa môi trường/cache của máy phát triển. Lần cài đầu cần dung lượng cho Python và các thư viện, lớn hơn đáng kể so với ZIP mã nguồn.

## Huấn luyện thêm (tùy chọn)

Trong GUI chọn [eng-vie-basic.txt](data/eng-vie-basic.txt), chỉnh tham số và nhấn **Huấn luyện biLSTM**. Đây là huấn luyện từ đầu, không tiếp tục optimizer của checkpoint. **Chạy Ablation Study** so sánh Encoder LSTM một chiều với biLSTM.

Chạy thí nghiệm ngắn:

```powershell
.\.venv-run\Scripts\python.exe train_basic.py
```

macOS/Linux dùng `.venv-run/bin/python train_basic.py`.
Script dùng embedding 96, hidden 192, batch 64, learning rate 0.003, max length 18, seed 42, tối đa 100 epoch và ngân sách 720 giây, kiểm tra sau mỗi epoch. Thời gian đánh giá/cài thư viện ngoài ngân sách; máy khác có thể chạy số epoch khác nhau. Script ghi đè `artifacts/basic_15min`, hãy sao lưu nếu muốn giữ mô hình đi kèm.

Huấn luyện GUI ghi `artifacts/bilstm.pt` hoặc `artifacts/lstm.pt`, không thay checkpoint tự nạp. Để dùng mô hình mới sau khi mở lại, chọn **Nạp checkpoint**.

## Dữ liệu và đánh giá

- [eng-vie.txt](data/eng-vie.txt): 50 cặp viết thủ công để kiểm thử nhanh.
- [eng-vie-basic.txt](data/eng-vie-basic.txt): tối đa 8.000 cặp từ [ManyThings/Tatoeba](https://www.manythings.org/anki/), bản nguồn ngày 2026-02-13. Câu Anh 2–10 token, câu Việt 2–16 token; giữ cột ghi công.
- Nguồn tải: https://www.manythings.org/anki/vie-eng.zip. Xem [thông tin nguồn và giấy phép](data/manythings-vie/_about.txt), CC BY 2.0. Dữ liệu có thể có lỗi, chưa được rà soát thủ công toàn bộ.

Định dạng `English<TAB>Vietnamese<TAB>Attribution`. Chuẩn hóa Unicode NFC, chữ thường, tách từ/âm tiết và dấu câu cơ bản. Loại cặp trùng; mọi bản dịch cùng câu nguồn ở cùng tập. Chia khoảng 80/10/10 cho train/validation/test; từ điển chỉ xây trên train. Cần ít nhất 10 câu nguồn khác nhau để chia tập.

Teacher forcing khi train; cross entropy/accuracy bỏ PAD; Adam, gradient clipping norm 1.0. Encoder dùng packed sequences. Greedy decoding bỏ PAD/SOS, dừng ở EOS hoặc giới hạn độ dài. Hai lớp Linear+tanh riêng chiếu hidden/cell state về kích thước Decoder.

Khôi phục epoch có validation loss tốt nhất trước khi đánh giá test. BLEU-1…4 là NLTK corpus BLEU, smoothing method1, thang 0–100; không cần tải corpus NLTK. Ablation giữ cùng split, seed, từ điển và ngân sách, nhưng số tham số hai mô hình khác nhau. Không kết luận biLSTM tốt hơn baseline chỉ từ lần chạy này.

## Kiểm thử và đóng gói

```powershell
.\.venv-run\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-run\Scripts\python.exe package_project.py
```

Kiểm thử GUI cần desktop. [src/](src/) chứa xử lý dữ liệu và mô hình; [app_gui.py](app_gui.py) quản lý giao diện và worker; [main.py](main.py) hỗ trợ CLI; [tests/](tests/) chứa kiểm thử hồi quy.

ZIP trong `dist/` chỉ gồm mã cần thiết, dữ liệu chọn lọc, giấy phép, hướng dẫn, kiểm thử và checkpoint mới. Không chứa môi trường ảo, cache hoặc checkpoint thử cũ.
