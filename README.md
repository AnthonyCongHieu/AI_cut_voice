# Voice Aligner (stable-ts + Streamlit)

Pipeline 2 pha:
- Align (stable-ts Whisper large-v3, ưu tiên GPU; tự fallback nếu OOM): trích xuất thời gian theo từ.
- Tái tạo ngắt nghỉ: chỉ cắt theo CỤM câu kết thúc bằng dấu chấm, snap rãnh năng lượng và tìm điểm “hết sóng” trước khi cắt; chèn đúng 24 frames @30fps sau dấu chấm.

## Yêu cầu
- Python 3.11
- FFmpeg có trong PATH (bắt buộc để export MP3). Windows: dùng static build từ https://www.gyan.dev/ffmpeg/builds/ và thêm thư mục `bin` (có ffmpeg.exe) vào PATH.
- GPU khuyến nghị cho `large-v3`.

## Cài đặt
```
cd voice-aligner
py -3.11 -m venv .venv311
.\\.venv311\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Chạy UI
```
streamlit run app/ui_streamlit.py
```

## Chạy CLI
```
python app/main_processing.py --audio data/input.wav --docx data/input.docx --config config.yaml --out_json data/outputs/aligned_words.json --out_audio data/outputs/output.mp3
```

Chỉnh `config.yaml` để đổi FPS/frames/padding/snap/crossfade.

## Troubleshooting
- Thiếu ffmpeg: cài bản static (Essentials) từ gyan.dev, giải nén, thêm thư mục `bin` vào PATH, mở lại terminal.
- Lỗi CUDA/ OOM: code tự fallback model: large-v3 → large → medium → small → base. Có thể đặt `device: cpu` nếu không có GPU.
- Chậm: thử model nhỏ hơn hoặc dùng GPU.

## Acceptance
- aligned_words.json chứa `word,start,end` (giây).
- Chỉ cắt sau dấu chấm khi “hết sóng” (snap + seek end-silence).
- Mỗi dấu chấm → nghỉ đúng 24 frames @30fps (~800ms).
- Dấu khác chỉ ngắt 5–7 frames khi an toàn; mặc định không ép cắt theo dấu phẩy.
- Case “xin chào, tôi tên là” không dính chữ.
- Streamlit UI: upload/paste → xử lý → nghe/tải OK.
- CLI chạy được & sinh output.
