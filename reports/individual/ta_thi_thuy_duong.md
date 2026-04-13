# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Tạ Thị Thùy Dương  
**Vai trò trong nhóm:** Tech Lead  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Tôi đảm nhận vai trò Tech Lead và dẫn dắt Sprint 1 (Indexing) lẫn Sprint 2 (Retrieval & Grounded Answer). Công việc chính bao gồm dựng khung toàn bộ file `rag_answer.py` — từ cấu trúc hàm, luồng dữ liệu, đến cách kết nối các module lại với nhau để pipeline chạy thông suốt end-to-end.

Phần trọng tâm nhất tôi trực tiếp implement là lớp generation: viết `build_grounded_prompt()` theo 4 quy tắc evidence-only, abstain, citation, stable output; viết `call_llm()` để gọi OpenAI API với `response_format=json_object`, parse kết quả trả về gồm `answer` và `grounded_spans`. Tôi cũng thiết kế cơ chế **abstain threshold** — nếu chunk tốt nhất có score dưới ngưỡng thì không gọi LLM, tránh hallucination.

Công việc của tôi là nền để thành viên phụ trách Retrieval (retrieve_dense, hybrid, rerank) và Eval có chỗ cắm vào mà không cần đụng vào lớp generation.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Concept tôi thực sự thấm hơn sau lab là **grounded prompt và structured output**. Trước đây tôi nghĩ chỉ cần bảo LLM "đừng bịa" là đủ, nhưng thực tế không phải vậy — LLM vẫn có thể trả lời nghe có vẻ hợp lý dù context không có. Khi tôi yêu cầu LLM trả về JSON với trường `grounded_spans` (đoạn văn nguyên văn từ context đã dùng), tôi mới có thể kiểm chứng được LLM thực sự bám vào tài liệu hay không. Đây chính là cách grounding trở thành điều có thể đo lường được, chứ không chỉ là lời hứa trong prompt.

Điều thứ hai là **abstain logic**: không phải cứ có chunk retrieved là nên trả lời. Score thấp tức là vector store cũng không tìm được gì liên quan — lúc đó im lặng còn tốt hơn câu trả lời sai. Hiểu được điều này giúp tôi thiết kế pipeline có tính trung thực hơn.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Khó khăn mất thời gian nhất là lỗi **"Collection [rag_lab] does not exist"**. Ban đầu tôi tưởng là bug trong ChromaDB hoặc đường dẫn sai, mất khoảng 20 phút để nhận ra `get_collection()` sẽ ném exception ngay nếu collection chưa được tạo — tức là phải chạy `index.py` trước. Fix là đổi sang `get_or_create_collection()` và thêm kiểm tra `collection.count() == 0` để báo lỗi rõ ràng thay vì crash mơ hồ.

Điều ngạc nhiên là **`response_format={"type": "json_object"}`** của OpenAI yêu cầu prompt phải *đề cập đến từ "JSON"* thì mới hoạt động — nếu không sẽ báo lỗi API. Tôi không đọc kỹ docs phần này nên bị mắc. Sau khi thêm hướng dẫn JSON rõ ràng vào prompt, output mới ổn định và parse được.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** *"Approval Matrix để cấp quyền hệ thống là tài liệu nào?"* (q07 — độ khó: hard)

**Phân tích:**

Đây là câu hỏi dùng tên cũ "Approval Matrix" trong khi tài liệu thực tế đã đổi tên thành "Access Control SOP". Baseline (dense retrieval) trả lời **sai hoặc thiếu** vì embedding của "Approval Matrix" và "Access Control SOP" không đủ gần nhau về mặt vector — tên gọi khác nhau hoàn toàn dù ý nghĩa giống nhau.

Lỗi nằm ở **lớp retrieval**, không phải generation: nếu chunk đúng không vào top-k thì LLM không có nguyên liệu để trả lời đúng dù prompt có tốt đến đâu.

Khi chuyển sang **hybrid retrieval** (dense + BM25), BM25 bắt được từ khóa "Approval" xuất hiện trong phần giải thích của tài liệu, đẩy chunk đúng lên top-3. Kết quả: câu trả lời cải thiện rõ — LLM trích dẫn được đúng tên tài liệu mới và giải thích sự thay đổi tên.

Bài học: với corpus có alias và tên đã đổi, hybrid retrieval không phải tối ưu hóa — mà là **yêu cầu tối thiểu** để pipeline hoạt động đúng.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Tôi sẽ thử **query expansion tự động** cho những câu hỏi dùng tên cũ hoặc alias: gọi LLM sinh 2–3 cách diễn đạt thay thế rồi retrieve song song, merge kết quả bằng RRF. Eval cho thấy q07 (alias query) và q10 (thiếu context đặc biệt) là hai điểm yếu nhất của pipeline hiện tại — expansion nhắm trúng vấn đề alias, còn q10 cần thêm một bước kiểm tra "câu hỏi này có trong scope của docs không" trước khi retrieve.

---

*Lưu file này với tên: `reports/individual/ta_thi_thuy_duong.md`*
