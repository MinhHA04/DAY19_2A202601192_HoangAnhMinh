# Failure analysis — Flat RAG và GraphRAG

## 1. Flat RAG hallucinate technology ở G03

Flat RAG trả `ChatGPT` thay vì `GPT-4`; judge chấm 2/5. Investment chunk được retrieve nhưng chunk công nghệ đúng không tạo thành ràng buộc quan hệ, nên generator dùng tên phổ biến. GraphRAG lấy hai edge có provenance `gold05::c0000` và `gold06::c0000`, trả đúng `GPT-4` và đạt 5/5.

## 2. Graph extraction đúng schema nhưng sai ontology

Run đầu có `Hugging Face -FOUNDED/LEADS-> Clément Delangue` và `Data I/O -LEADS-> market`. JSON, relation allowlist và confidence đều hợp lệ, nhưng direction/domain/range sai. Root cause là validation chỉ kiểm enum. Fix cuối gồm `RELATION_SIGNATURES`, reorientation bảo thủ cho `FOUNDED/LEADS`, confidence floor 0,75 và regression test. Sau sửa, cạnh trở thành `Clément Delangue -FOUNDED/LEADS-> Hugging Face`; pair Company→Technology cho LEADS bị loại.

## 3. Operational failures

- HF source chính gated: fallback sang public derivative và ghi lý do trong manifest.
- Aura credentials sai: fallback sang Neo4j 5.26.29 local, vẫn dùng Cypher thật.
- Groq model không còn quyền truy cập: circuit-break sau lỗi permanent và chuyển OpenAI.
- Checkpoint cũ không nhớ chunk có zero relation: thêm `raw_triples.processed.json`, run lặp lại còn 0 API call.

Các fallback đều fail-visible; không có lỗi nào bị biến thành kết quả “thành công” giả.
