# Báo cáo thực hành — Production GraphRAG vs Flat RAG

**Học viên:** Hoàng Anh Minh - 2A202601192

**Khóa học:** AICB-K34 · Track 3: GraphRAG

**Ngày thực hiện:** 19/08/2026

**Notebook:** `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb`

## 1. Kết quả thực thi và phạm vi dữ liệu

Notebook đã chạy `Restart & Run All` đủ 23/23 code cell, không có error output. Pipeline xử lý 1.500 dòng nguồn, còn 980 bài/980 chunk sau exact dedup và SimHash near-dedup; 30 chunk đầu được coreference và NER/RE. Kết quả cuối gồm 32 node, 31 edge, 11 dòng entity-resolution audit và **0 edge thiếu provenance**.

Trạng thái dịch vụ ngoài được ghi trung thực trong `outputs/run_manifest.json`:

- Dataset HackerNoon chính đang gated và `HF_TOKEN` chưa được cấp quyền, nên runner dùng public derivative `MongoDB/tech-news-embeddings`, đồng thời đặt 14 bản tin đã kiểm chứng ở đầu corpus để benchmark tái lập.
- Neo4j Aura trong `.env` trả `AuthError`, nên pipeline dùng Neo4j 5.26.29 Community trên Docker local. Toàn bộ node/edge vẫn được nạp thật bằng Cypher `UNWIND` batch.
- Groq model cấu hình trả `model_not_found`; client tự chuyển một lần sang OpenAI `gpt-4o-mini`. Không secret nào xuất hiện trong notebook hoặc output.

## 2. Mười câu thuyết minh kỹ thuật

### Câu 1 — Coreference Resolution

Trong `gold09::c0000`, câu thứ hai bắt đầu bằng “The company” sau khi câu trước nhắc `Apple Inc`. Đây là trường hợp có antecedent rõ trong cùng chunk. Tuy nhiên, audit của 30 chunk cho thấy mô hình bảo thủ giữ nguyên toàn bộ văn bản, không tạo false resolution và cũng không ghi unresolved mention. Đây là một **safe abstention**: giảm false edge nhưng có thể làm mất edge `Apple -USES-> ...` nếu extractor phía sau không tự hiểu đại từ. Với câu có nhiều công ty, resolver tuyệt đối không được chọn công ty gần nhất chỉ theo vị trí; sai antecedent sẽ chuyển cả M&A/đầu tư sang sai node.

### Câu 2 — Entity threshold và Lexical Guard

Ngưỡng vector là cosine `0.90`; lexical guard tối thiểu `0.72`, kèm quy tắc token/type bảo thủ. Cặp thật trong audit là `Llama 2` và `Llama 3`: cosine `0.9224769`, lexical ratio `0.8571429`, nhưng quyết định là `REJECT_GUARD` vì hai số phiên bản biểu diễn hai technology khác nhau. Manual alias hợp nhất 10 biến thể như `MSFT`, `Microsoft Corporation`, `GOOG`, `GOOGL`, `Google LLC`, `AAPL` và `Meta Platforms Inc`.

### Câu 3 — Vì sao dùng ANN trước, lexical guard sau

ANN giảm candidate space, nhưng embedding tên ngắn dễ gom các phiên bản/sản phẩm gần nghĩa. Lexical guard kiểm tra hậu tố doanh nghiệp, tập token và quy tắc riêng cho Person. Ví dụ adversarial trong test: `Sam Altman` và `Steve Altman` có cùng họ nhưng khác tên nên bị chặn; `Apple` và `Apple Watch` cũng không được gộp. Union-Find chỉ nhận các pair đã qua cả vector threshold và guard, tránh lan truyền một false merge sang cả cluster.

### Câu 4 — Top degree và Super-node Mitigation

Top graph hiện tại là Apple (degree 6), Microsoft (6), rồi Anthropic/Google/Meta đồng hạng degree 5. Sample nhỏ chưa có node tự nhiên vượt 100. Unit test dựng node degree 125 và xác nhận chỉ lấy 50 edge; code đồng thời chặn toàn context ở 250 edge và 14.000 ký tự. Ưu tiên 50 edge mới nhất phù hợp câu hỏi hiện tại và kiểm soát token, nhưng có thể cắt mất acquisition/partnership lịch sử; với câu hỏi có mốc thời gian, production nên lọc date range trước khi áp cap.

### Câu 5 — Integrity của edge provenance

Mỗi edge có `edge_id`, `source_chunk_id`, `published_date`, `evidence`, `confidence` và `dataset_id`. Validation chạy trước ingestion, còn Cypher sanity check chạy sau ingestion. Kết quả là `invalid_provenance_edges = 0`. `MERGE` dùng `edge_id` sinh từ source/relation/target/chunk nên giữ được nhiều sự kiện theo thời gian mà vẫn idempotent.

### Câu 6 — Flat RAG thắng ở đâu

Flat RAG phù hợp factoid và câu mà top-k chứa trọn bằng chứng. G01 đạt 5/5 ở cả ba tiêu chí. Trung bình Flat dùng 638,0 token, ít hơn GraphRAG 892,2 token. Nó cũng không chịu chi phí NER/RE, canonicalization và graph maintenance.

### Câu 7 — GraphRAG thắng ở đâu

GraphRAG nổi bật ở multi-hop: ba score trung bình đều tăng từ 3,5 lên 5,0. Ở G03, Flat RAG thấy investment chunk nhưng đoán công nghệ là `ChatGPT`; GraphRAG nối `Microsoft -INVESTED_IN-> OpenAI` với `OpenAI -DEVELOPED-> GPT-4` và trả đúng reference.

### Câu 8 — Quality, latency và token trade-off

| Chỉ số overall | Flat RAG | GraphRAG | Graph − Flat |
|---|---:|---:|---:|
| Comprehensiveness | 4,4 | 5,0 | +0,6 |
| Faithfulness | 4,4 | 5,0 | +0,6 |
| Multi-hop reasoning | 4,4 | 5,0 | +0,6 |
| Latency sinh câu trả lời (s) | 2,089 | 1,727 | -0,362 |
| Token | 638,0 | 892,2 | +254,2 |

Latency chỉ đo generation, không tính offline extraction/indexing và bị ảnh hưởng network cache, nên không kết luận GraphRAG vốn nhanh hơn. Kết luận đáng tin hơn là GraphRAG cải thiện chất lượng multi-hop nhưng dùng thêm khoảng 40% token trong run này.

### Câu 9 — Failure mode và biện pháp sửa

Run đầu phát hiện LLM đảo `Hugging Face -FOUNDED/LEADS-> Clément Delangue` và tạo `Data I/O -LEADS-> market`. Root cause là allowlist chỉ kiểm relation name, chưa kiểm domain/range. Bản cuối thêm `RELATION_SIGNATURES`, tự đảo chiều hợp lệ về `Person -FOUNDED/LEADS-> Company`, loại type pair sai và bỏ confidence dưới 0,75. Số edge giảm từ 32 xuống tập sạch hơn, trong khi G03 vẫn giữ đủ chuỗi hai-hop.

### Câu 10 — Agent control và scale 350MB

Đề xuất bị từ chối là so sánh cosine mọi cặp entity (`O(N²)`) và insert Neo4j từng row. Thay vào đó hệ thống dùng FAISS candidate search + lexical guard + Union-Find, và `UNWIND` batch 1.000. Ở 350MB, bottleneck đầu tiên là số LLM extraction call/rate limit, sau đó là embedding và entity resolution. Kiến trúc scale: streaming → content-addressed checkpoint → async bounded workers → schema dead-letter queue → ANN blocking theo type → idempotent bulk ingestion → partition/community summaries. Không gửi lại chunk đã xử lý và không giữ toàn bộ raw corpus trong RAM.

## 3. Phân tích hai ca lỗi điển hình

### Ca A — Flat RAG thất bại, GraphRAG thành công

- **Query:** G03 — công nghệ do công ty nhận investment mở rộng của Microsoft phát triển.
- **Triệu chứng:** Flat trả `ChatGPT`; judge chấm 2/5 cho cả ba tiêu chí.
- **Root cause:** top-6 vector context ưu tiên investment chunk nhưng không ràng buộc đường quan hệ sang technology; generator bổ sung một tên phổ biến.
- **Graph fix:** traversal hai hop thu cạnh có provenance `gold05::c0000` và `gold06::c0000`, trả `GPT-4`; judge chấm 5/5.

### Ca B — Graph extraction tạo edge sai kiểu/hướng

- **Triệu chứng run đầu:** `Company -LEADS-> Technology(market)` và `Company -FOUNDED-> Person`.
- **Root cause:** strict JSON/allowlist chỉ bảo đảm cú pháp, không bảo đảm ontology semantics.
- **Fix:** relation domain/range signatures, conservative reorientation chỉ cho `FOUNDED`/`LEADS`, confidence floor và test regression.
- **Rủi ro còn lại:** edge như `Technology -USES-> Technology` vẫn cần ontology-specific review; confidence của LLM không phải xác suất calibrated.

## 4. Mapping bài giảng vào code

| Module | Khái niệm | Hàm/khối code | Kết quả |
|---|---|---|---|
| M1 | Stream, exact/near dedup, chunk, coref | `stream_hackernoon_dataset`, `standardize_news`, `near_deduplicate`, `build_chunks`, `run_coref` | 1.500 → 980 bài; SimHash audit 12 near-dup |
| M2 | Strict NER/RE + schema | `run_extraction`, `_validate_relation`, `RELATION_SIGNATURES` | 31 edge cuối, không extraction error |
| M3 | ANN + lexical guard + Union-Find | `build_resolution_map`, `merge_guard`, `UnionFind` | 11 audit: 10 manual merge, 1 reject |
| M4 | Flat FAISS + BFS hybrid | `FlatRAGIndex`, `HybridRetriever`, `answer_flat_rag`, `answer_graph_rag` | graph context có evidence/date/chunk |
| M5 | Golden + judge + export | `run_evaluation`, `judge_answer`, `comparison_table` | 5 câu, đủ 3 group, 2 CSV |
| Bonus | Near-dedup, community, self-correction | `near_deduplicate`, `build_communities`, `self_correcting_context` | community report và route hop2→hop3→vector |

## 5. Debugging và bài học

Lỗi khó nhất không nằm trong thuật toán mà ở reproducibility: ba external dependency đều có trạng thái không hợp lệ. Giải pháp là smoke test trước, fallback có audit thay vì nuốt lỗi, local Neo4j thật thay mock, và checkpoint tách riêng danh sách chunk đã xử lý. Checkpoint extraction ban đầu chỉ nhớ chunk có triple, khiến chunk “0 relation” bị gọi lại; bản cuối có `.processed.json`, nên run lặp lại phát sinh 0 coref/extraction/evaluation call.

### Kết quả bonus đã chạy

- SimHash near-dedup loại 12 bản gần trùng sau exact dedup.
- NetworkX tạo 10 community, ghi `community_id` lại Neo4j bằng `UNWIND` và sinh `community_reports.csv`.
- Global community search đã chạy, chỉ dùng citation `[community_id=...]`, không giả citation chunk.
- Self-correction audit trên G03 chọn route `hop2` với 2.085 ký tự context; code chỉ mở hop 3 rồi vector fallback khi sufficiency check trả thiếu.

## 6. Action plan đồ án

Đồ án đề xuất là trợ lý điều tra sự cố phần mềm. Flat RAG đủ cho câu hỏi một tài liệu; GraphRAG cần cho chuỗi `Customer → Ticket → Error → ProductVersion → Fix → Release`. Node dự kiến: `Customer`, `Ticket`, `Error`, `Product`, `Version`, `Fix`, `Release`; relation: `REPORTED`, `AFFECTS`, `CAUSED_BY`, `RESOLVED_BY`, `SHIPPED_IN`, `DUPLICATES`.

Entity resolution sẽ block theo tenant + node type, dùng ID sản phẩm/version làm guard cứng và vector chỉ tạo candidate. Các customer/product phổ biến là super-node; traversal sẽ lọc tenant, time window và relation allowlist trước, sau đó cap theo recency. Mọi edge phải giữ ticket/message id, timestamp và evidence span.

## 7. Tự đánh giá

| Tiêu chí | Điểm (1–5) | Ghi chú |
|---|---:|---|
| Hiểu GraphRAG | 5 | Triển khai và kiểm thử đủ pipeline |
| Kiểm soát AI Coding Agent | 5 | Từ chối O(N²), thêm semantic guard sau failure thật |
| Chất lượng knowledge graph | 4 | Provenance đầy đủ; sample extraction còn cần human review |
| Phân tích/debug | 5 | Có checkpoint, manifest, regression tests và failure RCA |
