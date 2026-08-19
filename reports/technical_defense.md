# Thuyết minh kỹ thuật — Hoàng Anh Minh

1. **Coreference:** `gold09::c0000` có “The company” sau `Apple Inc`. Resolver bảo thủ đã abstain thay vì thay sai; trade-off là có thể mất recall. False antecedent sẽ tạo edge M&A/USES cho sai company.
2. **Entity threshold:** cosine `0.90`, lexical `0.72` cộng type/token guard. `Llama 2` và `Llama 3` có cosine `0.9224769` nhưng bị `REJECT_GUARD` vì khác phiên bản technology.
3. **Entity resolution:** FAISS sinh candidate, lexical guard chặn pair nguy hiểm, Union-Find chỉ cluster pair đã duyệt. Test chặn `Sam Altman/Steve Altman` và `Apple/Apple Watch`.
4. **Super-node:** Apple và Microsoft degree 6; Anthropic, Google, Meta degree 5. Không có super-node tự nhiên trong sample; test degree 125 xác nhận cap 50, global cap 250, context cap 14.000 ký tự.
5. **Temporal policy:** newest-first ưu tiên thông tin hiện hành nhưng có thể loại sự kiện lịch sử. Query có date phải lọc time range trước cap.
6. **Flat RAG:** mạnh ở factoid; G01 đạt 5/5. Overall dùng 638,0 token so với 892,2 của GraphRAG.
7. **GraphRAG:** multi-hop tăng 3,5 lên 5,0. G03 nối Microsoft → OpenAI → GPT-4, trong khi Flat đoán ChatGPT.
8. **Latency/token:** generation latency lần chạy này 2,089s Flat và 1,727s Graph, nhưng đây không gồm graph build và chịu network variance. Graph dùng thêm 254,2 token/query.
9. **Agent control:** từ chối all-pairs cosine `O(N²)` và per-row Cypher; dùng ANN blocking, lexical guard, Union-Find và `UNWIND` 1.000 row/batch.
10. **Scale 350MB:** bottleneck đầu là LLM rate/cost. Dùng streaming, async bounded workers, content-addressed checkpoints, schema DLQ, ANN theo type, bulk ingestion và community partitioning.

Số liệu chi tiết và phân tích đầy đủ nằm trong `reports/lab_report.md` và `outputs/run_manifest.json`.
