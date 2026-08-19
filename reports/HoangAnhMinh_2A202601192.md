# Reflection & Action Plan — Hoàng Anh Minh (2A202601192)

## Mapping bài giảng

| Module | Implementation | Bài học |
|---|---|---|
| Preprocessing | `standardize_news`, `near_deduplicate`, `build_chunks`, `run_coref` | Dedup và checkpoint quyết định cả chi phí phía sau |
| Extraction | `run_extraction`, `_validate_relation` | JSON schema chưa đủ; ontology cần domain/range |
| Entity resolution | `build_resolution_map`, `merge_guard`, `UnionFind` | Vector chỉ nên sinh candidate, không tự quyết merge |
| Retrieval | `FlatRAGIndex`, `HybridRetriever` | Graph đem lại structure; vector giữ recall |
| Evaluation | `run_evaluation`, `judge_answer`, `comparison_table` | Phải đo quality, cost và latency cùng lúc |

## Debugging

Khó nhất là làm pipeline tái lập khi dataset, model và database bên ngoài cùng lỗi. Tôi học được rằng cần smoke test trước khi chạy batch lớn, phân biệt permanent/transient error, ghi fallback vào manifest và checkpoint cả trường hợp “đã xử lý nhưng kết quả rỗng”. Failure thật về direction của `FOUNDED/LEADS` cũng cho thấy output đúng JSON vẫn có thể sai ngữ nghĩa.

## Kế hoạch đồ án

Tôi sẽ áp dụng Hybrid GraphRAG cho trợ lý điều tra sự cố phần mềm. Flat RAG xử lý lookup một tài liệu; graph xử lý chuỗi Customer → Ticket → Error → ProductVersion → Fix → Release. Entity resolution block theo tenant/type, ưu tiên ID/version exact, vector chỉ fallback. Super-node được lọc tenant, time range và relation trước khi cap. Provenance bắt buộc gồm ticket/message id, timestamp, evidence và extraction version.
