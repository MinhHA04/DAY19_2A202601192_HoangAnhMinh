from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from graphrag_lab import (
    _validate_relation,
    LabConfig,
    build_resolution_map,
    chunk_text,
    comparison_table,
    merge_guard,
    near_deduplicate,
    run_extraction,
    standardize_news,
    test_supernode_policy as check_supernode_policy,
    validate_golden,
    validate_provenance,
)


class FakeEmbedder:
    def encode(self, names, **kwargs):
        # Deliberately identical candidates: the lexical guard remains decisive.
        return np.ones((len(names), 4), dtype="float32") / 2


def test_preprocessing_exact_dedup_and_chunk_overlap(tmp_path):
    config = LabConfig(
        root=tmp_path,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        report_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        chunk_words=5,
        chunk_overlap_words=2,
        lab_max_articles=10,
    )
    text = "one two three four five six seven eight nine"
    long_text = text + " additional words" * 10
    raw = pd.DataFrame(
        [
            {"article_id": "a", "title": "T", "published_date": "2023-01-01", "text": long_text},
            {"article_id": "b", "title": "T", "published_date": "2023-01-01", "text": long_text},
        ]
    )
    assert len(standardize_news(raw, config)) == 1
    assert chunk_text(text, 5, 2) == [
        "one two three four five",
        "four five six seven eight",
        "seven eight nine",
    ]


def test_near_dedup_uses_simhash_audit():
    text = "alpha beta gamma delta epsilon " * 20
    frame = pd.DataFrame({"text": [text, text, "completely different article " * 20]})
    kept, audit = near_deduplicate(frame)
    assert len(kept) == 2
    assert audit.iloc[0]["decision"] == "DROP_NEAR_DUP"


def test_lexical_guard_blocks_known_false_merges():
    assert not merge_guard("Sam Altman", "Steve Altman", "Person")
    assert not merge_guard("Apple", "Apple Watch", "Company")
    assert merge_guard("Microsoft Corp", "Microsoft Corporation", "Company")


def test_entity_resolution_records_reject_guard():
    raw = pd.DataFrame(
        [
            {
                "source_type": "Company",
                "source_raw": "Apple",
                "target_type": "Company",
                "target_raw": "Apple Watch",
            }
        ]
    )
    _, audit = build_resolution_map(raw, threshold=0.90, embedder=FakeEmbedder())
    assert "REJECT_GUARD" in set(audit["decision"])


def test_provenance_rejects_empty_values():
    valid = pd.DataFrame(
        [{"source_chunk_id": "c1", "published_date": "2023-01-01", "evidence": "x", "confidence": 0.9}]
    )
    validate_provenance(valid)
    invalid = valid.copy()
    invalid.loc[0, "published_date"] = ""
    with pytest.raises(ValueError, match="published_date"):
        validate_provenance(invalid)


def test_relation_signature_reorients_person_relations_and_rejects_bad_types():
    founded = _validate_relation(
        {
            "source": "Hugging Face",
            "source_type": "Company",
            "relation": "FOUNDED",
            "target": "Clément Delangue",
            "target_type": "Person",
            "evidence": "Clément Delangue was its co-founder",
            "confidence": 1.0,
        },
        "c1",
        "2023-01-01",
    )
    assert founded["source_raw"] == "Clément Delangue"
    assert founded["target_raw"] == "Hugging Face"
    assert _validate_relation(
        {
            "source": "Data I/O",
            "source_type": "Company",
            "relation": "LEADS",
            "target": "market",
            "target_type": "Technology",
            "evidence": "market leader",
            "confidence": 0.8,
        },
        "c2",
        "2023-01-01",
    ) is None


def test_extraction_checkpoint_remembers_zero_relation_chunks(tmp_path):
    class EmptyClient:
        def __init__(self):
            self.calls = 0

        def json(self, system, user, **kwargs):
            self.calls += 1
            return {"items": [{"chunk_id": "empty::c0000", "relations": []}]}, {}

    source = pd.DataFrame(
        [
            {
                "chunk_id": "empty::c0000",
                "published_date": "2023-01-01",
                "text": "No graph relation is stated in this sufficiently long test chunk.",
                "resolved_text": "",
            }
        ]
    )
    client = EmptyClient()
    checkpoint = tmp_path / "triples.csv"
    first, _, _ = run_extraction(source, client, checkpoint=checkpoint)
    second, _, _ = run_extraction(source, client, checkpoint=checkpoint)
    assert first.empty and second.empty
    assert client.calls == 1


def test_supernode_policy_caps_at_50():
    class Store:
        def graph_checks(self, dataset_id):
            return {}, pd.DataFrame([{"id": "hub"}])

        def node_degree(self, node_id, dataset_id):
            return 125

        def recent_edges(self, node_id, limit, dataset_id):
            return [{} for _ in range(min(limit, 125))]

    retriever = SimpleNamespace(
        store=Store(),
        config=SimpleNamespace(
            dataset_id="test", super_node_edge_cap=50, super_node_degree=100
        ),
    )
    result = check_supernode_policy(retriever)
    assert result["status"] == "PASS"
    assert result["fetched"] == 50


def test_golden_and_comparison_cover_required_groups():
    golden = pd.DataFrame(
        [
            {"id": "1", "group": "factoid", "question": "q", "reference_answer": "a"},
            {"id": "2", "group": "multi-hop", "question": "q", "reference_answer": "a"},
            {"id": "3", "group": "cross-doc", "question": "q", "reference_answer": "a"},
        ]
    )
    validate_golden(golden)
    evaluation = golden.assign(
        flat_comprehensiveness=3,
        graph_comprehensiveness=4,
        flat_faithfulness=4,
        graph_faithfulness=4,
        flat_multi_hop_reasoning=2,
        graph_multi_hop_reasoning=4,
        flat_latency_s=1.0,
        graph_latency_s=2.0,
        flat_total_tokens=100,
        graph_total_tokens=150,
    )
    summary = comparison_table(evaluation)
    assert "overall" in set(summary["Question group"])
    assert len(summary) == 20
