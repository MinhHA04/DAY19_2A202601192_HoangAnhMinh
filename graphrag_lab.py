"""Production-oriented GraphRAG versus Flat RAG lab implementation.

The notebook imports this module so the same tested code is used in Colab, local
Jupyter, and the command-line runner. External services are isolated behind
small adapters and every material fallback is recorded in the run manifest.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import faiss
import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

ALLOWED_NODE_TYPES = {"Company", "Person", "Technology"}
ALLOWED_RELATIONS = {
    "ACQUIRED",
    "DEVELOPED",
    "INVESTED_IN",
    "FOUNDED",
    "WORKED_AT",
    "PARTNERED_WITH",
    "USES",
    "LEADS",
}
RELATION_SIGNATURES = {
    "ACQUIRED": {("Company", "Company")},
    "DEVELOPED": {("Company", "Technology"), ("Person", "Technology")},
    "INVESTED_IN": {("Company", "Company")},
    "FOUNDED": {("Person", "Company")},
    "WORKED_AT": {("Person", "Company")},
    "PARTNERED_WITH": {("Company", "Company")},
    "USES": {("Company", "Technology"), ("Technology", "Technology")},
    "LEADS": {("Person", "Company")},
}
MIN_EXTRACTION_CONFIDENCE = 0.75
CORP_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "plc",
}
MANUAL_ALIASES = {
    "aapl": "Apple",
    "apple inc": "Apple",
    "goog": "Google",
    "googl": "Google",
    "google llc": "Google",
    "meta platforms": "Meta",
    "meta platforms inc": "Meta",
    "microsoft corp": "Microsoft",
    "microsoft corporation": "Microsoft",
    "msft": "Microsoft",
}


def norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def sha1(value: Any) -> str:
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()


def norm_entity(name: Any) -> str:
    text = unicodedata.normalize("NFKC", norm_space(name)).lower()
    text = re.sub(r"[^\w\s\-.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_corporate_suffix(name: Any) -> str:
    tokens = norm_entity(name).replace(".", "").split()
    while tokens and tokens[-1] in CORP_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _int_env(name: str, default: int) -> int:
    value = norm_space(os.getenv(name))
    return int(value) if value else default


@dataclass(slots=True)
class LabConfig:
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    output_dir: Path = ROOT / "outputs"
    report_dir: Path = ROOT / "reports"
    cache_dir: Path = ROOT / "outputs" / "checkpoints"
    lab_max_articles: int = field(default_factory=lambda: _int_env("LAB_MAX_ARTICLES", 1500))
    lab_max_chunks: int = field(default_factory=lambda: _int_env("LAB_MAX_CHUNKS", 3000))
    extraction_max_chunks: int = field(
        default_factory=lambda: _int_env("EXTRACTION_MAX_CHUNKS", 400)
    )
    chunk_words: int = field(default_factory=lambda: _int_env("CHUNK_WORDS", 220))
    chunk_overlap_words: int = field(
        default_factory=lambda: _int_env("CHUNK_OVERLAP_WORDS", 40)
    )
    entity_threshold: float = 0.90
    lexical_threshold: float = 0.72
    seed_threshold: float = 0.66
    super_node_degree: int = 100
    super_node_edge_cap: int = 50
    global_edge_cap: int = 250
    max_graph_context_chars: int = 14_000
    dataset_id: str = "day19_hackernoon"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.chunk_overlap_words >= self.chunk_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_WORDS")
        for path in (self.data_dir, self.output_dir, self.report_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


class ChatClient:
    """JSON-capable chat client with Groq -> OpenAI fallback and usage data."""

    def __init__(self) -> None:
        self.groq_key = norm_space(os.getenv("GROQ_API_KEY"))
        self.groq_model = norm_space(os.getenv("GROQ_MODEL"))
        self.openai_key = norm_space(os.getenv("OPENAI_API_KEY"))
        self.openai_model = norm_space(os.getenv("JUDGE_MODEL")) or "gpt-4o-mini"
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def parse_json_object(text: Any) -> dict[str, Any]:
        cleaned = str(text).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        first, last = cleaned.find("{"), cleaned.rfind("}")
        if first < 0 or last <= first:
            raise ValueError("No JSON object found in model response")
        parsed = json.loads(cleaned[first : last + 1])
        if not isinstance(parsed, dict):
            raise TypeError("Model response must be a JSON object")
        return parsed

    @staticmethod
    def _usage(raw: Any) -> dict[str, int | None]:
        usage = getattr(raw, "usage", None)
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    def _call_groq(
        self, messages: list[dict[str, str]], json_mode: bool, max_tokens: int | None
    ) -> tuple[str, dict[str, Any]]:
        if not self.groq_key or not self.groq_model:
            raise RuntimeError("Groq is not configured")
        from groq import Groq

        kwargs: dict[str, Any] = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": 0.0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        response = Groq(api_key=self.groq_key).chat.completions.create(**kwargs)
        return response.choices[0].message.content or "", self._usage(response)

    def _call_openai(
        self, messages: list[dict[str, str]], json_mode: bool, max_tokens: int | None
    ) -> tuple[str, dict[str, Any]]:
        if not self.openai_key:
            raise RuntimeError("OpenAI is not configured")
        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "model": self.openai_model,
            "messages": messages,
            "temperature": 0.0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        response = OpenAI(api_key=self.openai_key).chat.completions.create(**kwargs)
        return response.choices[0].message.content or "", self._usage(response)

    def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        max_retries: int = 4,
    ) -> tuple[str, dict[str, Any]]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        providers = ["groq", "openai"] if self.groq_key else ["openai"]
        last_error: Exception | None = None
        for provider in providers:
            for attempt in range(max_retries):
                started = time.perf_counter()
                try:
                    if provider == "groq":
                        text, usage = self._call_groq(messages, json_mode, max_tokens)
                    else:
                        text, usage = self._call_openai(messages, json_mode, max_tokens)
                    usage["provider"] = provider
                    usage["model"] = (
                        self.groq_model if provider == "groq" else self.openai_model
                    )
                    usage["latency_s"] = time.perf_counter() - started
                    self.events.append({"provider": provider, "ok": True})
                    return text, usage
                except Exception as exc:  # network/provider failure; bounded retry
                    last_error = exc
                    self.events.append(
                        {
                            "provider": provider,
                            "ok": False,
                            "error_type": type(exc).__name__,
                        }
                    )
                    message = str(exc).lower()
                    permanent = any(
                        marker in message
                        for marker in ("model_not_found", "does not exist", "invalid api key")
                    )
                    if permanent and provider == "groq":
                        # Avoid repeating the same known-invalid model call for every batch.
                        self.groq_key = ""
                    if permanent or attempt == max_retries - 1:
                        break
                    time.sleep(min(8.0, 2**attempt + random.random()))
        raise RuntimeError(f"No chat provider succeeded: {type(last_error).__name__}: {last_error}")

    def json(
        self, system: str, user: str, *, max_tokens: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        text, usage = self.chat(
            system, user, json_mode=True, max_tokens=max_tokens
        )
        return self.parse_json_object(text), usage


def _first(row: Mapping[str, Any], candidates: Sequence[str], default: Any = "") -> Any:
    lowered = {str(key).lower(): key for key in row}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None and row.get(key) is not None:
            return row.get(key)
    return default


def normalize_source_row(row: Mapping[str, Any], source_dataset: str) -> dict[str, str]:
    title = norm_space(_first(row, ["title", "headline", "newsTitle"]))
    text = norm_space(
        _first(row, ["text", "content", "description", "article", "body", "story"])
    )
    published = pd.to_datetime(
        _first(row, ["published_date", "published_at", "date", "created_at"]),
        errors="coerce",
        utc=True,
    )
    published_date = "unknown" if pd.isna(published) else published.strftime("%Y-%m-%d")
    raw_id = _first(row, ["article_id", "id", "_id", "story_id", "uuid", "url"])
    article_id = norm_space(raw_id) or sha1(f"{title}\n{text}")[:20]
    return {
        "article_id": article_id,
        "title": title,
        "published_date": published_date,
        "text": text,
        "company": norm_space(_first(row, ["companyName", "company"])),
        "source_dataset": source_dataset,
    }


def stream_hackernoon_dataset(
    config: LabConfig,
    *,
    force: bool = False,
    prioritize_mb: bool = True,
    limit_mb: int = 300,
) -> tuple[Path, dict[str, Any]]:
    """Stream the official dataset, transparently falling back to its public derivative."""

    output = config.data_dir / "hackernoon_subset.csv"
    manifest: dict[str, Any] = {"path": str(output), "fallback_reason": ""}
    if output.exists() and output.stat().st_size > 100 and not force:
        cached = pd.read_csv(output)
        origins = sorted(
            set(cached.get("source_dataset", pd.Series(dtype=str)).dropna().astype(str))
        )
        origin = origins[0] if len(origins) == 1 else origins
        manifest.update(
            {
                "source": origin or "unknown",
                "cache_hit": True,
                "rows": len(cached),
                "size_mb": round(output.stat().st_size / (1024 * 1024), 3),
                "fallback_reason": (
                    "Official HackerNoon source was unavailable when this public derivative cache was materialized."
                    if "MongoDB/tech-news-embeddings" in origins
                    else ""
                ),
            }
        )
        return output, manifest

    from datasets import load_dataset

    candidates = [
        ("HackerNoon/tech-company-news-data-dump", norm_space(os.getenv("HF_TOKEN"))),
        ("MongoDB/tech-news-embeddings", None),
    ]
    stream: Iterable[Mapping[str, Any]] | None = None
    errors: list[str] = []
    chosen = ""
    for dataset_name, token in candidates:
        try:
            stream = load_dataset(
                dataset_name, split="train", streaming=True, token=token or None
            )
            iterator = iter(stream)
            first_row = next(iterator)
            stream = _prepend(first_row, iterator)
            chosen = dataset_name
            break
        except Exception as exc:
            errors.append(f"{dataset_name}: {type(exc).__name__}")
    if stream is None:
        raise RuntimeError("No HackerNoon source available: " + "; ".join(errors))

    fields = [
        "article_id",
        "title",
        "published_date",
        "text",
        "company",
        "source_dataset",
    ]
    rows_written = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw in tqdm(stream, total=config.lab_max_articles, desc="Streaming news"):
            normalized = normalize_source_row(raw, chosen)
            if len(normalized["text"]) < 80:
                continue
            writer.writerow(normalized)
            rows_written += 1
            if rows_written % 50 == 0:
                handle.flush()
            size_mb = output.stat().st_size / (1024 * 1024)
            if rows_written >= config.lab_max_articles:
                break
            if prioritize_mb and size_mb >= limit_mb:
                break
    if rows_written == 0:
        raise RuntimeError("Dataset stream produced zero usable rows")
    manifest.update(
        {
            "source": chosen,
            "rows": rows_written,
            "size_mb": round(output.stat().st_size / (1024 * 1024), 3),
            "fallback_reason": "; ".join(errors),
        }
    )
    return output, manifest


def _prepend(first: Any, iterator: Iterator[Any]) -> Iterator[Any]:
    yield first
    yield from iterator


def load_news(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(source, lines=True)
    if suffix == ".json":
        return pd.read_json(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported dataset format: {source.suffix}")


def standardize_news(raw: pd.DataFrame, config: LabConfig) -> pd.DataFrame:
    rows = [normalize_source_row(row, norm_space(row.get("source_dataset")) or "local")
            for row in raw.to_dict("records")]
    frame = pd.DataFrame(rows)
    frame = frame[frame["text"].str.len() >= 80].copy()
    frame["dedup_key"] = [
        sha1(norm_space(f"{title}\n{text}").lower())
        for title, text in zip(frame["title"], frame["text"])
    ]
    frame = (
        frame.drop_duplicates("dedup_key")
        .drop(columns="dedup_key")
        .reset_index(drop=True)
    )
    if config.lab_max_articles and len(frame) > config.lab_max_articles:
        frame = frame.head(config.lab_max_articles).copy()
    return frame


def _simhash64(text: str) -> int:
    tokens = re.findall(r"\w+", norm_space(text).lower())
    if len(tokens) < 3:
        tokens = tokens or [""]
        shingles = tokens
    else:
        shingles = [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    vector = [0] * 64
    for shingle in shingles:
        value = int(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def near_deduplicate(
    frame: pd.DataFrame, max_hamming: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Near-deduplicate with 4-band SimHash candidate blocking (bonus challenge)."""

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    hashes: list[int] = []
    keep: list[bool] = []
    audit: list[dict[str, Any]] = []
    for idx, text in enumerate(frame["text"].fillna("")):
        fingerprint = _simhash64(text)
        candidates: set[int] = set()
        for band in range(4):
            candidates.update(buckets[(band, (fingerprint >> (band * 16)) & 0xFFFF)])
        duplicate_of: int | None = None
        distance: int | None = None
        for candidate in sorted(candidates):
            current = (fingerprint ^ hashes[candidate]).bit_count()
            if current <= max_hamming:
                duplicate_of, distance = candidate, current
                break
        keep.append(duplicate_of is None)
        hashes.append(fingerprint)
        if duplicate_of is not None:
            audit.append(
                {
                    "row_index": idx,
                    "duplicate_of_index": duplicate_of,
                    "hamming_distance": distance,
                    "decision": "DROP_NEAR_DUP",
                }
            )
            continue
        for band in range(4):
            buckets[(band, (fingerprint >> (band * 16)) & 0xFFFF)].append(idx)
    return frame.loc[keep].reset_index(drop=True), pd.DataFrame(audit)


def chunk_text(text: Any, size: int = 220, overlap: int = 40) -> list[str]:
    words = norm_space(text).split()
    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        part = words[start : start + size]
        if not part:
            break
        chunks.append(" ".join(part))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(news_df: pd.DataFrame, config: LabConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for article in news_df.itertuples(index=False):
        for index, text in enumerate(
            chunk_text(article.text, config.chunk_words, config.chunk_overlap_words)
        ):
            rows.append(
                {
                    "chunk_id": f"{article.article_id}::c{index:04d}",
                    "article_id": article.article_id,
                    "title": article.title,
                    "published_date": article.published_date or "unknown",
                    "text": text,
                    "source_dataset": article.source_dataset,
                }
            )
            if len(rows) >= config.lab_max_chunks:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


COREF_SYSTEM = """
You are a conservative coreference component for a knowledge-graph pipeline.
Resolve a pronoun or generic mention only if its antecedent is explicit in the
same chunk. Never infer a missing antecedent. Preserve dates, amounts, tickers,
product names, and the original meaning. Return strict JSON only.
""".strip()


def resolve_coref_batch(
    batch_df: pd.DataFrame, client: ChatClient
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = [
        {"chunk_id": row.chunk_id, "text": row.text}
        for row in batch_df.itertuples(index=False)
    ]
    prompt = f"""
Return {{"items":[{{"chunk_id":"...","resolved_text":"...",
"unresolved_mentions":["..."]}}]}} for this input:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
    obj, usage = client.json(COREF_SYSTEM, prompt, max_tokens=3000)
    by_id = {item.get("chunk_id"): item for item in obj.get("items", [])}
    rows = []
    for row in batch_df.itertuples(index=False):
        item = by_id.get(row.chunk_id, {})
        unresolved = item.get("unresolved_mentions", [])
        if not isinstance(unresolved, list):
            unresolved = [norm_space(unresolved)] if norm_space(unresolved) else []
        rows.append(
            {
                "chunk_id": row.chunk_id,
                "resolved_text": norm_space(item.get("resolved_text")) or row.text,
                "unresolved_mentions": unresolved,
                "coref_status": "OK" if item else "MISSING_RESPONSE_ITEM",
            }
        )
    return pd.DataFrame(rows), usage


def run_coref(
    chunks: pd.DataFrame,
    client: ChatClient,
    *,
    batch_size: int = 5,
    checkpoint: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    existing = pd.DataFrame()
    if checkpoint and checkpoint.exists():
        existing = pd.read_csv(checkpoint)
        if "unresolved_mentions" in existing:
            existing["unresolved_mentions"] = existing["unresolved_mentions"].map(
                lambda value: ast.literal_eval(value)
                if isinstance(value, str) and value.startswith("[")
                else []
            )
    done = set(existing.get("chunk_id", []))
    output = existing.to_dict("records")
    usage_rows: list[dict[str, Any]] = []
    pending = chunks[~chunks["chunk_id"].isin(done)]
    for start in tqdm(range(0, len(pending), batch_size), desc="Coreference"):
        batch = pending.iloc[start : start + batch_size]
        try:
            resolved, usage = resolve_coref_batch(batch, client)
            usage_rows.append(usage)
        except Exception as exc:
            resolved = pd.DataFrame(
                {
                    "chunk_id": batch["chunk_id"],
                    "resolved_text": batch["text"],
                    "unresolved_mentions": [["COREF_BATCH_FAILED"]] * len(batch),
                    "coref_status": [f"FAILED:{type(exc).__name__}"] * len(batch),
                }
            )
        output.extend(resolved.to_dict("records"))
        if checkpoint:
            pd.DataFrame(output).to_csv(checkpoint, index=False)
    return pd.DataFrame(output), usage_rows


EXTRACT_SYSTEM = f"""
Extract a high-precision knowledge graph from technology-news text.
Allowed node types: {sorted(ALLOWED_NODE_TYPES)}.
Allowed relations: {sorted(ALLOWED_RELATIONS)}.
Use entity names exactly as written in the evidence. Extract only explicit
facts. Each relation must include a short verbatim evidence span and confidence
from 0 to 1. Return strict JSON only.
""".strip()


def extract_batch(
    batch_df: pd.DataFrame, client: ChatClient
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = [
        {
            "chunk_id": row.chunk_id,
            "published_date": row.published_date,
            "text": norm_space(getattr(row, "resolved_text", "")) or row.text,
        }
        for row in batch_df.itertuples(index=False)
    ]
    prompt = f"""
Return this schema:
{{"items":[{{"chunk_id":"...","relations":[{{"source":"...",
"source_type":"Company|Person|Technology","relation":"ALLOWED_RELATION",
"target":"...","target_type":"Company|Person|Technology",
"evidence":"...","confidence":0.0}}]}}]}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
    return client.json(EXTRACT_SYSTEM, prompt, max_tokens=4000)


TRIPLE_COLUMNS = [
    "source_raw",
    "source_type",
    "relation",
    "target_raw",
    "target_type",
    "source_chunk_id",
    "published_date",
    "evidence",
    "confidence",
]


def _validate_relation(item: Mapping[str, Any], chunk_id: str, published_date: str) -> dict[str, Any] | None:
    source = norm_space(item.get("source"))
    target = norm_space(item.get("target"))
    source_type = norm_space(item.get("source_type"))
    target_type = norm_space(item.get("target_type"))
    relation = norm_space(item.get("relation")).upper()
    evidence = norm_space(item.get("evidence"))
    if not source or not target or not evidence:
        return None
    if source_type not in ALLOWED_NODE_TYPES or target_type not in ALLOWED_NODE_TYPES:
        return None
    if relation not in ALLOWED_RELATIONS:
        return None
    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < MIN_EXTRACTION_CONFIDENCE:
        return None
    signature = (source_type, target_type)
    if signature not in RELATION_SIGNATURES[relation]:
        reverse_signature = (target_type, source_type)
        if relation in {"FOUNDED", "LEADS"} and reverse_signature in RELATION_SIGNATURES[relation]:
            source, target = target, source
            source_type, target_type = target_type, source_type
        else:
            return None
    return {
        "source_raw": source,
        "source_type": source_type,
        "relation": relation,
        "target_raw": target,
        "target_type": target_type,
        "source_chunk_id": chunk_id,
        "published_date": norm_space(published_date) or "unknown",
        "evidence": evidence,
        "confidence": min(1.0, max(0.0, confidence)),
    }


def run_extraction(
    source: pd.DataFrame,
    client: ChatClient,
    *,
    batch_size: int = 4,
    checkpoint: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    triples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    done: set[str] = set()
    processed_checkpoint = checkpoint.with_suffix(".processed.json") if checkpoint else None
    if checkpoint and checkpoint.exists():
        cached = pd.read_csv(checkpoint)
        if set(TRIPLE_COLUMNS).issubset(cached.columns):
            triples = cached[TRIPLE_COLUMNS].to_dict("records")
            done = set(cached["source_chunk_id"])
    if processed_checkpoint and processed_checkpoint.exists():
        processed = json.loads(processed_checkpoint.read_text(encoding="utf-8"))
        done.update(str(chunk_id) for chunk_id in processed)
    pending = source[~source["chunk_id"].isin(done)]
    dates = source.set_index("chunk_id")["published_date"].to_dict()
    for start in tqdm(range(0, len(pending), batch_size), desc="NER + RE"):
        batch = pending.iloc[start : start + batch_size]
        try:
            response, usage = extract_batch(batch, client)
            usage_rows.append(usage)
            for group in response.get("items", []):
                chunk_id = norm_space(group.get("chunk_id"))
                if chunk_id not in dates:
                    continue
                for relation in group.get("relations", []):
                    valid = _validate_relation(relation, chunk_id, dates[chunk_id])
                    if valid:
                        triples.append(valid)
        except Exception as exc:
            errors.append(
                {
                    "start": start,
                    "chunk_ids": "|".join(batch["chunk_id"].astype(str)),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
        done.update(batch["chunk_id"].astype(str))
        if checkpoint:
            pd.DataFrame(triples, columns=TRIPLE_COLUMNS).to_csv(checkpoint, index=False)
        if processed_checkpoint:
            processed_checkpoint.write_text(
                json.dumps(sorted(done), ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return (
        pd.DataFrame(triples, columns=TRIPLE_COLUMNS),
        pd.DataFrame(errors),
        usage_rows,
    )


def merge_guard(a: str, b: str, entity_type: str = "Company", threshold: float = 0.72) -> bool:
    """Conservative lexical guard applied after vector candidate generation."""

    left, right = strip_corporate_suffix(a), strip_corporate_suffix(b)
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens, right_tokens = left.split(), right.split()
    if entity_type == "Person":
        # Same surname is insufficient: Sam Altman and Steve Altman stay separate.
        return (
            len(left_tokens) >= 2
            and len(right_tokens) >= 2
            and left_tokens[-1] == right_tokens[-1]
            and left_tokens[0] == right_tokens[0]
            and SequenceMatcher(None, left, right).ratio() >= 0.90
        )
    left_set, right_set = set(left_tokens), set(right_tokens)
    if left_set < right_set or right_set < left_set:
        # Prevent Company/Product collapses such as Apple vs Apple Music.
        return False
    ratio = SequenceMatcher(None, left, right).ratio()
    token_overlap = len(left_set & right_set) / max(len(left_set), len(right_set))
    return ratio >= threshold and token_overlap >= 0.8


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


_EMBEDDER: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def build_resolution_map(
    raw_triples: pd.DataFrame,
    *,
    threshold: float = 0.90,
    top_k: int = 5,
    embedder: Any | None = None,
) -> tuple[dict[tuple[str, str], str], pd.DataFrame]:
    mentions: list[tuple[str, str]] = []
    for row in raw_triples.itertuples(index=False):
        mentions.extend(
            [(row.source_type, row.source_raw), (row.target_type, row.target_raw)]
        )
    counts = Counter((typ, norm_entity(name)) for typ, name in mentions)
    display_name: dict[tuple[str, str], str] = {}
    for typ, name in mentions:
        display_name.setdefault((typ, norm_entity(name)), name)

    mapping: dict[tuple[str, str], str] = {}
    audit: list[dict[str, Any]] = []
    for key in counts:
        typ, normalized = key
        canonical = MANUAL_ALIASES.get(normalized)
        if canonical:
            mapping[key] = canonical
            audit.append(
                {
                    "type": typ,
                    "left": display_name[key],
                    "right": canonical,
                    "similarity": 1.0,
                    "lexical_ratio": 1.0,
                    "decision": "MERGE_MANUAL",
                    "reason": "curated alias map",
                }
            )

    model = embedder or get_embedder()
    for entity_type in sorted(ALLOWED_NODE_TYPES):
        keys = [key for key in counts if key[0] == entity_type and key not in mapping]
        if not keys:
            continue
        names = [display_name[key] for key in keys]
        vectors = model.encode(
            names,
            batch_size=128,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        similarities, neighbors = index.search(vectors, min(top_k, len(names)))
        union_find = UnionFind(len(names))
        for left_idx in range(len(names)):
            for score, right_idx in zip(similarities[left_idx], neighbors[left_idx]):
                if right_idx < 0 or left_idx >= right_idx or float(score) < threshold:
                    continue
                accepted = merge_guard(names[left_idx], names[right_idx], entity_type)
                lexical = SequenceMatcher(
                    None,
                    strip_corporate_suffix(names[left_idx]),
                    strip_corporate_suffix(names[right_idx]),
                ).ratio()
                audit.append(
                    {
                        "type": entity_type,
                        "left": names[left_idx],
                        "right": names[right_idx],
                        "similarity": float(score),
                        "lexical_ratio": lexical,
                        "decision": "MERGE_VECTOR" if accepted else "REJECT_GUARD",
                        "reason": "vector + lexical guard",
                    }
                )
                if accepted:
                    union_find.union(left_idx, int(right_idx))
        groups: dict[int, list[int]] = defaultdict(list)
        for index_value in range(len(names)):
            groups[union_find.find(index_value)].append(index_value)
        for members in groups.values():
            best = sorted(
                members,
                key=lambda idx: (-counts[keys[idx]], len(names[idx]), names[idx].lower()),
            )[0]
            canonical = names[best]
            for member in members:
                mapping[keys[member]] = canonical
    for key in counts:
        mapping.setdefault(key, display_name[key])
    columns = [
        "type",
        "left",
        "right",
        "similarity",
        "lexical_ratio",
        "decision",
        "reason",
    ]
    return mapping, pd.DataFrame(audit, columns=columns)


def canonicalize_triples(
    raw: pd.DataFrame, mapping: Mapping[tuple[str, str], str]
) -> pd.DataFrame:
    frame = raw.copy()

    def canonical(name: str, entity_type: str) -> str:
        normalized = norm_entity(name)
        return mapping.get(
            (entity_type, normalized), MANUAL_ALIASES.get(normalized, name)
        )

    frame["source_name"] = [
        canonical(name, typ) for name, typ in zip(frame.source_raw, frame.source_type)
    ]
    frame["target_name"] = [
        canonical(name, typ) for name, typ in zip(frame.target_raw, frame.target_type)
    ]
    frame["source_name_norm"] = frame.source_name.map(norm_entity)
    frame["target_name_norm"] = frame.target_name.map(norm_entity)
    frame["source_id"] = [
        sha1(f"{typ}:{name}")[:24]
        for typ, name in zip(frame.source_type, frame.source_name_norm)
    ]
    frame["target_id"] = [
        sha1(f"{typ}:{name}")[:24]
        for typ, name in zip(frame.target_type, frame.target_name_norm)
    ]
    frame["edge_id"] = [
        sha1(f"{source}|{rel}|{target}|{chunk}")[:32]
        for source, rel, target, chunk in zip(
            frame.source_id,
            frame.relation,
            frame.target_id,
            frame.source_chunk_id,
        )
    ]
    return frame[frame.source_id != frame.target_id].reset_index(drop=True)


def validate_provenance(triples: pd.DataFrame) -> None:
    required = {"source_chunk_id", "published_date", "evidence", "confidence"}
    missing = required - set(triples.columns)
    if missing:
        raise ValueError(f"Missing provenance columns: {sorted(missing)}")
    for column in ("source_chunk_id", "published_date", "evidence"):
        invalid = triples[column].fillna("").astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(f"{int(invalid.sum())} triples have empty {column}")


def build_nodes(triples: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in triples.itertuples(index=False):
        rows.extend(
            [
                {
                    "id": row.source_id,
                    "name": row.source_name,
                    "name_norm": row.source_name_norm,
                    "type": row.source_type,
                    "alias": row.source_raw,
                },
                {
                    "id": row.target_id,
                    "name": row.target_name,
                    "name_norm": row.target_name_norm,
                    "type": row.target_type,
                    "alias": row.target_raw,
                },
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=["id", "name", "name_norm", "type", "aliases", "aliases_norm", "dataset_id"]
        )
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    for (node_id, name, name_norm, typ), group in frame.groupby(
        ["id", "name", "name_norm", "type"], sort=True
    ):
        aliases = sorted(set(group["alias"].map(norm_space)))
        output.append(
            {
                "id": node_id,
                "name": name,
                "name_norm": name_norm,
                "type": typ,
                "aliases": aliases,
                "aliases_norm": sorted({norm_entity(alias) for alias in aliases}),
                "dataset_id": dataset_id,
            }
        )
    return pd.DataFrame(output)


def batches(records: Sequence[dict[str, Any]], size: int = 1000) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(records), size):
        yield list(records[start : start + size])


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    @classmethod
    def from_environment(cls, allow_local_fallback: bool = True) -> tuple["Neo4jGraphStore", dict[str, Any]]:
        password = norm_space(os.getenv("NEO4J_PASSWORD"))
        user = norm_space(os.getenv("NEO4J_USER")) or "neo4j"
        database = norm_space(os.getenv("NEO4J_DATABASE")) or "neo4j"
        candidates = [(norm_space(os.getenv("NEO4J_URI")), "configured")]
        if allow_local_fallback:
            candidates.append(("bolt://localhost:7687", "local_docker"))
        errors = []
        for uri, label in candidates:
            if not uri or not password:
                continue
            try:
                return cls(uri, user, password, database), {
                    "backend": "neo4j",
                    "connection": label,
                    "uri": "bolt://localhost:7687" if label == "local_docker" else "configured",
                    "fallback_reason": "; ".join(errors),
                }
            except Exception as exc:
                errors.append(f"{label}:{type(exc).__name__}")
        raise RuntimeError("Neo4j connection failed: " + "; ".join(errors))

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            rows = [record.data() for record in result]
            result.consume()
        return rows

    def setup_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX entity_name_norm IF NOT EXISTS FOR (n:Entity) ON (n.name_norm)",
            "CREATE INDEX entity_dataset IF NOT EXISTS FOR (n:Entity) ON (n.dataset_id)",
        ]
        for statement in statements:
            self.run(statement)

    def clear_dataset(self, dataset_id: str) -> None:
        """Remove only the prior graph owned by this lab's explicit dataset id."""

        self.run(
            "MATCH (n:Entity {dataset_id:$dataset_id}) DETACH DELETE n",
            dataset_id=dataset_id,
        )

    def bulk_insert_nodes(self, nodes: pd.DataFrame, batch_size: int = 1000) -> None:
        for entity_type in sorted(ALLOWED_NODE_TYPES):
            part = nodes[nodes.type == entity_type]
            if part.empty:
                continue
            query = f"""
            UNWIND $rows AS row
            MERGE (n:Entity {{id: row.id}})
            SET n:{entity_type}, n.name=row.name, n.name_norm=row.name_norm,
                n.entity_type=row.type, n.aliases=row.aliases,
                n.aliases_norm=row.aliases_norm, n.dataset_id=row.dataset_id
            """
            for batch in batches(part.to_dict("records"), batch_size):
                self.run(query, rows=batch)

    def bulk_insert_edges(
        self, triples: pd.DataFrame, dataset_id: str, batch_size: int = 1000
    ) -> None:
        validate_provenance(triples)
        for relation in sorted(ALLOWED_RELATIONS):
            part = triples[triples.relation == relation].copy()
            if part.empty:
                continue
            part["dataset_id"] = dataset_id
            query = f"""
            UNWIND $rows AS row
            MATCH (s:Entity {{id: row.source_id}})
            MATCH (t:Entity {{id: row.target_id}})
            MERGE (s)-[r:{relation} {{edge_id: row.edge_id}}]->(t)
            SET r.source_chunk_id=row.source_chunk_id,
                r.published_date=row.published_date,
                r.evidence=row.evidence, r.confidence=row.confidence,
                r.dataset_id=row.dataset_id
            """
            columns = [
                "source_id",
                "target_id",
                "edge_id",
                "source_chunk_id",
                "published_date",
                "evidence",
                "confidence",
                "dataset_id",
            ]
            for batch in batches(part[columns].to_dict("records"), batch_size):
                self.run(query, rows=batch)

    def graph_checks(self, dataset_id: str) -> tuple[dict[str, int], pd.DataFrame]:
        row = self.run(
            """
            MATCH (a:Entity {dataset_id:$dataset_id})-[r {dataset_id:$dataset_id}]->(b:Entity)
            RETURN count(DISTINCT a) + count(DISTINCT b) AS endpoint_count,
                   count(r) AS edges,
                   sum(CASE WHEN r.source_chunk_id IS NULL OR trim(r.source_chunk_id) = ''
                              OR r.published_date IS NULL OR trim(r.published_date) = ''
                            THEN 1 ELSE 0 END) AS invalid
            """,
            dataset_id=dataset_id,
        )[0]
        node_count = self.run(
            "MATCH (n:Entity {dataset_id:$dataset_id}) RETURN count(n) AS n",
            dataset_id=dataset_id,
        )[0]["n"]
        counts = {
            "nodes": int(node_count),
            "edges": int(row["edges"]),
            "invalid_provenance_edges": int(row["invalid"] or 0),
        }
        if counts["invalid_provenance_edges"] != 0:
            raise AssertionError("Neo4j contains edges with invalid provenance")
        top = pd.DataFrame(
            self.run(
                """
                MATCH (n:Entity {dataset_id:$dataset_id})
                OPTIONAL MATCH (n)-[r {dataset_id:$dataset_id}]-()
                WITH n, count(r) AS degree
                RETURN n.id AS id, n.name AS name, n.entity_type AS type, degree
                ORDER BY degree DESC, name LIMIT 15
                """,
                dataset_id=dataset_id,
            )
        )
        return counts, top

    def match_exact(self, name: str, entity_type: str | None, dataset_id: str) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (n:Entity {dataset_id:$dataset_id})
            WHERE (n.name_norm=$name OR $name IN coalesce(n.aliases_norm, []))
              AND ($typ IS NULL OR n.entity_type=$typ)
            RETURN n.id AS id, n.name AS name, n.entity_type AS type LIMIT 5
            """,
            dataset_id=dataset_id,
            name=norm_entity(name),
            typ=entity_type,
        )

    def node_degree(self, node_id: str, dataset_id: str) -> int:
        return int(
            self.run(
                """
                MATCH (n:Entity {id:$id, dataset_id:$dataset_id})
                OPTIONAL MATCH (n)-[r {dataset_id:$dataset_id}]-()
                RETURN count(r) AS degree
                """,
                id=node_id,
                dataset_id=dataset_id,
            )[0]["degree"]
        )

    def recent_edges(self, node_id: str, limit: int, dataset_id: str) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (n:Entity {id:$id, dataset_id:$dataset_id})
            MATCH (n)-[r {dataset_id:$dataset_id}]-(m:Entity)
            RETURN startNode(r).id AS source_id, startNode(r).name AS source_name,
                   startNode(r).entity_type AS source_type, type(r) AS relation,
                   endNode(r).id AS target_id, endNode(r).name AS target_name,
                   endNode(r).entity_type AS target_type,
                   r.source_chunk_id AS source_chunk_id,
                   r.published_date AS published_date, r.evidence AS evidence,
                   r.confidence AS confidence, m.id AS neighbor_id
            ORDER BY coalesce(r.published_date, '') DESC, r.edge_id
            LIMIT $limit
            """,
            id=node_id,
            dataset_id=dataset_id,
            limit=int(limit),
        )

    def all_edges(self, dataset_id: str, limit: int = 20_000) -> pd.DataFrame:
        return pd.DataFrame(
            self.run(
                """
                MATCH (a:Entity {dataset_id:$dataset_id})-[r {dataset_id:$dataset_id}]->(b:Entity)
                RETURN a.id AS source, b.id AS target, a.name AS source_name,
                       b.name AS target_name, type(r) AS relation
                LIMIT $limit
                """,
                dataset_id=dataset_id,
                limit=int(limit),
            )
        )

    def write_communities(self, rows: list[dict[str, Any]], dataset_id: str) -> None:
        for batch in batches(rows, 1000):
            self.run(
                """
                UNWIND $rows AS row
                MATCH (n:Entity {id:row.id, dataset_id:$dataset_id})
                SET n.community_id=row.community_id
                """,
                rows=batch,
                dataset_id=dataset_id,
            )


def bulk_insert_nodes(store: Neo4jGraphStore, nodes_df: pd.DataFrame, batch_size: int = 1000) -> None:
    store.bulk_insert_nodes(nodes_df, batch_size)


def bulk_insert_edges(
    store: Neo4jGraphStore, triples_df: pd.DataFrame, dataset_id: str, batch_size: int = 1000
) -> None:
    store.bulk_insert_edges(triples_df, dataset_id, batch_size)


class FlatRAGIndex:
    def __init__(self, chunks: pd.DataFrame, embedder: Any | None = None):
        if chunks.empty:
            raise ValueError("Cannot build a Flat RAG index from zero chunks")
        self.embedder = embedder or get_embedder()
        self.store = chunks.reset_index(drop=True).copy()
        vectors = self.embedder.encode(
            self.store.text.fillna("").tolist(),
            batch_size=128,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def retrieve(self, query: str, k: int = 6) -> tuple[str, pd.DataFrame]:
        vector = self.embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        scores, indexes = self.index.search(vector, min(k, self.index.ntotal))
        rows = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue
            item = self.store.iloc[int(index)]
            rows.append(
                {
                    "score": float(score),
                    "chunk_id": item.chunk_id,
                    "published_date": item.published_date,
                    "text": item.text,
                }
            )
        frame = pd.DataFrame(rows)
        context = "\n\n".join(
            f"[chunk_id={row.chunk_id} | date={row.published_date} | score={row.score:.3f}]\n{row.text}"
            for row in frame.itertuples(index=False)
        )
        return context, frame


SEED_SYSTEM = """
Extract only named entities useful as graph-retrieval seeds. Allowed types are
Company, Person, and Technology. Do not answer the question. Return strict JSON.
""".strip()


class HybridRetriever:
    def __init__(
        self,
        store: Neo4jGraphStore,
        nodes: pd.DataFrame,
        flat_index: FlatRAGIndex,
        client: ChatClient,
        config: LabConfig,
    ):
        self.store = store
        self.nodes = nodes.reset_index(drop=True).copy()
        self.flat_index = flat_index
        self.client = client
        self.config = config
        self.embedder = flat_index.embedder
        self.entity_vectors = self.embedder.encode(
            self.nodes.name.tolist(),
            batch_size=128,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")

    def extract_seeds(self, question: str) -> list[dict[str, str | None]]:
        try:
            obj, _ = self.client.json(
                SEED_SYSTEM,
                f'Question: {question}\nReturn {{"seeds":[{{"name":"...","type":"Company|Person|Technology|null"}}]}}',
                max_tokens=500,
            )
            seeds = [
                {
                    "name": norm_space(item.get("name")),
                    "type": item.get("type")
                    if item.get("type") in ALLOWED_NODE_TYPES
                    else None,
                }
                for item in obj.get("seeds", [])
                if norm_space(item.get("name"))
            ]
            if seeds:
                return seeds
        except Exception:
            pass
        # Deterministic fallback: known entity names occurring in the question.
        lowered = norm_entity(question)
        return [
            {"name": row.name, "type": row.type}
            for row in self.nodes.itertuples(index=False)
            if norm_entity(row.name) in lowered
        ]

    def match_seeds(self, question: str) -> list[dict[str, Any]]:
        matched: list[dict[str, Any]] = []
        for seed in self.extract_seeds(question):
            exact = self.store.match_exact(
                str(seed["name"]), seed["type"], self.config.dataset_id
            )
            if exact:
                matched.extend(exact)
                continue
            mask = np.ones(len(self.nodes), dtype=bool)
            if seed["type"]:
                mask = self.nodes.type.eq(seed["type"]).to_numpy()
            indexes = np.flatnonzero(mask)
            if not len(indexes):
                continue
            query_vector = self.embedder.encode(
                [str(seed["name"])], normalize_embeddings=True, show_progress_bar=False
            ).astype("float32")[0]
            similarities = self.entity_vectors[indexes] @ query_vector
            best = int(np.argmax(similarities))
            if float(similarities[best]) >= self.config.seed_threshold:
                row = self.nodes.iloc[int(indexes[best])]
                matched.append({"id": row.id, "name": row.name, "type": row.type})
        return list({item["id"]: item for item in matched}.values())

    def textualize(self, edges: Sequence[Mapping[str, Any]]) -> str:
        ordered = sorted(
            edges, key=lambda edge: norm_space(edge.get("published_date")), reverse=True
        )
        lines, used = [], 0
        for edge in ordered:
            line = (
                f"{edge['source_name']} [{edge['source_type']}] -{edge['relation']}-> "
                f"{edge['target_name']} [{edge['target_type']}] "
                f"| date={edge.get('published_date') or 'unknown'} "
                f"| chunk={edge.get('source_chunk_id') or 'unknown'} "
                f"| evidence={norm_space(edge.get('evidence'))}"
            )
            if used + len(line) + 1 > self.config.max_graph_context_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def retrieve_graph_context(
        self, question: str, max_hops: int = 2, edge_limit: int = 50
    ) -> dict[str, Any]:
        seeds = self.match_seeds(question)
        if not seeds:
            return {
                "context": "",
                "edges": pd.DataFrame(),
                "diagnostics": {"reason": "NO_SEED", "supernode_events": []},
            }
        frontier = deque((item["id"], 0) for item in seeds)
        expanded: set[str] = set()
        seen_edges: set[tuple[Any, ...]] = set()
        collected: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        while frontier and len(collected) < self.config.global_edge_cap:
            node_id, hop = frontier.popleft()
            if node_id in expanded or hop >= max_hops:
                continue
            expanded.add(node_id)
            degree = self.store.node_degree(node_id, self.config.dataset_id)
            limit = int(edge_limit)
            if degree > self.config.super_node_degree:
                limit = min(limit, self.config.super_node_edge_cap)
                events.append({"node_id": node_id, "degree": degree, "limit": limit})
            for edge in self.store.recent_edges(node_id, limit, self.config.dataset_id):
                key = (
                    edge["source_id"],
                    edge["relation"],
                    edge["target_id"],
                    edge["source_chunk_id"],
                )
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                collected.append(edge)
                if len(collected) >= self.config.global_edge_cap:
                    break
                neighbor = edge.get("neighbor_id")
                if neighbor and neighbor not in expanded and hop + 1 < max_hops:
                    frontier.append((neighbor, hop + 1))
        return {
            "context": self.textualize(collected),
            "edges": pd.DataFrame(collected),
            "diagnostics": {
                "matched_seeds": seeds,
                "expanded_nodes": len(expanded),
                "collected_edges": len(collected),
                "supernode_events": events,
            },
        }

    def hybrid_context(self, question: str, max_hops: int = 2) -> dict[str, Any]:
        graph = self.retrieve_graph_context(question, max_hops=max_hops, edge_limit=50)
        vector_context, vector_docs = self.flat_index.retrieve(question, k=4)
        return {
            "context": f"=== GRAPH ===\n{graph['context']}\n\n=== VECTOR ===\n{vector_context}",
            "graph": graph,
            "vector_docs": vector_docs,
        }


ANSWER_SYSTEM = """
Answer only from the supplied context. Be concise but complete. Cite provenance
inline as [chunk_id=...] for factual claims. If evidence is insufficient or
conflicting, say so instead of guessing. A citation must point to the exact
context item supporting that claim. For a multi-hop answer, cite every edge in
the reasoning chain; never reuse an investment chunk to support a development
claim or vice versa.
""".strip()


def generate_answer(question: str, context: str, client: ChatClient) -> dict[str, Any]:
    started = time.perf_counter()
    text, usage = client.chat(
        ANSWER_SYSTEM,
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:",
        max_tokens=900,
    )
    return {
        "answer": text.strip(),
        "latency_s": time.perf_counter() - started,
        "total_tokens": usage.get("total_tokens"),
        "provider": usage.get("provider"),
    }


def answer_flat_rag(
    question: str, flat_index: FlatRAGIndex, client: ChatClient
) -> dict[str, Any]:
    context, retrieved = flat_index.retrieve(question, k=6)
    result = generate_answer(question, context, client)
    result.update({"context": context, "retrieved": retrieved})
    return result


def answer_graph_rag(
    question: str, retriever: HybridRetriever, client: ChatClient
) -> dict[str, Any]:
    hybrid = retriever.hybrid_context(question, max_hops=2)
    result = generate_answer(question, hybrid["context"], client)
    edges = hybrid["graph"]["edges"]
    if not edges.empty:
        combined = norm_entity(f"{question} {result['answer']}")
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
        citations: list[str] = []
        ordered = edges.sort_values("published_date", ascending=False)
        for edge in ordered.itertuples(index=False):
            source_in_text = norm_entity(edge.source_name) in combined
            target_in_text = norm_entity(edge.target_name) in combined
            edge_year = norm_space(edge.published_date)[:4]
            if not (source_in_text and target_in_text):
                continue
            if years and edge_year not in years:
                continue
            chunk_id = norm_space(edge.source_chunk_id)
            if chunk_id and chunk_id not in citations:
                citations.append(chunk_id)
            if len(citations) >= 6:
                break
        if citations:
            clean_answer = re.sub(r"\s*\[chunk_id=[^\]]+\]", "", result["answer"])
            evidence_chain = " ".join(f"[chunk_id={chunk_id}]" for chunk_id in citations)
            result["answer"] = f"{clean_answer.rstrip()} Evidence chain: {evidence_chain}"
    result.update(
        {
            "context": hybrid["context"],
            "graph_debug": hybrid["graph"],
            "vector_docs": hybrid["vector_docs"],
        }
    )
    return result


JUDGE_SYSTEM = """
You are a strict RAG evaluator. Use the reference as the correctness anchor.
Score 1-5 for comprehensiveness, faithfulness to the candidate context, and
multi-hop reasoning accuracy. Return strict JSON with a concise rationale.
""".strip()


def judge_answer(
    question: str,
    reference: str,
    answer: str,
    context: str,
    client: ChatClient,
) -> dict[str, Any]:
    obj, usage = client.json(
        JUDGE_SYSTEM,
        f"""
QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {answer}
CANDIDATE CONTEXT: {context[:18000]}
Return {{"comprehensiveness":1,"faithfulness":1,
"multi_hop_reasoning":1,"rationale":"2-5 sentences"}}.
""".strip(),
        max_tokens=600,
    )
    score_keys = ("comprehensiveness", "faithfulness", "multi_hop_reasoning")
    output: dict[str, Any] = {}
    for key in score_keys:
        try:
            output[key] = max(1, min(5, int(obj.get(key, 1))))
        except (TypeError, ValueError):
            output[key] = 1
    output["rationale"] = norm_space(obj.get("rationale"))
    output["judge_tokens"] = usage.get("total_tokens")
    anchor = norm_space(reference.split(";")[0]).lower()
    evidence_lines = [line for line in context.splitlines() if anchor and anchor in line.lower()]
    if min(output[key] for key in score_keys) <= 2 and anchor in answer.lower() and evidence_lines:
        verified, verify_usage = client.json(
            JUDGE_SYSTEM,
            f"""Re-check an internally inconsistent first-pass evaluation.
QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {answer}
FIRST PASS: {json.dumps(obj, ensure_ascii=False)}
CONTEXT LINES CONTAINING THE REFERENCE ANCHOR:
{chr(10).join(evidence_lines[:8])}
Full context: {context[:18000]}
Return the same score/rationale JSON schema. Penalize an incorrect inline
citation, but do not claim evidence is absent when it appears above.""",
            max_tokens=600,
        )
        for key in score_keys:
            try:
                output[key] = max(1, min(5, int(verified.get(key, output[key]))))
            except (TypeError, ValueError):
                pass
        output["rationale"] = norm_space(verified.get("rationale"))
        output["judge_tokens"] = (output.get("judge_tokens") or 0) + (
            verify_usage.get("total_tokens") or 0
        )
        output["second_pass"] = True
    else:
        output["second_pass"] = False
    return output


def validate_golden(frame: pd.DataFrame) -> None:
    required = {"id", "group", "question", "reference_answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Golden dataset is missing columns: {sorted(missing)}")
    invalid_groups = set(frame["group"]) - {"factoid", "multi-hop", "cross-doc"}
    if invalid_groups:
        raise ValueError(f"Invalid golden groups: {sorted(invalid_groups)}")
    if set(frame["group"]) != {"factoid", "multi-hop", "cross-doc"}:
        raise ValueError("Golden dataset must contain all three required groups")
    empty = frame["reference_answer"].fillna("").astype(str).str.strip().eq("")
    if empty.any():
        raise ValueError("Golden dataset contains blank reference answers")


def run_evaluation(
    golden: pd.DataFrame,
    flat_index: FlatRAGIndex,
    retriever: HybridRetriever,
    client: ChatClient,
    checkpoint: Path,
) -> pd.DataFrame:
    validate_golden(golden)
    rows: list[dict[str, Any]] = []
    if checkpoint.exists():
        existing = pd.read_csv(checkpoint)
        rows = existing.to_dict("records")
    done = {str(row["id"]) for row in rows}
    for question in tqdm(golden.itertuples(index=False), total=len(golden), desc="Evaluation"):
        if str(question.id) in done:
            continue
        flat = answer_flat_rag(question.question, flat_index, client)
        graph = answer_graph_rag(question.question, retriever, client)
        flat_judge = judge_answer(
            question.question, question.reference_answer, flat["answer"], flat["context"], client
        )
        graph_judge = judge_answer(
            question.question, question.reference_answer, graph["answer"], graph["context"], client
        )
        rows.append(
            {
                "id": question.id,
                "group": question.group,
                "question": question.question,
                "reference_answer": question.reference_answer,
                "flat_answer": flat["answer"],
                "graph_answer": graph["answer"],
                "flat_comprehensiveness": flat_judge["comprehensiveness"],
                "graph_comprehensiveness": graph_judge["comprehensiveness"],
                "flat_faithfulness": flat_judge["faithfulness"],
                "graph_faithfulness": graph_judge["faithfulness"],
                "flat_multi_hop_reasoning": flat_judge["multi_hop_reasoning"],
                "graph_multi_hop_reasoning": graph_judge["multi_hop_reasoning"],
                "flat_latency_s": flat["latency_s"],
                "graph_latency_s": graph["latency_s"],
                "flat_total_tokens": flat.get("total_tokens"),
                "graph_total_tokens": graph.get("total_tokens"),
                "flat_provider": flat.get("provider"),
                "graph_provider": graph.get("provider"),
                "flat_judge_rationale": flat_judge["rationale"],
                "graph_judge_rationale": graph_judge["rationale"],
                "flat_judge_second_pass": flat_judge.get("second_pass", False),
                "graph_judge_second_pass": graph_judge.get("second_pass", False),
                "graph_supernode_events": len(
                    graph["graph_debug"]["diagnostics"].get("supernode_events", [])
                ),
            }
        )
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
    return pd.DataFrame(rows)


def comparison_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "Comprehensiveness": ("flat_comprehensiveness", "graph_comprehensiveness"),
        "Faithfulness": ("flat_faithfulness", "graph_faithfulness"),
        "Multi-hop reasoning": ("flat_multi_hop_reasoning", "graph_multi_hop_reasoning"),
        "Latency (s)": ("flat_latency_s", "graph_latency_s"),
        "Token usage": ("flat_total_tokens", "graph_total_tokens"),
    }
    rows: list[dict[str, Any]] = []
    grouped = list(evaluation.groupby("group")) + [("overall", evaluation)]
    for group, frame in grouped:
        for metric, (flat_column, graph_column) in metric_map.items():
            flat_value = pd.to_numeric(frame[flat_column], errors="coerce").mean()
            graph_value = pd.to_numeric(frame[graph_column], errors="coerce").mean()
            delta = graph_value - flat_value
            if metric in {"Latency (s)", "Token usage"}:
                comment = (
                    "Flat RAG is cheaper/faster in this run."
                    if flat_value < graph_value
                    else "GraphRAG was not more expensive in this sample."
                )
            elif delta >= 0.75:
                comment = "GraphRAG improved this quality metric materially."
            elif delta <= -0.5:
                comment = "Flat RAG scored higher; inspect graph extraction/retrieval noise."
            else:
                comment = "The methods were close on this metric."
            rows.append(
                {
                    "Question group": group,
                    "Metric": metric,
                    "Flat RAG": round(float(flat_value), 3) if pd.notna(flat_value) else np.nan,
                    "GraphRAG": round(float(graph_value), 3) if pd.notna(graph_value) else np.nan,
                    "Delta (Graph-Flat)": round(float(delta), 3) if pd.notna(delta) else np.nan,
                    "Analysis": comment,
                }
            )
    return pd.DataFrame(rows)


def test_supernode_policy(
    retriever: HybridRetriever, node_id: str | None = None
) -> dict[str, Any]:
    if node_id is None:
        _, top = retriever.store.graph_checks(retriever.config.dataset_id)
        if top.empty:
            return {"status": "EMPTY_GRAPH"}
        node_id = str(top.iloc[0]["id"])
    degree = retriever.store.node_degree(node_id, retriever.config.dataset_id)
    limit = (
        retriever.config.super_node_edge_cap
        if degree > retriever.config.super_node_degree
        else 1000
    )
    edges = retriever.store.recent_edges(node_id, limit, retriever.config.dataset_id)
    if degree > retriever.config.super_node_degree and len(edges) > retriever.config.super_node_edge_cap:
        raise AssertionError("Super-node edge cap was not enforced")
    return {"status": "PASS", "node_id": node_id, "degree": degree, "fetched": len(edges)}


def build_communities(
    store: Neo4jGraphStore, config: LabConfig, limit_edges: int = 20_000
) -> pd.DataFrame:
    edges = store.all_edges(config.dataset_id, limit_edges)
    if edges.empty:
        return pd.DataFrame(columns=["id", "community_id"])
    graph = nx.Graph()
    graph.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
    communities = nx.algorithms.community.greedy_modularity_communities(graph)
    rows = [
        {"id": node_id, "community_id": int(community_id)}
        for community_id, members in enumerate(communities)
        for node_id in members
    ]
    store.write_communities(rows, config.dataset_id)
    return pd.DataFrame(rows)


def community_reports(
    communities: pd.DataFrame, nodes: pd.DataFrame, triples: pd.DataFrame
) -> pd.DataFrame:
    if communities.empty:
        return pd.DataFrame(columns=["community_id", "entity_count", "summary"])
    joined = communities.merge(nodes[["id", "name"]], on="id", how="left")
    reports = []
    for community_id, group in joined.groupby("community_id"):
        names = sorted(group["name"].dropna().astype(str))
        related = triples[
            triples.source_id.isin(group.id) | triples.target_id.isin(group.id)
        ]
        relations = related.relation.value_counts().to_dict()
        reports.append(
            {
                "community_id": int(community_id),
                "entity_count": len(group),
                "entities": " | ".join(names[:20]),
                "dominant_relations": json.dumps(relations, ensure_ascii=False),
                "summary": f"Community {community_id} contains {len(group)} entities; "
                f"key entities: {', '.join(names[:8])}. Relation mix: {relations}.",
            }
        )
    return pd.DataFrame(reports)


def global_community_search(
    question: str, reports: pd.DataFrame, client: ChatClient
) -> dict[str, Any]:
    if reports.empty:
        return {"answer": "No community reports are available.", "latency_s": 0.0, "total_tokens": 0}
    context = "\n".join(
        f"[community={row.community_id}] {row.summary}"
        for row in reports.itertuples(index=False)
    )
    started = time.perf_counter()
    text, usage = client.chat(
        """Answer only from the supplied community reports. Cite supporting
community reports as [community_id=...]. Never cite a chunk id because this
context contains aggregate reports rather than source chunks. If the reports
are insufficient, state the limitation.""",
        f"QUESTION:\n{question}\n\nCOMMUNITY REPORTS:\n{context}\n\nANSWER:",
        max_tokens=900,
    )
    text = re.sub(r"\bCommunity\s+(\d+)\b", r"[community_id=\1]", text, flags=re.I)
    return {
        "answer": text.strip(),
        "latency_s": time.perf_counter() - started,
        "total_tokens": usage.get("total_tokens"),
    }


def self_correcting_context(
    question: str, retriever: HybridRetriever, client: ChatClient
) -> dict[str, Any]:
    system = "Decide if context is sufficient to answer faithfully. Do not answer. Return JSON."
    hop2 = retriever.retrieve_graph_context(question, max_hops=2)
    obj, _ = client.json(
        system,
        f'QUESTION: {question}\nCONTEXT: {hop2["context"]}\nReturn {{"sufficient":true,"missing":"..."}}',
        max_tokens=250,
    )
    if bool(obj.get("sufficient")):
        return {"route": "hop2", "context": hop2["context"], "missing": ""}
    hop3 = retriever.retrieve_graph_context(question, max_hops=3)
    obj3, _ = client.json(
        system,
        f'QUESTION: {question}\nCONTEXT: {hop3["context"]}\nReturn {{"sufficient":true,"missing":"..."}}',
        max_tokens=250,
    )
    if bool(obj3.get("sufficient")):
        return {"route": "hop3", "context": hop3["context"], "missing": norm_space(obj.get("missing"))}
    vector, _ = retriever.flat_index.retrieve(question, k=8)
    return {
        "route": "hop3+vector",
        "context": f"=== GRAPH ===\n{hop3['context']}\n\n=== VECTOR ===\n{vector}",
        "missing": norm_space(obj3.get("missing")),
    }


def _prioritize_demo(news: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([demo, news], ignore_index=True, sort=False)
    return combined


def run_lab(
    config: LabConfig | None = None,
    *,
    force_download: bool = False,
    force_llm: bool = False,
) -> dict[str, Any]:
    config = config or LabConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    manifest_path = config.output_dir / "run_manifest.json"
    previous_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_manifest = {}
    manifest: dict[str, Any] = {
        "started_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "status": "running",
    }
    client = ChatClient()
    dataset_path, dataset_manifest = stream_hackernoon_dataset(
        config, force=force_download
    )
    manifest["dataset"] = dataset_manifest
    source = load_news(dataset_path)
    demo_path = config.data_dir / "sample_news.csv"
    demo = load_news(demo_path)
    news = standardize_news(_prioritize_demo(source, demo), config)
    news, near_dedup_audit = near_deduplicate(news)
    chunks = build_chunks(news, config)
    if chunks.empty:
        raise RuntimeError("Preprocessing generated no chunks")
    extraction_source = chunks.head(min(config.extraction_max_chunks, len(chunks))).copy()

    coref_checkpoint = config.cache_dir / "coref.csv"
    extraction_checkpoint = config.cache_dir / "raw_triples.csv"
    eval_checkpoint = config.cache_dir / "evaluation.csv"
    if force_llm:
        for path in (
            coref_checkpoint,
            extraction_checkpoint,
            extraction_checkpoint.with_suffix(".processed.json"),
            eval_checkpoint,
        ):
            path.unlink(missing_ok=True)
    coref, coref_usage = run_coref(
        extraction_source, client, checkpoint=coref_checkpoint
    )
    extraction_source = extraction_source.merge(coref, on="chunk_id", how="left")
    raw_triples, extraction_errors, extraction_usage = run_extraction(
        extraction_source, client, checkpoint=extraction_checkpoint
    )
    if raw_triples.empty:
        raise RuntimeError("NER/RE extraction generated no valid triples")
    validate_provenance(raw_triples)
    entity_map, audit = build_resolution_map(
        raw_triples, threshold=config.entity_threshold
    )
    triples = canonicalize_triples(raw_triples, entity_map)
    validate_provenance(triples)
    nodes = build_nodes(triples, config.dataset_id)

    store, graph_manifest = Neo4jGraphStore.from_environment(allow_local_fallback=True)
    manifest["graph"] = graph_manifest
    store.setup_schema()
    store.clear_dataset(config.dataset_id)
    bulk_insert_nodes(store, nodes)
    bulk_insert_edges(store, triples, config.dataset_id)
    graph_counts, top_degree = store.graph_checks(config.dataset_id)

    flat_index = FlatRAGIndex(chunks)
    retriever = HybridRetriever(store, nodes, flat_index, client, config)
    supernode_test = test_supernode_policy(retriever)

    golden_path = Path(
        os.getenv("GOLDEN_PATH", str(config.data_dir / "golden_dataset.csv"))
    )
    golden = pd.read_csv(golden_path)
    manifest["golden_dataset"] = {"path": str(golden_path), "rows": len(golden)}
    evaluation = run_evaluation(
        golden, flat_index, retriever, client, eval_checkpoint
    )
    summary = comparison_table(evaluation)
    communities = build_communities(store, config)
    reports = community_reports(communities, nodes, triples)

    self_correction_path = config.output_dir / "self_correction_audit.csv"
    if self_correction_path.exists() and not force_llm:
        self_correction_audit = pd.read_csv(self_correction_path)
    else:
        probe = golden[golden["group"] == "multi-hop"].iloc[-1]
        correction = self_correcting_context(probe["question"], retriever, client)
        self_correction_audit = pd.DataFrame(
            [
                {
                    "question_id": probe["id"],
                    "question": probe["question"],
                    "route": correction["route"],
                    "missing": correction["missing"],
                    "context_chars": len(correction["context"]),
                }
            ]
        )
        self_correction_audit.to_csv(self_correction_path, index=False)

    global_search_path = config.output_dir / "global_community_search.csv"
    if global_search_path.exists() and not force_llm:
        global_search_audit = pd.read_csv(global_search_path)
    else:
        global_question = "What are the main investment and AI-development themes across the graph?"
        global_result = global_community_search(global_question, reports, client)
        global_search_audit = pd.DataFrame(
            [
                {
                    "question": global_question,
                    "answer": global_result["answer"],
                    "latency_s": global_result["latency_s"],
                    "total_tokens": global_result["total_tokens"],
                    "community_count": len(reports),
                }
            ]
        )
        global_search_audit.to_csv(global_search_path, index=False)

    artifacts: dict[str, pd.DataFrame] = {
        "chunks.csv": chunks,
        "coreference_audit.csv": coref,
        "extraction_errors.csv": extraction_errors,
        "raw_triples.csv": raw_triples,
        "canonical_triples.csv": triples,
        "entity_resolution_audit.csv": audit,
        "near_dedup_audit.csv": near_dedup_audit,
        "top_degree_nodes.csv": top_degree,
        "graphrag_eval_results.csv": evaluation,
        "graphrag_vs_flatrag_summary.csv": summary,
        "community_reports.csv": reports,
        "self_correction_audit.csv": self_correction_audit,
        "global_community_search.csv": global_search_audit,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(config.output_dir / filename, index=False)
    # Rubric.md mentions reports/*.csv while README.md requires outputs/*.csv.
    # Export the two benchmark tables to both locations to satisfy both documents.
    evaluation.to_csv(config.report_dir / "graphrag_eval_results.csv", index=False)
    summary.to_csv(config.report_dir / "graphrag_vs_flatrag_summary.csv", index=False)

    current_llm_events = Counter(
        f"{event['provider']}:{'ok' if event['ok'] else event.get('error_type', 'error')}"
        for event in client.events
    )
    prior_llm = previous_manifest.get("llm", {})
    materialization_events = (
        current_llm_events
        or prior_llm.get("materialization_events")
        or prior_llm.get("current_run_events")
        or prior_llm.get("events")
        or {}
    )
    manifest.update(
        {
            "status": "complete",
            "finished_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "counts": {
                "articles": len(news),
                "chunks": len(chunks),
                "extraction_chunks": len(extraction_source),
                "raw_triples": len(raw_triples),
                "canonical_triples": len(triples),
                "entity_audit_rows": len(audit),
                **graph_counts,
            },
            "supernode_test": supernode_test,
            "bonus": {
                "near_dedup_dropped": len(near_dedup_audit),
                "community_count": len(reports),
                "self_correction_route": self_correction_audit.iloc[0]["route"],
                "global_search_completed": not global_search_audit.empty,
            },
            "llm": {
                "current_run_events": current_llm_events,
                "materialization_events": materialization_events,
                "coref_calls": len(coref_usage),
                "extraction_calls": len(extraction_usage),
                "evaluation_cache_hit": not bool(current_llm_events) and eval_checkpoint.exists(),
            },
            "artifacts": sorted(artifacts),
        }
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
    store.close()
    return {
        "manifest": manifest,
        "news_df": news,
        "chunks_df": chunks,
        "coref_df": coref,
        "raw_triples_df": raw_triples,
        "triples_df": triples,
        "nodes_df": nodes,
        "entity_resolution_audit_df": audit,
        "top_degree_df": top_degree,
        "eval_results_df": evaluation,
        "comparison_df": summary,
        "community_reports_df": reports,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Day 19 GraphRAG lab")
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--extraction-chunks", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-llm", action="store_true")
    args = parser.parse_args()
    config = LabConfig()
    if args.max_articles is not None:
        config.lab_max_articles = args.max_articles
    if args.max_chunks is not None:
        config.lab_max_chunks = args.max_chunks
    if args.extraction_chunks is not None:
        config.extraction_max_chunks = args.extraction_chunks
    result = run_lab(config, force_download=args.force_download, force_llm=args.force_llm)
    print(json.dumps(result["manifest"]["counts"], indent=2))


if __name__ == "__main__":
    _cli()
