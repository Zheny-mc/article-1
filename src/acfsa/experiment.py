from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .context import build_context
from .embeddings import E5Embedder
from .filtering import filter_candidates, remove_exact_duplicates
from .loader import load_documents
from .llm import generate_answer
from .metrics import mean_pairwise_similarity, retrieval_metrics, token_count
from .mmr import mmr_select
from .retrieval import retrieve

SUPPORTED_METHODS = ("baseline", "adaptive", "mmr", "acfsa")


def _load_questions(path: str | Path) -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if line.strip():
                item = json.loads(line)
                if "question" not in item or "relevant_doc_ids" not in item:
                    raise ValueError(f"Invalid QA record at line {line_no}: expected question and relevant_doc_ids")
                questions.append(item)
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def _summary_plot(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("context_tokens", "Число токенов в контексте", "tokens_comparison.png"),
        ("recall_at_k", "Recall@K", "recall_comparison.png"),
        ("ndcg_at_k", "nDCG@K", "ndcg_comparison.png"),
        ("latency_ms", "Время обработки (мс)", "latency_comparison.png"),
        ("context_reduction_vs_baseline", "Сокращение контекста относительно Baseline", "compression_comparison.png"),
    ]
    method_labels = {"baseline": "Baseline", "adaptive": "Adaptive", "mmr": "MMR", "acfsa": "ACFSA"}
    methods = [method for method in SUPPORTED_METHODS if method in set(df["method"])]
    for column, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        groups = [df.loc[df["method"] == method, column].dropna().values for method in methods]
        ax.boxplot(groups, labels=[method_labels[m] for m in methods], showmeans=True)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ": сравнение методов")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def run_experiment(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    seed = int(config.get("project", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    root = config_path.parent.parent
    resolve = lambda value: (root / value).resolve()
    corpus_cfg = config["corpus"]
    chunks = load_documents(
        resolve(corpus_cfg["directory"]),
        max_chars=int(corpus_cfg.get("max_chars", 1400)),
        overlap_chars=int(corpus_cfg.get("overlap_chars", 180)),
    )
    questions = _load_questions(resolve(config["experiment"]["questions_file"]))

    llm_cfg = config.get("llm", {})
    llm_client = None
    llm_model = None
    if llm_cfg.get("enabled", False):
        try:
            from dotenv import load_dotenv
            load_dotenv(root / ".env")
        except ImportError:
            pass
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to enable answer generation: pip install openai") from exc
        key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
        model_env = llm_cfg.get("model_env", "LLM_MODEL")
        base_url_env = llm_cfg.get("base_url_env", "OPENAI_BASE_URL")
        api_key = os.environ.get(key_env)
        llm_model = os.environ.get(model_env)
        base_url = os.environ.get(base_url_env)
        if not api_key or not llm_model:
            raise RuntimeError(f"LLM enabled, but {key_env} and/or {model_env} environment variables are missing")
        llm_client = OpenAI(api_key=api_key, base_url=base_url or None)

    embed_cfg = config["embedding"]
    embedder = E5Embedder(embed_cfg["model_name"], embed_cfg.get("device"))
    embedding_started = time.perf_counter()
    chunk_embeddings = embedder.encode_passages([chunk.text for chunk in chunks])
    embedding_seconds = time.perf_counter() - embedding_started

    retrieval_cfg = config["retrieval"]
    top_n = int(retrieval_cfg.get("top_n", 20))
    top_k = int(retrieval_cfg.get("top_k", 5))
    acfsa_cfg = config["acfsa"]
    methods = config["experiment"].get("methods", list(SUPPORTED_METHODS))
    invalid_methods = sorted(set(methods) - set(SUPPORTED_METHODS))
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {invalid_methods}; choose from {SUPPORTED_METHODS}")

    output_dir = resolve(config["experiment"].get("output_dir", "outputs/latest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_name = config["experiment"].get("tokenizer", "cl100k_base")

    rows: list[dict] = []
    audit_rows: list[dict] = []
    answer_rows: list[dict] = []
    for q_index, item in enumerate(questions, start=1):
        query = item["question"]
        relevant = set(item["relevant_doc_ids"])

        started = time.perf_counter()
        retrieved = retrieve(query, chunks, chunk_embeddings, embedder, top_n=top_n)
        retrieve_ms = (time.perf_counter() - started) * 1000

        method_audits = {}
        for method in methods:
            started = time.perf_counter()
            filter_audit = None
            if method == "baseline":
                selected = retrieved[:top_k]
            elif method == "adaptive":
                filtered, filter_audit = filter_candidates(
                    retrieved, beta=float(acfsa_cfg.get("beta", 1.0)),
                    min_threshold=float(acfsa_cfg.get("min_threshold", 0.60)),
                )
                selected = filtered[:top_k]
            elif method == "mmr":
                unique = remove_exact_duplicates(retrieved)
                selected = mmr_select(unique, k=top_k,
                                      lambda_param=float(acfsa_cfg.get("lambda_param", 0.70)))
            else:  # full ACFSA
                filtered, filter_audit = filter_candidates(
                    retrieved, beta=float(acfsa_cfg.get("beta", 1.0)),
                    min_threshold=float(acfsa_cfg.get("min_threshold", 0.60)),
                )
                filtered = remove_exact_duplicates(filtered)
                selected = mmr_select(filtered, k=top_k,
                                      lambda_param=float(acfsa_cfg.get("lambda_param", 0.70)))
            processing_ms = (time.perf_counter() - started) * 1000

            context = build_context(selected)
            answer = None
            llm_ms = 0.0
            if llm_client is not None:
                llm_started = time.perf_counter()
                answer = generate_answer(llm_client, llm_model, query, context,
                                         temperature=float(llm_cfg.get("temperature", 0.0)))
                llm_ms = (time.perf_counter() - llm_started) * 1000
                answer_rows.append({
                    "query_id": q_index, "question": query,
                    "reference_answer": item.get("reference_answer", ""),
                    "method": method, "answer": answer,
                    "expert_factuality_0_4": "", "expert_completeness_0_4": "",
                    "expert_grounding_0_4": "", "expert_unsupported_claims_0_4": "",
                })

            retrieval_stats = retrieval_metrics(selected, relevant, top_k)
            row = {
                "query_id": q_index, "method": method, "question": query,
                **retrieval_stats,
                "context_tokens": token_count(context, tokenizer_name),
                "selected_chunks": len(selected),
                "unique_sources": len({c.chunk.doc_id for c in selected}),
                "mean_pairwise_similarity": mean_pairwise_similarity(selected),
                "retrieval_ms": retrieve_ms,
                "filter_mmr_ms": processing_ms,
                "llm_ms": llm_ms,
                "latency_ms": retrieve_ms + processing_ms + llm_ms,
                "threshold": filter_audit["threshold"] if filter_audit else None,
                "answer": answer,
            }
            rows.append(row)
            method_audits[method] = {
                "filter": filter_audit,
                "selected": [
                    {"chunk_id": c.chunk.chunk_id, "doc_id": c.chunk.doc_id,
                     "relevance": c.relevance, "mmr_score": c.mmr_score}
                    for c in selected
                ],
            }

        audit_rows.append({
            "query_id": q_index,
            "question": query,
            "retrieved": [
                {"chunk_id": c.chunk.chunk_id, "doc_id": c.chunk.doc_id, "relevance": c.relevance}
                for c in retrieved
            ],
            "methods": method_audits,
        })

    results = pd.DataFrame(rows)
    # Per-query context compression relative to the baseline run.
    if "baseline" in set(results["method"]):
        baseline_tokens = results.loc[results["method"] == "baseline"].set_index("query_id")["context_tokens"]
        reductions = []
        for row in results.itertuples():
            baseline_count = baseline_tokens.get(row.query_id, np.nan)
            if pd.notna(baseline_count) and baseline_count > 0:
                reductions.append(1.0 - float(row.context_tokens) / float(baseline_count))
            else:
                reductions.append(np.nan)
        results["context_reduction_vs_baseline"] = reductions
    else:
        results["context_reduction_vs_baseline"] = np.nan

    results.to_csv(output_dir / "results_per_query.csv", index=False, encoding="utf-8-sig")
    summary = results.groupby("method").agg({
        "precision_at_k": "mean", "recall_at_k": "mean", "mrr": "mean", "ndcg_at_k": "mean",
        "context_tokens": ["mean", "median", "std"], "selected_chunks": "mean",
        "unique_sources": "mean", "mean_pairwise_similarity": "mean", "latency_ms": "mean",
        "context_reduction_vs_baseline": ["mean", "std"],
    })
    summary.to_csv(output_dir / "summary.csv", encoding="utf-8-sig")
    with open(output_dir / "audit.json", "w", encoding="utf-8") as stream:
        json.dump(audit_rows, stream, ensure_ascii=False, indent=2)
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as stream:
        json.dump({
            "seed": seed,
            "embedding_model": embed_cfg["model_name"],
            "documents": len({chunk.doc_id for chunk in chunks}),
            "chunks": len(chunks), "questions": len(questions),
            "top_n": top_n, "top_k": top_k,
            "retrieval_metric_granularity": "unique documents (first occurrence per doc_id)",
            "methods": methods,
            "acfsa": acfsa_cfg,
            "embedding_seconds": embedding_seconds,
            "llm_model": llm_model,
        }, stream, ensure_ascii=False, indent=2)
    if answer_rows:
        pd.DataFrame(answer_rows).to_csv(output_dir / "answers_for_blind_review.csv", index=False, encoding="utf-8-sig")
    _summary_plot(results, output_dir)
    return output_dir
