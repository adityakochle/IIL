"""Airflow DAG: Daily document ingestion pipeline.

Schedule: Daily at 2:00 AM PST
Tasks:
  1. scan_new_docs    — Check for new/modified documents
  2. load_and_chunk   — Load documents and create parent-child nodes
  3. embed_and_index  — Embed child nodes and upsert to Qdrant
  4. validate_index   — Verify index integrity (count + spot-check)
  5. notify_complete  — Log completion summary

Usage:
  Requires OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY in Airflow Variables or env.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

# Airflow imports — only available when running inside Airflow
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

log = logging.getLogger(__name__)

# ─── Task functions ──────────────────────────────────────────────────────────

def task_scan_new_docs(**context) -> dict:
    """Scan the data directory for new or modified documents since last run."""
    from config import settings

    data_dir = settings.synthetic_dir
    last_run_file = settings.data_dir / ".last_ingestion_run"

    last_run = datetime.min
    if last_run_file.exists():
        ts = last_run_file.read_text().strip()
        last_run = datetime.fromisoformat(ts)

    new_files = []
    for f in data_dir.rglob("*"):
        if f.is_file() and f.suffix in {".md", ".pdf", ".txt", ".json"}:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime > last_run:
                new_files.append(str(f))

    log.info(f"Found {len(new_files)} new/modified files since {last_run}")
    context["ti"].xcom_push(key="new_files", value=new_files)
    context["ti"].xcom_push(key="total_files", value=len(list(data_dir.rglob("*.md"))))

    return {"new_files": len(new_files)}


def task_load_and_chunk(**context) -> dict:
    """Load all documents and create parent-child node pairs."""
    from pipelines.loaders import load_all_documents
    from pipelines.chunkers import create_parent_child_nodes

    log.info("Loading documents...")
    documents = load_all_documents()

    log.info(f"Chunking {len(documents)} documents...")
    child_nodes, parent_store = create_parent_child_nodes(documents)

    # Store node count in XCom for downstream tasks
    context["ti"].xcom_push(key="child_node_count", value=len(child_nodes))
    context["ti"].xcom_push(key="parent_count", value=len(parent_store))

    log.info(f"Created {len(child_nodes)} child nodes from {len(parent_store)} parents")
    return {"child_nodes": len(child_nodes), "parents": len(parent_store)}


def task_embed_and_index(**context) -> dict:
    """Embed child nodes and upsert into Qdrant collection."""
    from pipelines.loaders import load_all_documents
    from pipelines.chunkers import create_parent_child_nodes
    from pipelines.indexer import index_nodes, ensure_collection, _get_qdrant_client

    log.info("Re-loading documents for embedding...")
    documents = load_all_documents()
    child_nodes, _ = create_parent_child_nodes(documents)

    log.info(f"Indexing {len(child_nodes)} nodes into Qdrant...")
    index = index_nodes(child_nodes)

    # Verify index was created
    client = _get_qdrant_client()
    collection_info = client.get_collection(
        collection_name=os.environ.get("QDRANT_COLLECTION", "sfsu_policies")
    )
    vector_count = collection_info.vectors_count or 0

    context["ti"].xcom_push(key="indexed_vectors", value=vector_count)
    log.info(f"Qdrant collection now contains {vector_count} vectors")
    return {"indexed_vectors": vector_count}


def task_validate_index(**context) -> dict:
    """Spot-check the index with a known test query."""
    from pipelines.indexer import load_existing_index
    from rag.retriever import ParentDocumentRetriever

    test_queries = [
        "transfer credit maximum units CSU",
        "CS major prerequisites calculus",
        "graduation residency requirement",
    ]

    log.info("Validating index with spot-check queries...")
    index = load_existing_index()
    retriever = ParentDocumentRetriever(index)

    results = {}
    for q in test_queries:
        contexts = retriever.retrieve(q)
        results[q] = len(contexts)
        log.info(f"  '{q[:40]}...' → {len(contexts)} contexts")

    all_passed = all(v > 0 for v in results.values())
    if not all_passed:
        raise ValueError(f"Validation failed: some queries returned no results. {results}")

    log.info("✅ Index validation passed")
    return {"validation": results, "passed": all_passed}


def task_notify_complete(**context) -> None:
    """Log ingestion completion summary."""
    ti = context["ti"]
    child_count = ti.xcom_pull(task_ids="load_and_chunk", key="child_node_count") or 0
    vector_count = ti.xcom_pull(task_ids="embed_and_index", key="indexed_vectors") or 0

    from config import settings
    last_run_file = settings.data_dir / ".last_ingestion_run"
    last_run_file.write_text(datetime.now().isoformat())

    summary = {
        "run_date": datetime.now().isoformat(),
        "child_nodes_processed": child_count,
        "vectors_in_qdrant": vector_count,
        "status": "SUCCESS",
    }
    log.info(f"Ingestion complete: {json.dumps(summary, indent=2)}")


# ─── DAG definition ───────────────────────────────────────────────────────────

if _AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "iil-system",
        "depends_on_past": False,
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="iil_document_ingestion",
        default_args=default_args,
        description="Daily ingestion pipeline: load SFSU policy docs → chunk → embed → index in Qdrant",
        schedule_interval="0 2 * * *",  # 2:00 AM daily
        start_date=days_ago(1),
        catchup=False,
        tags=["iil", "ingestion", "rag"],
    ) as dag:

        scan = PythonOperator(
            task_id="scan_new_docs",
            python_callable=task_scan_new_docs,
        )

        load_chunk = PythonOperator(
            task_id="load_and_chunk",
            python_callable=task_load_and_chunk,
        )

        embed_index = PythonOperator(
            task_id="embed_and_index",
            python_callable=task_embed_and_index,
        )

        validate = PythonOperator(
            task_id="validate_index",
            python_callable=task_validate_index,
        )

        notify = PythonOperator(
            task_id="notify_complete",
            python_callable=task_notify_complete,
        )

        # Task dependency chain
        scan >> load_chunk >> embed_index >> validate >> notify
